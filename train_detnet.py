import argparse
import os
import time
import csv
import json
import matplotlib.pyplot as plt
import gc
import shutil
import hashlib

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from progress.bar import Bar
from tqdm import tqdm

import losses as losses
import utils.misc as misc
from datasets.egodexter import EgoDexter
from datasets.handataset import HandDataset
from model.detnet import detnet
from utils import func, align
from utils.eval.evalutils import AverageMeter, accuracy_heatmap
from utils.eval.zimeval import EvalUtil
from BioMC.BMCLoss import BMCLoss
import BioMC.config_bmc_loss as cfg_bmc

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
DEBUG = 0

def compute_mpjpe_torch(pred, target):
    return torch.norm(pred - target, dim=-1).mean()

def compute_mpjpe_numpy(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f"MPJPE shape mismatch: pred shape = {pred.shape}, target shape = {target.shape}")
    return np.linalg.norm(pred - target, axis=-1).mean()

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_dict_list_to_csv(rows, csv_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def plot_curve(x, y, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_learning_curves(curve_dir, loss_all, train_mpjpe_all, auc_all, acc_hm_all, mpjpe_all, use_bmc=False):
    ensure_dir(curve_dir)

    # 1) training losses
    if len(loss_all["lossH"]) > 0:
        epochs = list(range(1, len(loss_all["lossH"]) + 1))
        plot_curve(epochs, loss_all["lossH"],
                   "Training LossH Curve", "Epoch", "LossH",
                   os.path.join(curve_dir, "train_lossH_curve.png"))
        plot_curve(epochs, loss_all["lossD"],
                   "Training LossD Curve", "Epoch", "LossD",
                   os.path.join(curve_dir, "train_lossD_curve.png"))
        plot_curve(epochs, loss_all["lossL"],
                   "Training LossL Curve", "Epoch", "LossL",
                   os.path.join(curve_dir, "train_lossL_curve.png"))

        # ---- BMC loss curves (optional) ----
        if use_bmc and "bmc_total" in loss_all and len(loss_all["bmc_total"]) > 0:
            plot_curve(epochs, loss_all["bmc_total"],
                       "Training BMC Total Loss Curve", "Epoch", "BMC Total Loss",
                       os.path.join(curve_dir, "train_bmc_total_curve.png"))
            plot_curve(epochs, loss_all["bmc_bl"],
                       "Training BMC Bone Length Loss Curve", "Epoch", "BMC Bone Length Loss",
                       os.path.join(curve_dir, "train_bmc_bl_curve.png"))
            plot_curve(epochs, loss_all["bmc_rb"],
                       "Training BMC Root Bone Loss Curve", "Epoch", "BMC Root Bone Loss",
                       os.path.join(curve_dir, "train_bmc_rb_curve.png"))
            plot_curve(epochs, loss_all["bmc_ja"],
                       "Training BMC Joint Angle Loss Curve", "Epoch", "BMC Joint Angle Loss",
                       os.path.join(curve_dir, "train_bmc_ja_curve.png"))

    # 2) training MPJPE
    if len(train_mpjpe_all) > 0:
        epochs = [x[0] for x in train_mpjpe_all]
        values = [x[1] for x in train_mpjpe_all]
        plot_curve(epochs, values,
                   "Training MPJPE Curve", "Epoch", "MPJPE",
                   os.path.join(curve_dir, "train_mpjpe_curve.png"))

    # 3) testing curves for each dataset
    for key in auc_all.keys():
        if len(auc_all[key]) > 0:
            epochs = [x[0] for x in auc_all[key]]
            values = [x[1] for x in auc_all[key]]
            plot_curve(epochs, values,
                       f"{key} Test AUC Curve", "Epoch", "AUC",
                       os.path.join(curve_dir, f"{key}_test_auc_curve.png"))

        if len(acc_hm_all[key]) > 0:
            epochs = [x[0] for x in acc_hm_all[key]]
            values = [x[1] for x in acc_hm_all[key]]
            plot_curve(epochs, values,
                       f"{key} Test AccHM Curve", "Epoch", "AccHM",
                       os.path.join(curve_dir, f"{key}_test_acchm_curve.png"))

        if len(mpjpe_all[key]) > 0:
            epochs = [x[0] for x in mpjpe_all[key]]
            values = [x[1] for x in mpjpe_all[key]]
            plot_curve(epochs, values,
                       f"{key} Test MPJPE Curve", "Epoch", "MPJPE",
                       os.path.join(curve_dir, f"{key}_test_mpjpe_curve.png"))

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def list_to_str(x):
    if isinstance(x, (list, tuple)):
        return "-".join(map(str, x))
    return str(x)


def build_run_tag(args):
    return (
        f"lr{args.learning_rate}"
        f"_tb{args.train_batch}"
        f"_lres{list_to_str(args.layers_resnet)}"
        f"_bp{list_to_str(args.block_planes_resnet)}"
        f"_in{args.inplanes_resnet}"
        f"_out{args.out_feature_dim_resnet}"
        f"_hid{args.hidden_dim_detnet}"
        f"_ln2d{list_to_str(args.layers_net2d)}"
        f"_st{args.stacks}"
        f"_ep{args.epochs}"
        f"_g{args.gamma}"
        f"_decay{args.lr_decay_step}"
    )

def _short_name(text, max_len=40):
    text = str(text)
    if len(text)<= max_len:
        return text
    
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    keep = max_len-9

    return f"{text[:keep]}_{digest}"

def prepare_experiment_dirs(args):
    """
    统一实验目录:
        exp_dir/
            checkpoints/
            outputs/
            metrics.json
            metrics_row.csv
            config.json
    """
    if args.exp_dir is None or str(args.exp_dir).strip() == "":
        run_tag = build_run_tag(args)
        args.run_name = args.run_name if args.run_name else run_tag

        # short_run_dir = _short_name(args.run_name, max_len = 40)
        parts = args.run_name.split("_")
        short_run_dir = "_".join(parts[:2]) if len(parts)>=2 else args.run_name
        args.exp_dir = os.path.join("experiments", short_run_dir)
    else:
        if args.run_name is None or str(args.run_name).strip() == "":
            args.run_name = os.path.basename(os.path.normpath(args.exp_dir))

    print(f"exp_dir: {args.exp_dir}")

    args.checkpoint = os.path.join(args.exp_dir, "checkpoints")
    args.outpath = os.path.join(args.exp_dir, "outputs")

    print(f"checkpoint: { args.checkpoint}")
    print(f"outpath: {args.outpath}")

    for path in [args.exp_dir, args.checkpoint, args.outpath]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # 若saved_prefix没显式设置成个性化名字，就自动加参数信息
    if args.saved_prefix == "ckp_detnet":
        #args.saved_prefix = f"ckp_detnet_{args.run_name}"
        # short_prefix_name = _short_name(args.run_name, max_len = 30)
        parts = args.run_name.split("_")
        short_prefix_name = "_".join(parts[:2]) if len(parts)>=2 else args.run_name
        args.saved_prefix = f"ckp_detnet_{short_prefix_name}"

def to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    return obj

def save_config(args):
    config_path = os.path.join(args.exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

def save_metrics(args, best_acc, auc_all, acc_hm_all, loss_all, train_mpjpe_all, mpjpe_all):
    metrics = {
        "run_name": args.run_name,
        "exp_dir": args.exp_dir,

        "learning_rate": args.learning_rate,
        "train_batch": args.train_batch,
        "test_batch": args.test_batch,
        "epochs": args.epochs,
        "workers": args.workers,
        "lr_decay_step": args.lr_decay_step,
        "gamma": args.gamma,

        "datasets_train": args.datasets_train,
        "datasets_test": args.datasets_test,

        "layers_resnet": args.layers_resnet,
        "block_planes_resnet": args.block_planes_resnet,
        "inplanes_resnet": args.inplanes_resnet,
        "out_feature_dim_resnet": args.out_feature_dim_resnet,
        "hidden_dim_detnet": args.hidden_dim_detnet,
        "layers_net2d": args.layers_net2d,
        "stacks": args.stacks,

        "final_lossH": loss_all["lossH"][-1] if len(loss_all["lossH"]) > 0 else None,
        "final_lossD": loss_all["lossD"][-1] if len(loss_all["lossD"]) > 0 else None,
        "final_lossL": loss_all["lossL"][-1] if len(loss_all["lossL"]) > 0 else None,

        # ---- BMC metrics (optional) ----
        "final_bmc_total": loss_all["bmc_total"][-1] if ("bmc_total" in loss_all and len(loss_all["bmc_total"]) > 0) else None,
        "final_bmc_bl": loss_all["bmc_bl"][-1] if ("bmc_bl" in loss_all and len(loss_all["bmc_bl"]) > 0) else None,
        "final_bmc_rb": loss_all["bmc_rb"][-1] if ("bmc_rb" in loss_all and len(loss_all["bmc_rb"]) > 0) else None,
        "final_bmc_ja": loss_all["bmc_ja"][-1] if ("bmc_ja" in loss_all and len(loss_all["bmc_ja"]) > 0) else None,

        "last_train_mpjpe": to_python(train_mpjpe_all[-1][1]) if len(train_mpjpe_all) > 0 else None,
        "best_train_mpjpe": to_python(min([x[1] for x in train_mpjpe_all])) if len(train_mpjpe_all) > 0 else None,
    }

    for key in args.datasets_test:
        metrics[f"best_auc_{key}"] = to_python(best_acc.get(key, None))

        if key in auc_all and len(auc_all[key]) > 0:
            metrics[f"last_auc_{key}"] = to_python(auc_all[key][-1][1])
        else:
            metrics[f"last_auc_{key}"] = None

        if key in acc_hm_all and len(acc_hm_all[key]) > 0:
            metrics[f"last_acc_hm_{key}"] = to_python(acc_hm_all[key][-1][1])
        else:
            metrics[f"last_acc_hm_{key}"] = None

        if key in mpjpe_all and len(mpjpe_all[key]) > 0:
            metrics[f"last_mpjpe_{key}"] = to_python(mpjpe_all[key][-1][1])
            metrics[f"best_mpjpe_{key}"] = to_python(min([x[1] for x in mpjpe_all[key]]))
        else:
            metrics[f"last_mpjpe_{key}"] = None
            metrics[f"best_mpjpe_{key}"] = None

    metrics_json_path = os.path.join(args.exp_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    metrics_csv_path = os.path.join(args.exp_dir, "metrics_row.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

def prepare_joints_for_bmc(joints):
    """
    Only for BMCLoss:
    1) root-relative
    2) scale-invariant using cfg_bmc.REF_BONE_LINK

    Args:
        joints: [B, 21, 3]

    Returns:
        joints_bmc: [B, 21, 3]
    """
    # root-relative
    root = joints[:, cfg_bmc.JOINT_ROOT_IDX:cfg_bmc.JOINT_ROOT_IDX + 1, :]
    joints_bmc = joints - root

    # scale-invariant
    ref_a, ref_b = cfg_bmc.REF_BONE_LINK
    ref_bone = joints_bmc[:, ref_a, :] - joints_bmc[:, ref_b, :]
    ref_len = torch.norm(ref_bone, dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-8)

    joints_bmc = joints_bmc / ref_len
    return joints_bmc


def main(args):
    print("During dir preparation:")
    prepare_experiment_dirs(args)
    save_config(args)

    for path in [args.checkpoint, args.outpath]:
        if not os.path.isdir(path):
            os.makedirs(path)
    print(f"outpath: {args.outpath}")
    result_dir = os.path.join(args.outpath, "epoch_results") # 放训练和测试结果表
    curve_dir = os.path.join(args.outpath, "learning_curves") # 放learning curves图
    ensure_dir(result_dir)
    ensure_dir(curve_dir)

    misc.print_args(args)

    print("\nCREATE NETWORK")
    model = detnet(
        layers_resnet=args.layers_resnet,
        block_planes_resnet=args.block_planes_resnet,
        inplanes_resnet=args.inplanes_resnet,
        out_feature_dim_resnet=args.out_feature_dim_resnet,
        hidden_dim_detnet=args.hidden_dim_detnet,
        layers_net2d=args.layers_net2d,
        layers_net3d=args.layers_net3d,
        net2d_version=args.net2d_version,
        net3d_version=args.net3d_version,
        stacks=args.stacks
    )
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion_det = losses.DetLoss(
        lambda_hm=100.,
        lambda_dm=1.,
        lambda_lm=10.,
    )
    criterion = {
        'det': criterion_det
    }

    # ---- BMC loss (optional) ----
    if args.bmc_loss:
        criterion['bmc'] = BMCLoss(
            lambda_bl=args.lambda_bmc_bl,
            lambda_rb=args.lambda_bmc_rb,
            lambda_ja=args.lambda_bmc_ja,
            bmc_dir=args.bmc_dir,
            device=device,
        )

    optimizer = torch.optim.Adam(
        [
            {
                'params': model.parameters(),
                'initial_lr': args.learning_rate
            },

        ],
        lr=args.learning_rate
    )

    test_set_dic = {}
    test_loader_dic = {}
    best_acc = {}
    best_mpjpe = {}
    auc_all = {}
    acc_hm_all = {}
    mpjpe_all = {}

    for test_set_name in args.datasets_test:
        if test_set_name in ['stb', 'rhd', 'do']:
            test_set_dic[test_set_name] = HandDataset(
                data_split='test',
                train=False,
                subset_name=test_set_name,
                data_root=args.data_root,
            )
        elif test_set_name == 'eo':
            test_set_dic[test_set_name] = EgoDexter(
                data_split='test',
                data_root=args.data_root,
                hand_side="right"
            )
            print(test_set_dic[test_set_name])

        test_loader_dic[test_set_name] = torch.utils.data.DataLoader(
            test_set_dic[test_set_name],
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True, drop_last=False
        )
        best_acc[test_set_name] = 0
        best_mpjpe[test_set_name] = float("inf")
        auc_all[test_set_name] = []
        acc_hm_all[test_set_name] = []
        mpjpe_all[test_set_name] = []

    total_test_set_size = 0
    for key, value in test_set_dic.items():
        total_test_set_size += len(value)
    print("Total test set size: {}".format(total_test_set_size))

    if args.resume or args.evaluate:
        print("\nLOAD CHECKPOINT")
        state_dict = torch.load(os.path.join(
            args.checkpoint,
            'ckp_detnet_{}.pth'.format(args.evaluate_id)
        ))
        # if args.clean:
        state_dict = misc.clean_state_dict(state_dict)

        model.load_state_dict(state_dict)
    else:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    if args.evaluate:
        for key, value in test_loader_dic.items():
            validate(value, model, criterion, key, args=args)
        return 0

    train_dataset = HandDataset(
        data_split='train',
        train=True,
        subset_name=args.datasets_train,
        data_root=args.data_root,
        scale_jittering=0.1,
        center_jettering=0.1,
        max_rot=0.5 * np.pi,
    )

    print("Total train dataset size: {}".format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True, drop_last=False
    )

    # DataParallel so u can use multi GPUs
    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    acc_hm = {}
    mpjpe = {}

    loss_all = {"lossH": [],
                "lossD": [],
                "lossL": [],
                "bmc_total": [],
                "bmc_bl": [],
                "bmc_rb": [],
                "bmc_ja": [],
                }

    train_mpjpe_all = []
    train_epoch_results = [] # 保存每个 epoch 的训练结果
    test_epoch_results = [] # 保存每个 epoch、每个 test set 的测试结果

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])
        #############  trian for one epoch  ###############
        train_result = train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args, loss_all=loss_all
        )
        train_mpjpe_all.append([epoch + 1, train_result["train_mpjpe"]])

        train_epoch_info = {
            "epoch": epoch + 1,
            "train_lossH": train_result["train_lossH"],
            "train_lossD": train_result["train_lossD"],
            "train_lossL": train_result["train_lossL"],
            "train_mpjpe": train_result["train_mpjpe"],
        }

        if args.bmc_loss:
            train_epoch_info.update({
                "train_bmc_total": train_result["train_bmc_total"],
                "train_bmc_bl": train_result["train_bmc_bl"],
                "train_bmc_rb": train_result["train_bmc_rb"],
                "train_bmc_ja": train_result["train_bmc_ja"],
            })

        train_epoch_results.append(train_epoch_info)
        ##################################################
        auc = best_acc.copy() # need to deepcopy it because it's a dict

        print("Before Validating")
        print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")
        for i, (key, value) in enumerate(test_loader_dic.items()):
            print(f"{i}th: key={key}, value={value}")
            auc[key], acc_hm[key], mpjpe[key] = validate(value, model, criterion, key, args=args)
            auc_all[key].append([epoch + 1, auc[key]])
            acc_hm_all[key].append([epoch + 1, acc_hm[key]])
            mpjpe_all[key].append([epoch + 1, mpjpe[key]])

            test_epoch_results.append({
                "epoch": int(epoch + 1),
                "test_set": str(key),
                "auc": float(auc[key]) if auc[key] is not None else None,
                "acc_hm": float(acc_hm[key]) if acc_hm[key] is not None else None,
                "mpjpe": float(mpjpe[key]) if mpjpe[key] is not None else None,
            })

        print("After Validating, Before Save Checkpoints")
        print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")

        misc.save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': model,
            },
            checkpoint=args.checkpoint,
            filename='{}.pth'.format(args.saved_prefix), # to do 在saved_prefix后面加参数配置
            snapshot=args.snapshot,
            is_best=[auc, best_acc]
        )

        # 保存 best MPJPE checkpoint（MPJPE 越小越好）
        current_ckpt_path = os.path.join(args.checkpoint, '{}.pth'.format(args.saved_prefix))
        fileprefix = args.saved_prefix

        for key in test_loader_dic.keys():
            if np.isfinite(mpjpe[key]) and mpjpe[key] < best_mpjpe[key]:
                best_mpjpe[key] = mpjpe[key]

                dst_path = os.path.join(
                    args.checkpoint,
                    f'{fileprefix}_{key}_best_mpjpe.pth'
                )

                print("[DEBUG] src =", os.path.abspath(current_ckpt_path), "len =", len(os.path.abspath(current_ckpt_path)))
                print("[DEBUG] dst =", os.path.abspath(dst_path), "len =", len(os.path.abspath(dst_path)))

                shutil.copyfile(current_ckpt_path, dst_path)

        print("After Save Checkpoints")
        print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")

        for key, value in test_loader_dic.items():
            if auc[key] > best_acc[key]:
                best_acc[key] = auc[key]

        misc.out_loss_auc(
            loss_all,
            auc_all,
            acc_hm_all,
            outpath=args.outpath,
            train_mpjpe_all_=train_mpjpe_all,
            mpjpe_all_=mpjpe_all
            )

        # 保存训练/测试结果
        save_dict_list_to_csv(
            train_epoch_results,
            os.path.join(result_dir, "train_epoch_results.csv")
        )
        save_json(
            train_epoch_results,
            os.path.join(result_dir, "train_epoch_results.json")
        )

        save_dict_list_to_csv(
            test_epoch_results,
            os.path.join(result_dir, "test_epoch_results.csv")
        )
        save_json(
            test_epoch_results,
            os.path.join(result_dir, "test_epoch_results.json")
        )

        # 保存learning curves
        save_learning_curves(
            curve_dir=curve_dir,
            loss_all=loss_all,
            train_mpjpe_all=train_mpjpe_all,
            auc_all=auc_all,
            acc_hm_all=acc_hm_all,
            mpjpe_all=mpjpe_all,
            use_bmc=args.bmc_loss
        )

        scheduler.step()

    save_metrics(args, best_acc, auc_all, acc_hm_all, loss_all, train_mpjpe_all, mpjpe_all)

    del model
    del optimizer
    del scheduler
    del train_loader
    del train_dataset
    del test_loader_dic
    del test_set_dic
    del criterion
    del loss_all
    del auc_all
    del acc_hm_all

    cleanup()

    return 0  # end of main


def one_forward_pass(metas, model, criterion, args, train=True):
    clr = metas['clr'].to(device, non_blocking=True)

    ''' prepare infos '''
    if 'hm_veil' in metas.keys():
        hm_veil = metas['hm_veil'].to(device, non_blocking=True)  # (B,21)

        infos = {
            'hm_veil': hm_veil,
            'batch_size': clr.shape[0]
        }

        ''' prepare targets '''

        hm = metas['hm'].to(device, non_blocking=True)
        delta_map = metas['delta_map'].to(device, non_blocking=True)
        location_map = metas['location_map'].to(device, non_blocking=True)
        flag_3d = metas['flag_3d'].to(device, non_blocking=True)
        joint = metas['joint'].to(device, non_blocking=True)

        targets = {
            'clr': clr,
            'hm': hm,
            'dm': delta_map,
            'lm': location_map,
            "flag_3d": flag_3d,
            "joint": joint

        }
    else:
        infos = {
            'batch_size': clr.shape[0]
        }
        tips = metas['tips'].to(device, non_blocking=True)
        targets = {
            'clr': clr,
            "joint": tips

        }

    ''' ----------------  Forward Pass  ---------------- '''
    results = model(clr)
    ''' ----------------  Forward End   ---------------- '''

    total_loss = torch.Tensor([0]).cuda()
    losses = {}

    if not train:
        return results, {**targets, **infos}, total_loss, losses

    ''' compute losses '''
    if args.det_loss:
        det_total_loss, det_losses, batch_3d_size = criterion['det'].compute_loss(
            results, targets, infos
        )
        total_loss += det_total_loss
        losses.update(det_losses)

        targets["batch_3d_size"] = batch_3d_size

    # ---- optional BMCLoss branch ----
    if args.bmc_loss:
        # only use normalized joints for BMCLoss
        # do NOT modify results['xyz']
        joints_bmc = prepare_joints_for_bmc(results['xyz'])

        bmc_total_loss, bmc_losses = criterion['bmc'].compute_loss(joints_bmc)
        total_loss += bmc_total_loss

        losses['bmc_total_loss'] = bmc_total_loss
        losses.update(bmc_losses)

    return results, {**targets, **infos}, total_loss, losses


def validate(val_loader, model, criterion, key, args, stop=-1):
    print("{}_test_set under test".format(key))
    # switch to evaluate mode
    model.eval()

    if key in ["stb", "rhd"]:
        am_accH = AverageMeter()

    am_mpjpe = AverageMeter()

    evaluator = EvalUtil()

    if args.evaluate:
        gt_joints = []
        pre_joints = []

    with torch.no_grad():
        for i, metas in tqdm(enumerate(val_loader)):
            preds, targets, _1, _2 = one_forward_pass(
                metas, model, criterion, args=None, train=False
            )

            if key in ["stb", "rhd"]:
                # heatmap accuracy
                avg_acc_hm, _ = accuracy_heatmap(
                    preds['h_map'],
                    targets['hm'],
                    targets['hm_veil']
                )
                am_accH.update(avg_acc_hm, targets['batch_size'])

            pred_joint = func.to_numpy(preds['xyz'])

            gt_joint = func.to_numpy(targets['joint'])

            if args.evaluate:
                gt_joints.extend(gt_joint.tolist())
                pre_joints.extend(pred_joint.tolist())

            gt_joint, pred_joint_align = align.global_align(gt_joint, pred_joint, key=key)

            if key in ["stb", "rhd"]:
                # "stb"和"rhd"数据集是固定 21x3，直接按照原思路batch内计算
                batch_mpjpe = compute_mpjpe_numpy(pred_joint_align, gt_joint)
                am_mpjpe.update(batch_mpjpe, targets['batch_size'])

                for targj, predj_a in zip(gt_joint, pred_joint_align):
                    evaluator.feed(targj * 1000.0, predj_a * 1000.0)
                    # vis.multi_plot3d([targj * 1000.0, predj_a * 1000.0], title=["target", "pred"])

            elif key in ["do", "eo"]:
                # "do"和"eo"数据集，每个样本有效joints数可能不同（5/4/3），故而逐样本计算
                for targj, predj_a in zip(gt_joint, pred_joint_align):
                    sample_mpjpe = compute_mpjpe_numpy(predj_a, targj)
                    am_mpjpe.update(sample_mpjpe, 1)
                    evaluator.feed(targj * 1000.0, predj_a * 1000.0)

            if stop != -1 and i >= stop:
                break

    if args.evaluate:

        gt_joints = np.array(gt_joints)
        pre_joints = np.array(pre_joints)
        out_path = "out_testset"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        np.save("{}/{}_gt_joints.npy".format(out_path, key), gt_joints)
        np.save("{}/{}_pre_joints.npy".format(out_path, key), pre_joints)

    (
        _1, _2, _3,
        auc_all,
        pck_curve_all,
        thresholds
    ) = evaluator.get_measures(
        20, 50, 15
    )
    print("AUC all of {}_test_set is : {}".format(key, auc_all))

    print("MPJPE of {}_test_set is : {}".format(key, am_mpjpe.avg))

    if key in ["stb", "rhd"]:
        return auc_all, am_accH.avg, am_mpjpe.avg
    elif key in ["do", "eo"]:
        return auc_all, 0, am_mpjpe.avg


def train(train_loader, model, criterion, optimizer, args, loss_all):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    am_loss_hm = AverageMeter()
    am_loss_dm = AverageMeter()
    am_loss_lm = AverageMeter()
    am_bmc_total = AverageMeter()
    am_bmc_bl = AverageMeter()
    am_bmc_rb = AverageMeter()
    am_bmc_ja = AverageMeter()
    am_mpjpe = AverageMeter()

    last = time.time()
    # switch to trian
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
    # for i, metas in tqdm(enumerate(train_loader)):
    for i, metas in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, total_loss, losses = one_forward_pass(
            metas, model, criterion, args, train=True
        )

        am_loss_hm.update(losses['det_hm'].item(), targets['batch_size'])
        am_loss_dm.update(losses['det_dm'].item(), targets['batch_3d_size'].item())
        am_loss_lm.update(losses['det_lm'].item(), targets['batch_3d_size'].item())

        if args.bmc_loss:
            am_bmc_total.update(losses['bmc_total_loss'].item(), targets['batch_size'])
            am_bmc_bl.update(losses['bmc_bl'].item(), targets['batch_size'])
            am_bmc_rb.update(losses['bmc_rb'].item(), targets['batch_size'])
            am_bmc_ja.update(losses['bmc_ja'].item(), targets['batch_size'])

        batch_mpjpe = compute_mpjpe_torch(results['xyz'], targets['joint'])
        am_mpjpe.update(batch_mpjpe.item(), targets['batch_size'])

        ''' backward and step '''
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ''' progress '''
        batch_time.update(time.time() - last)
        last = time.time()
        bar.suffix = (
            '({batch}/{size}) '
            'd: {data:.2f}s | '
            'b: {bt:.2f}s | '
            't: {total:}s | '
            'eta:{eta:}s | '
            'lH: {lossH:.7f} | '
            'lD: {lossD:.5f} | '
            'lL: {lossL:.5f} | '
            'bmcT: {bmcT:.5f} | '
            'bmcBL: {bmcBL:.5f} | '
            'bmcRB: {bmcRB:.5f} | '
            'bmcJA: {bmcJA:.5f} | '
            'MPJPE: {mpjpe:.5f} | '

        ).format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            lossH=am_loss_hm.avg,
            lossD=am_loss_dm.avg,
            lossL=am_loss_lm.avg,
            bmcT=am_bmc_total.avg if args.bmc_loss else 0.0,
            bmcBL=am_bmc_bl.avg if args.bmc_loss else 0.0,
            bmcRB=am_bmc_rb.avg if args.bmc_loss else 0.0,
            bmcJA=am_bmc_ja.avg if args.bmc_loss else 0.0,
            mpjpe=am_mpjpe.avg,

        )

        if DEBUG:
            if i == 1:
                break
        bar.next()
    bar.finish()

    loss_all["lossH"].append(am_loss_hm.avg)
    loss_all["lossD"].append(am_loss_dm.avg)
    loss_all["lossL"].append(am_loss_lm.avg)
    loss_all["bmc_total"].append(am_bmc_total.avg if args.bmc_loss else 0.0)
    loss_all["bmc_bl"].append(am_bmc_bl.avg if args.bmc_loss else 0.0)
    loss_all["bmc_rb"].append(am_bmc_rb.avg if args.bmc_loss else 0.0)
    loss_all["bmc_ja"].append(am_bmc_ja.avg if args.bmc_loss else 0.0)

    return {
        "train_lossH": am_loss_hm.avg,
        "train_lossD": am_loss_dm.avg,
        "train_lossL": am_loss_lm.avg,
        "train_bmc_total": am_bmc_total.avg if args.bmc_loss else 0.0,
        "train_bmc_bl": am_bmc_bl.avg if args.bmc_loss else 0.0,
        "train_bmc_rb": am_bmc_rb.avg if args.bmc_loss else 0.0,
        "train_bmc_ja": am_bmc_ja.avg if args.bmc_loss else 0.0,
        "train_mpjpe": am_mpjpe.avg,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Train: DetNet')
    # Dataset setting
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default="data",
        help='dataset root directory'
    )
    parser.add_argument(
        "-trs",
        "--datasets_train",
        nargs="+",
        default=['cmu', 'rhd', 'gan'],
        type=str,
        help="sub datasets, should be listed in: [cmu|rhd|gan]"
    )

    parser.add_argument(
        "-tes",
        "--datasets_test",
        nargs="+",
        default=['rhd', 'stb', "do", "eo"],
        type=str,
        help="sub datasets, should be listed in: [rhd|stb|do|eo]"
    )

    # Miscs
    parser.add_argument(
        '-ckp',
        '--checkpoint',
        default='checkpoints',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )

    parser.add_argument(
        '-sp',
        '--saved_prefix',
        default='ckp_detnet',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )

    parser.add_argument(
        '-op',
        '--outpath',
        default='out_loss_auc',
        type=str,
        metavar='PATH',
        help='path to out_testset loss and auc (default: out_testset)'
    )

    parser.add_argument(
        '--run_name',
        default='',
        type=str,
        help='name of current experiment run'
    )

    parser.add_argument(
        '--exp_dir',
        default='',
        type=str,
        help='root directory of current experiment'
    )

    parser.add_argument(
        '--snapshot',
        default=1, type=int,
        help='save models for every #snapshot epochs (default: 0)'
    )

    parser.add_argument(
        '-r', '--resume',
        dest='resume',
        action='store_true',
        help='whether to load checkpoint (default: none)'
    )
    parser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set'
    )

    # Training Parameters
    parser.add_argument(
        '-eid', '--evaluate_id',
        default=319,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '-c', '--clean',
        dest='clean',
        action='store_true',
        help='clean model on one gpu if trained on 2 gpus'
    )
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '--epochs',
        default=500,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '-se', '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)'
    )
    parser.add_argument(
        '-b', '--train_batch',
        default=32,
        type=int,
        metavar='N',
        help='train batchsize'
    )
    parser.add_argument(
        '-tb', '--test_batch',
        default=128,
        type=int,
        metavar='N',
        help='test batchsize'
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        default=1e-3,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        "--lr_decay_step",
        default=250,
        type=int,
        help="Epochs after which to decay learning rate",
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )

    parser.add_argument(
        '--det_loss',
        dest='det_loss',
        action='store_true',
        help='Calculate detnet loss',
        default=True
    )
    # detnet hyperparameters
    parser.add_argument(
        '--layers_resnet',
        nargs='+',
        type=int,
        default=[2, 4, 6],
        help='detnet: layers_resnet'
    )
    parser.add_argument(
        '--block_planes_resnet',
        nargs='+',
        type=int,
        default=[64, 128, 256],
        help='detnet: block_planes_resnet'
    )
    parser.add_argument(
        '--inplanes_resnet',
        type=int,
        default=64,
        help='detnet: inplanes_resnet'
    )
    parser.add_argument(
        '--out_feature_dim_resnet',
        type=int,
        default=256,
        help='detnet: out_feature_dim_resnet'
    )
    parser.add_argument(
        '--hidden_dim_detnet',
        type=int,
        default=256,
        help='detnet: hidden_dim_detnet'
    )
    parser.add_argument(
        '--layers_net2d',
        nargs='+',
        type=int,
        default=[3, 3],
        help='detnet: layers_net2d'
    )
    parser.add_argument(
        '--layers_net3d',
        nargs='+',
        type=int,
        default=[3, 3],
        help='detnet: layers_net3d'
    )
    parser.add_argument(
        '--net2d_version',
        type=str,
        default="bottleneck",
        help='detnet: net2d_version, choose bottleneck or legacy'
    )
    parser.add_argument(
        '--net3d_version',
        type=str,
        default="bottleneck",
        help='detnet: net3d_version, choose bottleneck or legacy'
    )
    parser.add_argument(
        '--stacks',
        type=int,
        default=1,
        help='detnet: stacks'
    )
    # BMC Loss
    parser.add_argument(
        '--bmc_loss',
        action='store_true',
        help='enable BMCLoss as an additional optional loss'
    )
    parser.add_argument(
        '--lambda_bmc_bl',
        type=float,
        default=0.0,
        help='weight for BMC bone-length loss'
    )
    parser.add_argument(
        '--lambda_bmc_rb',
        type=float,
        default=0.0,
        help='weight for BMC root-bone loss'
    )
    parser.add_argument(
        '--lambda_bmc_ja',
        type=float,
        default=0.0,
        help='weight for BMC joint-angle loss'
    )
    parser.add_argument(
        '--bmc_dir',
        type=str,
        default='BMC',
        help='directory containing precomputed BMC npy files'
    )


    main(parser.parse_args())