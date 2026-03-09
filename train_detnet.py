import argparse
import os
import time
import json
import csv

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

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
DEBUG = 0

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
        args.exp_dir = os.path.join("experiments", args.run_name)
    else:
        if args.run_name is None or str(args.run_name).strip() == "":
            args.run_name = os.path.basename(os.path.normpath(args.exp_dir))

    args.checkpoint = os.path.join(args.exp_dir, "checkpoints")
    args.outpath = os.path.join(args.exp_dir, "outputs")

    for path in [args.exp_dir, args.checkpoint, args.outpath]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # 若saved_prefix没显式设置成个性化名字，就自动加参数信息
    if args.saved_prefix == "ckp_detnet":
        #args.saved_prefix = f"ckp_detnet_{args.run_name}"
        args.saved_prefix = f"ckp_detnet_{args.run_name[:30]}"

def to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    return obj

def save_config(args):
    config_path = os.path.join(args.exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

def save_metrics(args, best_acc, auc_all, acc_hm_all, loss_all):
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

    metrics_json_path = os.path.join(args.exp_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    metrics_csv_path = os.path.join(args.exp_dir, "metrics_row.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main(args):
    prepare_experiment_dirs(args)
    save_config(args)

    for path in [args.checkpoint, args.outpath]:
        if not os.path.isdir(path):
            os.makedirs(path)

    misc.print_args(args)

    print("\nCREATE NETWORK")
    model = detnet(
        layers_resnet=args.layers_resnet,
        block_planes_resnet=args.block_planes_resnet,
        inplanes_resnet=args.inplanes_resnet,
        out_feature_dim_resnet=args.out_feature_dim_resnet,
        hidden_dim_detnet=args.hidden_dim_detnet,
        layers_net2d=args.layers_net2d,
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
    auc_all = {}
    acc_hm_all = {}
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
        auc_all[test_set_name] = []
        acc_hm_all[test_set_name] = []

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
    loss_all = {"lossH": [],
                "lossD": [],
                "lossL": [],

                }

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
        
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])
        #############  trian for one epoch  ###############
        print("Before Training")
        print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args, loss_all=loss_all
        )
        ##################################################
        auc = best_acc.copy() # need to deepcopy it because it's a dict
        print("Before Validating")
        print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")
        for key, value in test_loader_dic.items():
            i = 0
            print(f"{i}th: key={key}, value={value}")
            auc[key], acc_hm[key] = validate(value, model, criterion, key, args=args)
            auc_all[key].append([epoch + 1, auc[key]])
            acc_hm_all[key].append([epoch + 1, acc_hm[key]])
            i = i+1

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

        print("After Save Checkpoints")
        print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")

        for key, value in test_loader_dic.items():
            if auc[key] > best_acc[key]:
                best_acc[key] = auc[key]

        misc.out_loss_auc(loss_all, auc_all, acc_hm_all, outpath=args.outpath) # to do

        scheduler.step()

    save_metrics(args, best_acc, auc_all, acc_hm_all, loss_all)

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

    return results, {**targets, **infos}, total_loss, losses


def validate(val_loader, model, criterion, key, args, stop=-1):
    print("{}_test_set under test".format(key))
    # switch to evaluate mode
    model.eval()

    if key in ["stb", "rhd"]:
        am_accH = AverageMeter()

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


            for targj, predj_a in zip(gt_joint, pred_joint_align):
                evaluator.feed(targj * 1000.0, predj_a * 1000.0)
                # vis.multi_plot3d([targj * 1000.0, predj_a * 1000.0], title=["target", "pred"])

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

    if key in ["stb", "rhd"]:
        return auc_all, am_accH.avg
    elif key in ["do", "eo"]:
        return auc_all, 0


def train(train_loader, model, criterion, optimizer, args, loss_all):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    am_loss_hm = AverageMeter()
    am_loss_dm = AverageMeter()
    am_loss_lm = AverageMeter()

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

        )

        if DEBUG:
            if i == 1:
                break
        bar.next()
    bar.finish()

    loss_all["lossH"].append(am_loss_hm.avg)
    loss_all["lossD"].append(am_loss_dm.avg)
    loss_all["lossL"].append(am_loss_lm.avg)


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
        '--stacks',
        type=int,
        default=1,
        help='detnet: stacks'
    )

    main(parser.parse_args())
