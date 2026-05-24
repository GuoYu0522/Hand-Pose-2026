import os
import sys
import json
import itertools
import subprocess
import csv
from datetime import datetime

TRAIN_SCRIPT = "train_detnet.py"   # 训练文件（注意文件名字需要对应！！！）
BASE_EXP_DIR = "experiments"


def list_to_str(x):
    if isinstance(x, (list, tuple)):
        return "-".join(map(str, x))
    return str(x)


def make_run_name(exp_id, config):
    run_name = (
        f"exp_{exp_id:04d}"
        f"_lr{config['learning_rate']}"
        f"_tb{config['train_batch']}"
        f"_lres{list_to_str(config['layers_resnet'])}"
        f"_bp{list_to_str(config['block_planes_resnet'])}"
        f"_in{config['inplanes_resnet']}"
        f"_out{config['out_feature_dim_resnet']}"
        f"_hid{config['hidden_dim_detnet']}"
        f"_ln2d{list_to_str(config['layers_net2d'])}"
        f"_ln3d{list_to_str(config['layers_net3d'])}"
        f"_vr2d{config['net2d_version']}"
        f"_vr3d{config['net3d_version']}"
        f"_st{config['stacks']}"
        f"_ep{config['epochs']}"
        f"_g{config['gamma']}"
        f"_decay{config['lr_decay_step']}"
    )

    if config.get("bmc_loss", False):
        run_name += (
            f"_bmc"
            f"_bl{config['lambda_bmc_bl']}"
            f"_rb{config['lambda_bmc_rb']}"
            f"_ja{config['lambda_bmc_ja']}"
        )

    return run_name

def get_short_name(run_name):
    parts = str(run_name).split("_")
    short_name = "_".join(parts[:2]) if len(parts)>=2 else str(run_name)
    return short_name

def append_summary(summary_csv, metrics_json_path):
    if not os.path.exists(metrics_json_path):
        return

    with open(metrics_json_path, "r", encoding="utf-8") as f:
        row = json.load(f)

    file_exists = os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_grid_search_config(grid_search_dir, timestamp, search_space, fixed_args):
    config_path = os.path.join(grid_search_dir, "grid_search_config.json")
    config = {
        "timestamp": timestamp,
        "grid_search_dir": grid_search_dir,
        "train_script": TRAIN_SCRIPT,
        "base_exp_dir": BASE_EXP_DIR,
        "search_space": search_space,
        "fixed_args": fixed_args,
        "datasets_train": fixed_args.get("datasets_train", []),
        "datasets_test": fixed_args.get("datasets_test", []),
    }

    total_experiments = 1
    for v in search_space.values():
        total_experiments *= len(v)
    config["num_total_experiments"] = total_experiments

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def main():
    os.makedirs(BASE_EXP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_search_dir = os.path.join(BASE_EXP_DIR, f"grid_search_{timestamp}")
    os.makedirs(grid_search_dir, exist_ok=True)

    summary_csv = os.path.join(grid_search_dir, "summary.csv")

    # grid search空间
    search_space = {
        "learning_rate": [8e-4],
        # "learning_rate": [1e-3],
        #"train_batch": [16, 32],
        "train_batch": [32],
        "test_batch": [128],
        "epochs": [99],
        "workers": [8],
        #"lr_decay_step": [50, 100],
        "lr_decay_step": [100],
        "gamma": [0.1],
        "inplanes_resnet": [64],
        "layers_resnet": [
            # [2, 3, 4],
            [2, 4, 6],
        ],
        "block_planes_resnet": [
            # #[32, 64, 128],
            [64, 128, 128],
            ## [64, 128, 256],
           # [64, 128, 512]
        ],

        # "out_feature_dim_resnet": [256, 512],
        "out_feature_dim_resnet": [256],
        #"hidden_dim_detnet": [256, 512],
        "hidden_dim_detnet": [256],
        "layers_net2d": [
            # #[3, 3],
            [2, 3, 4]
        ],
        "layers_net3d": [
            [3, 3],
           # [2, 3, 4]
        ],
        "net2d_version": [
           "bottleneck",
          #  "legacy"
        ],
        "net3d_version": [
           "bottleneck",
          #  "legacy"
        ],
        #"stacks": [1, 2],
        "stacks": [1],
        
        # BMC loss search space (optional)
        "bmc_loss": [True],
        # "bmc_loss": [True, False],
        "lambda_bmc_bl": [2.5],
        "lambda_bmc_rb": [2.5],
        "lambda_bmc_ja": [2.5],
    }

    fixed_args = {
        "data_root": "data",   # 你的路径！！！
        "datasets_train": ["cmu", "rhd"],
        "datasets_test": ["rhd", "do", "eo"],
        "snapshot": 1,
        "bmc_dir": "BMC",
    }

    save_grid_search_config(grid_search_dir, timestamp, search_space, fixed_args)

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    exp_id = 1
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        config.update(fixed_args)

        run_name = make_run_name(exp_id, config)
        short_run_name = get_short_name(run_name)

        # ===== [CHANGED] =====
        exp_dir = os.path.join(grid_search_dir, short_run_name)

        metrics_json_path = os.path.join(exp_dir, "metrics.json")

        if os.path.exists(metrics_json_path):
            print(f"[Skip] {short_run_name} already finished.")
            append_summary(summary_csv, metrics_json_path)
            exp_id += 1
            continue

        cmd = [
            sys.executable, TRAIN_SCRIPT,
            "--exp_dir", exp_dir,
            "--run_name", short_run_name,

            "--data_root", str(config["data_root"]),
            "--datasets_train", *map(str, config["datasets_train"]),
            "--datasets_test", *map(str, config["datasets_test"]),

            "--snapshot", str(config["snapshot"]),
            "--workers", str(config["workers"]),
            "--epochs", str(config["epochs"]),
            "--train_batch", str(config["train_batch"]),
            "--test_batch", str(config["test_batch"]),
            "--learning-rate", str(config["learning_rate"]),
            "--lr_decay_step", str(config["lr_decay_step"]),
            "--gamma", str(config["gamma"]),

            "--layers_resnet", *map(str, config["layers_resnet"]),
            "--block_planes_resnet", *map(str, config["block_planes_resnet"]),
            "--inplanes_resnet", str(config["inplanes_resnet"]),
            "--out_feature_dim_resnet", str(config["out_feature_dim_resnet"]),
            "--hidden_dim_detnet", str(config["hidden_dim_detnet"]),
            "--layers_net2d", *map(str, config["layers_net2d"]),
            "--layers_net3d", *map(str, config["layers_net3d"]),
            "--net2d_version", str(config["net2d_version"]),
            "--net3d_version", str(config["net3d_version"]),
            "--stacks", str(config["stacks"]),
        ]

        # ---- optional BMCLoss args ----
        if config.get("bmc_loss", False):
            cmd.extend([
                "--bmc_loss",
                "--lambda_bmc_bl", str(config["lambda_bmc_bl"]),
                "--lambda_bmc_rb", str(config["lambda_bmc_rb"]),
                "--lambda_bmc_ja", str(config["lambda_bmc_ja"]),
                "--bmc_dir", str(config["bmc_dir"]),
            ])

        print("=" * 120)
        print("Running:")
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Failed] {short_run_name}: {e}")
            exp_id += 1
            continue

        append_summary(summary_csv, metrics_json_path)
        exp_id += 1


if __name__ == "__main__":
    main()