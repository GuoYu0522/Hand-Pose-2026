$TRAIN_SCRIPT = "train_detnet.py"
$BASE_EXP_DIR = "experiments"

if (!(Test-Path $BASE_EXP_DIR)) {
    New-Item -ItemType Directory -Path $BASE_EXP_DIR | Out-Null
}

$learning_rates = @(0.001, 0.0005)
$train_batches = @(16, 32)
$test_batches = @(128)
$epochs_list = @(100)
$workers_list = @(8)
$lr_decay_steps = @(50, 100)
$gammas = @(0.1)

$layers_resnet_list = @(
    @(2,4,6),
    @(2,3,4)
)

$block_planes_resnet_list = @(
    @(64,128,256),
    @(64,128,512)
)

$inplanes_resnet_list = @(64)
$out_feature_dim_resnet_list = @(256, 512)
$hidden_dim_detnet_list = @(256, 512)

$layers_net2d_list = @(
    @(3,3),
    @(2,2)
)

$stacks_list = @(1,2)

$exp_id = 1

foreach ($lr in $learning_rates) {
    foreach ($tb in $train_batches) {
        foreach ($testb in $test_batches) {
            foreach ($ep in $epochs_list) {
                foreach ($wk in $workers_list) {
                    foreach ($decay in $lr_decay_steps) {
                        foreach ($gamma in $gammas) {
                            foreach ($lres in $layers_resnet_list) {
                                foreach ($bp in $block_planes_resnet_list) {
                                    foreach ($inp in $inplanes_resnet_list) {
                                        foreach ($outf in $out_feature_dim_resnet_list) {
                                            foreach ($hid in $hidden_dim_detnet_list) {
                                                foreach ($ln2d in $layers_net2d_list) {
                                                    foreach ($st in $stacks_list) {

                                                        $lres_str = ($lres -join "-")
                                                        $bp_str = ($bp -join "-")
                                                        $ln2d_str = ($ln2d -join "-")

                                                        $run_name = "exp_{0:D4}_lr{1}_tb{2}_lres{3}_bp{4}_in{5}_out{6}_hid{7}_ln2d{8}_st{9}_ep{10}_g{11}_decay{12}" -f `
                                                            $exp_id, $lr, $tb, $lres_str, $bp_str, $inp, $outf, $hid, $ln2d_str, $st, $ep, $gamma, $decay

                                                        $exp_dir = Join-Path $BASE_EXP_DIR $run_name
                                                        $metrics_json = Join-Path $exp_dir "metrics.json"

                                                        if (Test-Path $metrics_json) {
                                                            Write-Host "[Skip] $run_name already finished."
                                                            $exp_id++
                                                            continue
                                                        }

                                                        $cmd = @(
                                                            "python", $TRAIN_SCRIPT,
                                                            "--exp_dir", $exp_dir,
                                                            "--run_name", $run_name,
                                                            "--data_root", "/home/yg/datasets/",
                                                            "--datasets_train", "cmu", "rhd", "gan",
                                                            "--datasets_test", "rhd", "stb", "do", "eo",
                                                            "--snapshot", "1",
                                                            "--workers", "$wk",
                                                            "--epochs", "$ep",
                                                            "--train_batch", "$tb",
                                                            "--test_batch", "$testb",
                                                            "--learning-rate", "$lr",
                                                            "--lr_decay_step", "$decay",
                                                            "--gamma", "$gamma",
                                                            "--layers_resnet"
                                                        ) + ($lres | ForEach-Object { "$_" }) + @(
                                                            "--block_planes_resnet"
                                                        ) + ($bp | ForEach-Object { "$_" }) + @(
                                                            "--inplanes_resnet", "$inp",
                                                            "--out_feature_dim_resnet", "$outf",
                                                            "--hidden_dim_detnet", "$hid",
                                                            "--layers_net2d"
                                                        ) + ($ln2d | ForEach-Object { "$_" }) + @(
                                                            "--stacks", "$st"
                                                        )

                                                        Write-Host ("=" * 120)
                                                        Write-Host ($cmd -join " ")

                                                        & $cmd[0] $cmd[1..($cmd.Length - 1)]
                                                        $exp_id++
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}