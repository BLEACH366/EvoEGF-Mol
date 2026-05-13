#!/bin/bash

conda activate evoegfmol

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="MolEGF_best_${TIMESTAMP}.log"

generated_dir=./logs/gauss_no_mask_fisher_pos_s1_10_02_type_s1_02_bond_ema_std8_full_b16_t1_gate_ada_halfdir111_epoch25/test_outputs_v2_sample_steps_20/20260326-101500


nohup python eval_generated.py \
    --generated_path $generated_dir \
    --docking_mode vina_dock \
    --test_only --no_wandb \
    --config_file $generated_dir/config.yaml \
    > "$LOG_FILE" 2>&1 &

# 显示后台任务信息
echo "Training started in background. PID: $!"
