#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

current_run_pattern='python.*ssm_edge_classification[.]py.*'
waited=0

echo "Waiting for the ec_focal_low_lr edge-classification process to finish..."
while pgrep -f "$current_run_pattern" >/dev/null; do
    sleep 300
    waited=$((waited + 30))
    if (( waited % 60 == 0 )); then
        echo "Still waiting (${waited}s)..."
    fi
done

echo "Previous run finished; starting flag-label experiment."
python ssm_edge_classification.py \
    --dataset li_small \
    --device cuda \
    --epochs 50 \
    --runs 1 \
    --batches 128 \
    --pos_weight sqrt \
    --head_type linear \
    --lr 3e-4 \
    --wd 1e-3 \
    --use_relational True \
    --loss focal \
    --gamma 2.0 \
    --eval_metric AP \
    --patience 10 \
    --lr_patience 5 \
    --name ec_focal_low_lr_flag_128 \
    --label_source flag \
    --wandb
