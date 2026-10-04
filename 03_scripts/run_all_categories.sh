#!/bin/bash
# Run from the root of the cloned LogiCo repo (patched), conda/venv activated.
# Trains and evaluates every category, best and epoch-200 checkpoints.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for c in breakfast_box juice_bottle pushpins screw_bag splicing_connectors; do
  python train_logico.py --dataset mvtec_loco --category $c --save_dir saved_results --accum_steps 2 \
      > logs/${c}_train.log 2>&1
  python evaluation.py --dataset mvtec_loco --category $c --save_dir saved_results --batch_size 7 \
      > logs/${c}_eval_best.log 2>&1
  python evaluation.py --dataset mvtec_loco --category $c --save_dir saved_results --batch_size 7 \
      --ckpt_file_name LogiCo_epoch_200_ckpt.pkl > logs/${c}_eval_last.log 2>&1
  # keep disk usage down: delete intermediate checkpoints, keep best and epoch 200
  find saved_results/mvtec_loco/$c -name "LogiCo_epoch_*_ckpt.pkl" ! -name "LogiCo_epoch_200_ckpt.pkl" -delete
done
