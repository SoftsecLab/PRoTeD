#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=01

# python -m src.experiment.PRoTeD_test \
#   --mode eval \
#   --model_dir saved_models/PRoTeD/ieee-polish_run_seed11/best_model \
#   --eval_path data/final_training_data/ieee-polish-test.jsonl \
#   --out_dir saved_models/PRoTeD/ieee-polish_run_seed11/test_eval \
#   --perturb_mode B --softmix_lambda 0.5 --gumbel_tau 1.0 --softmix_max_sentences 8

python -m src.experiment.PRoTeD_test \
  --mode eval \
  --model_dir saved_models/PRoTeD/ieee-polish_run_seed333/best_model \
  --eval_path data/final_training_data/ieee-polish-test.jsonl \
  --out_dir saved_models/PRoTeD/ieee-polish_run_seed333/test_eval \
  --perturb_mode B --softmix_lambda 0.5 --gumbel_tau 1.0 --softmix_max_sentences 8

