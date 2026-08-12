#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=01
export NLTK_DATA=models/nltk_data


datasets=(
    "ieee-polish"
    "ieee-generation"

    "multi_domains_arxiv"
    "multi_domains_xsum"
     "multi_domains_writing_prompt"
    "multi_domains_yelp_review"
    "multi_llms_ChatGPT"
    "multi_llms_Claude-instant"
    "multi_llms_Google-PaLM"
    "multi_llms_Llama-2-70b"
)


data_seeds=(33)

EPOCHS=15
OFFSET=1
BATCH_SIZE=12
LR=2e-5

T5_MODEL_PATH="models/T5_base/t5-base_group_6"
ROBERTA_MODEL_PATH="models/roberta-base"


VAL_RATIO=0.1
SPLIT_SEED=2027
TRAIN_SEED=2027


PERTURB_MODE="B"             # A / B
SOFTMIX_LAMBDA=0.5
GUMBEL_TAU=1.0
SOFTMIX_MAX_SENTENCES=8
SOFTMIX_T5_GRAD=0


LOG_DIR="logs/PRoTeD_train"
mkdir -p "$LOG_DIR"

for dataset in "${datasets[@]}"; do
  for dseed in "${data_seeds[@]}"; do
    DATA_PATH="data/final_training_data/${dataset}-seed${dseed}-train.jsonl"
    EVAL_PATH="data/final_training_data/${dataset}-test.jsonl"
    SAVE_DIR="saved_models/PRoTeD/${dataset}_run/seed${dseed}_{$SPLIT_SEED}"


    if [[ ! -f "$DATA_PATH" ]]; then
      echo "⚠️  Skip ${dataset} seed${dseed}: train file not found: ${DATA_PATH}"
      continue
    fi
    if [[ ! -f "$EVAL_PATH" ]]; then
      echo "⚠️  Skip ${dataset} seed${dseed}: test file not found: ${EVAL_PATH}"
      continue
    fi

    ARGS="--data_path $DATA_PATH \
    --eval_path $EVAL_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --save_dir $SAVE_DIR \
    --offset $OFFSET \
    --t5_model_path $T5_MODEL_PATH \
    --roberta_model_path $ROBERTA_MODEL_PATH \
    --perturb_mode $PERTURB_MODE \
    --softmix_lambda $SOFTMIX_LAMBDA \
    --gumbel_tau $GUMBEL_TAU \
    --softmix_max_sentences $SOFTMIX_MAX_SENTENCES \
    --val_ratio $VAL_RATIO \
    --split_seed $SPLIT_SEED \
    --train_seed $TRAIN_SEED"


    if [[ "$SOFTMIX_T5_GRAD" -eq 1 ]]; then
      ARGS="$ARGS --softmix_t5_grad"
    fi

    echo "🚀 启动训练：$dataset (data_seed=${dseed})"
    echo "python -m src.experiment.PRoTeD_train $ARGS"

    ts=$(date +"%Y%m%d_%H%M%S")
    log_file="${LOG_DIR}/${dataset}_seed${dseed}_${ts}.log"

    python -m src.experiment.PRoTeD_train $ARGS 2>&1 | tee "$log_file"

    echo "✅ 完成训练：$dataset (data_seed=${dseed})"
    echo "-----------------------------------------------"
  done
done
