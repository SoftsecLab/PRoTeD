#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


datasets=(
    # "tiny"
    # "ieee-generation"
    "ieee-polish"

    # "multi_domains_arxiv"
    # "multi_domains_xsum"
    # "multi_domains_writing_prompt"
    # "multi_domains_yelp_review"

    # "multi_llms_ChatGPT"
    # "multi_llms_Claude-instant"
    # "multi_llms_Google-PaLM"
    # "multi_llms_Llama-2-70b"
)

EPOCHS=10
OFFSET=6
BATCH_SIZE=12
LR=2e-5


T5_MODEL_PATH="models/T5_base/t5-base_group_6"
ROBERTA_MODEL_PATH="models/roberta-base"


for dataset in "${datasets[@]}"; do

    DATA_PATH="data/final_training_data/${dataset}-train.jsonl"
    EVAL_PATH="data/final_training_data/${dataset}-test.jsonl"
    MODEL_DIR="saved_models/PRoTeD/${dataset}_run/best_model"
    SAVE_DIR="saved_models/PRoTeD/${dataset}_run/explain"

    ARGS="--model_dir $MODEL_DIR \
    --save_dir $SAVE_DIR \
    --eval_path $EVAL_PATH"


    echo "🚀 启动训练：$dataset"
    echo "python -m src.experiment.PRoTeD_train $ARGS"
    python -m src.experiment.PRoTeD_test_explain $ARGS
    echo "✅ 完成训练：$dataset"
    echo "-----------------------------------------------"
done
