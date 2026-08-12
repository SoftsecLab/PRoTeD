#!/bin/bash


export CUDA_VISIBLE_DEVICES=01
export TRANSFORMERS_OFFLINE=1


DATASETS=(
    # "yelp-polish"
  # "xsum-polish"
  # "multi_full"
  # "multi_domains_yelp_review-polish"
  "tiny"

#  "ieee-generation"
  # "ieee-polish"

  # "multi_domains_arxiv"
#  "multi_domains_xsum"
  # "multi_domains_writing_prompt"
  # "multi_domains_yelp_review"

    # "multi_llms_ChatGPT"
    # "multi_llms_Claude-instant"
    # "multi_llms_Google-PaLM"
    # "multi_llms_Llama-2-70b"
)


DATA_DIR="/home/jxy/my_project/Detectors/human0llm1"


BASE_MODEL="/home/jxy/my_project/models/gpt-neo-2.7B"
MASK_MODEL="/home/jxy/my_project/models/T5_base/t5-base"
ROBERTA_MODEL="/home/jxy/my_project/models/roberta-base-openai-detector"
LLAMA3_MODEL="/home/jxy/my_project/models/Meta-Llama-3-8B-Instruct"
GPT_MEDIUM="/home/jxy/my_project/models/gpt2-medium"


NTRAIN=3600
echo ">>> 自动设置 NTRAIN 为 $NTRAIN"


for DATASET in "${DATASETS[@]}"; do
  TRAIN_DATA="${DATA_DIR}/${DATASET}/${DATASET}-train.json"
  TEST_DATA="${DATA_DIR}/${DATASET}/${DATASET}-test.json"

  echo "=================== Running on Dataset: $DATASET ==================="

  echo "=== Running LRR Evaluation ==="
  python -m baseline.Detectors.LRR_evaluation \
    --test_data_path $TEST_DATA \
    --base_model $BASE_MODEL \
    --DEVICE cuda

  echo "=== Running ENTROPY Evaluation ==="
  python -m baseline.Detectors.entropy_evaluation \
    --test_data_path $TEST_DATA \
    --base_model $BASE_MODEL \
    --DEVICE cuda

  echo "=== Running LIKELIHOOD Evaluation ==="
  python -m baseline.Detectors.likelihood_evaluation \
    --test_data_path $TEST_DATA \
    --base_model $BASE_MODEL \
    --DEVICE cuda

  echo "=== Running LOGRANK Evaluation ==="
  python -m baseline.Detectors.logRank_evaluation \
    --test_data_path $TEST_DATA \
    --base_model $BASE_MODEL \
    --DEVICE cuda

  echo "=== Running RANK Evaluation ==="
  python -m baseline.Detectors.rank_evaluation \
    --test_data_path $TEST_DATA \
    --base_model $BASE_MODEL \
    --DEVICE cuda

  echo "=== Running ROBERTA Evaluation ==="
  python -m baseline.Detectors.roberta_evaluation \
    --test_data_path $TEST_DATA \
    --model_name $ROBERTA_MODEL \
    --DEVICE cuda

  echo "=== Running FAST_DETECTGPT Evaluation ==="
  python -m baseline.Detectors.Fast_DetectGPT_evaluation \
    --test_data_path $TEST_DATA \
    --reference_model $BASE_MODEL \
    --scoring_model $BASE_MODEL \
    --DEVICE cuda \
    --seed 42

  echo "=== Running DETECTGPT_EVALUATION_NEW Evaluation ==="
  python -m baseline.Detectors.DetectGPT_evaluation_new \
    --train_data_path $TRAIN_DATA \
    --test_data_path $TEST_DATA \
    --base_model $BASE_MODEL \
    --mask_model $MASK_MODEL \
    --ntrain $NTRAIN \
    --n_perturbation_list "[20]" \
    --span_length 3 \
    --buffer_size 1 \
    --pct_words_masked 0.3 \
    --mask_top_p 1.0 \
    --device cuda

  echo "=== Running NPR Evaluation ==="
  python -m baseline.Detectors.NPR_evaluation \
   --test_data_path $TEST_DATA \
   --base_model $BASE_MODEL \
   --mask_model $MASK_MODEL \
   --DEVICE cuda


  echo "=== Running DNA_GPT Evaluation ==="
  python -m baseline.Detectors.dna_gpt_evaluation \
    --test_data_path $TEST_DATA \
    --base_model $BASE_MODEL \
    --regen_model $BASE_MODEL \
    --DEVICE cuda

  echo "=================== Finished Dataset: $DATASET ==================="
  echo ""
done


DATASETS=(
#   "ieee-generation"
#   "ieee-polish"
#   "multi_domains_arxiv"
#   "multi_domains_xsum"
#   "multi_domains_writing_prompt"
#   "multi_domains_yelp_review"
    # "multi_llms_ChatGPT"
    # "multi_llms_Claude-instant"
    # "multi_llms_Google-PaLM"
    # "multi_llms_Llama-2-70b"

)

for DATASET in "${DATASETS[@]}"; do
  TRAIN_DATA="${DATA_DIR}/${DATASET}/${DATASET}-train.json"
  TEST_DATA="${DATA_DIR}/${DATASET}/${DATASET}-test.json"

  echo "=================== Running on Dataset: $DATASET ==================="

  echo "=== Running DNA_GPT Evaluation ==="
  python -m baseline.Detectors.dna_gpt_evaluation \
    --test_data_path $TEST_DATA \
    --base_model $BASE_MODEL \
    --regen_model $BASE_MODEL \
    --DEVICE cuda

  echo "=================== Finished Dataset: $DATASET ==================="
  echo ""
done
