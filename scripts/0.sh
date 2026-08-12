#!/bin/bash

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0


# The following three steps are only required to reproduce
# the data preparation process.
# The final training and test datasets are already available in:
# data/final_training_data


# # 1. Download the CHEAT and DetectRL datasets
# python src/gen_data/download_dataset.py


# # 2. Clean the original DetectRL dataset
# INPUT_DIR="data/raw/DetectRL_original"
# OUTPUT_DIR="data/raw/DetectRL"

# python -m src.utils.clean_detectrl \
#     --input_dir "${INPUT_DIR}" \
#     --output_dir "${OUTPUT_DIR}"


# # 3. Generate the training and test datasets
# python -m src.gen_data.gen_dataset


# 4. Required: perform simple pretraining of the T5-base model
# The pretraining data are already available in the train_pairs directory
python -m src.T5_Pretraining.T5_base_Training