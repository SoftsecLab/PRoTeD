#!/bin/bash



# python -m src.gen_data.gen_DetectRL_data



# python -m src.gen_data.gen_CHEAT_data

# python -m src.gen_data.gen_dataset

# python -m src.gen_data.gen_DetectRL_train \
#   --input_dir  data/raw/DetectRL \
#   --output_dir data/final_training_data \
#   --seeds 11,22,33 \
#   --sample_size 50

python -m src.gen_data.gen_DetectRL_test \
  --input_dir  data/raw/DetectRL \
  --output_dir data/final_training_data \