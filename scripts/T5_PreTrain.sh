#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# python -m src.T5_Pretraining.T5_small_Training

# python -m src.T5_Pretraining.T5_base_Training

python -m src.T5_Pretraining.T5_large_Training
