#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="data/raw/DetectRL_original"
OUTPUT_DIR="data/raw/DetectRL_clear"

OUTPUT_DIR1="data/raw/DetectRL_clear1"
OUTPUT_DIR2="data/raw/DetectRL_clear2"
OUTPUT_DIR3="data/raw/DetectRL_clear3"
OUTPUT_DIR4="data/raw/DetectRL_clear4"
OUTPUT_DIR5="data/raw/DetectRL_clear5"
OUTPUT_DIR6="data/raw/DetectRL_clear6"
# python -m src.utils.clean_detectrl_dir1 \
#   --input_dir  "${INPUT_DIR}" \
#   --output_dir "${OUTPUT_DIR1}"


# python -m src.utils.clean_detectrl_dir2 \
#   --input_dir  "${OUTPUT_DIR1}" \
#   --output_dir "${OUTPUT_DIR2}"

# python -m src.utils.clean_detectrl_dir3 \
#   --input_dir  "${OUTPUT_DIR2}" \
#   --output_dir "${OUTPUT_DIR3}"

# python -m src.utils.clean_detectrl_dir4 \
#   --input_dir  "${OUTPUT_DIR3}" \
#   --output_dir "${OUTPUT_DIR4}"

# python -m src.utils.clean_detectrl_dir5 \
#   --input_dir  "${OUTPUT_DIR4}" \
#   --output_dir "${OUTPUT_DIR5}"

# python -m src.utils.clean_detectrl_dir6 \
#   --input_dir  "${OUTPUT_DIR5}" \
#   --output_dir "${OUTPUT_DIR6}"

# python -m src.utils.clean_detectrl_dir \
#   --input_dir  "${INPUT_DIR}" \
#   --output_dir "${OUTPUT_DIR}"

python -m src.utils.clean_detectrl --input_dir data/raw/DetectRL_original --output_dir data/raw/DetectRL_clear

# python -m src.utils.clean_compare \
#   --base_dir data/raw/DetectRL_clear7 \
#   --target_dir data/raw/DetectRL_clear