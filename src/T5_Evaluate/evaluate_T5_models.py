# python -m scripts.reorderAndmetrics
import os
import numpy as np
from src.utils.jsonl_handler import read_jsonl, save_results
from src.preprocess.sentence_splitter import split_sentences
from src.preprocess.sentence_shuffler import generate_all_shuffles, apply_random_shuffle
from src.preprocess.sentence_reorder import reorder_with_all_models
from src.evaluator.metrics_evaluator import compute_metrics_for_batch
from src.utils.pretty_print import pretty_print

INPUT_FILE = "data/raw/arxiv_2800.jsonl"
OUTPUT_DIR = "data/T5_evaluate/models_rank_10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

shuffle_sentences =read_jsonl("data/T5_evaluate/models_rank_10/shuffle_data.jsonl")
all_model_outputs = reorder_with_all_models(shuffle_sentences)
all_outputs_path = os.path.join(OUTPUT_DIR, "All_model_outputs.jsonl")
save_results(all_model_outputs, all_outputs_path)
print(f"✅ 重组数据完成，模型输出数量: {len(all_model_outputs)}")

output_dir = "data/T5_evaluate/models_rank_10"
metrics=[]
for model in all_model_outputs:

    model_name = f"{model['model']}_{model['group_id']}"
    print(f"正在评估模型: {model_name}...")
    results=model["results"]
    medel_metrics = compute_metrics_for_batch(results, use_ppl=True)
    tmp ={
        "model": model_name,
        "metrics": medel_metrics
    }
    metrics.append(tmp)
    #save_results(tmp, os.path.join(output_dir, f"{model_name}_metrics.jsonl"))
    print(f"模型 {model_name} 评估完成，结果已保存。")

save_results(metrics, "data/T5_evaluate/models_rank_10/All_metrics_eval_results.jsonl")
print(f"✅ 评估结果构造完成，模型数量: {len(metrics)}")
