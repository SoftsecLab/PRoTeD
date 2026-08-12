# python -m scripts.run_models_rank
import os
import numpy as np
import torch
from tqdm import tqdm
from src.utils.jsonl_handler import read_jsonl, save_results
from src.preprocess.sentence_splitter import split_sentences
from src.preprocess.sentence_shuffler import generate_all_shuffles,SentenceShuffler
from src.preprocess.sentence_reorder import reorder_with_all_models
from src.preprocess.t5_reorder_engine import reorder_one_by_one,load_model
from src.evaluator.metrics_evaluator import compute_metrics_for_batch
from src.utils.pretty_print import pretty_print

INPUT_FILE = "data/raw/CHEAT/ieee-init.jsonl"
OUTPUT_DIR = "data/T5_evaluate"
MODEL_PATH = "models/T5_base/t5-base_group_6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer, model = load_model(MODEL_PATH)
model = model.to(device)

# data = read_jsonl(INPUT_FILE, max_records=1000)





# split_data = split_sentences(data, auto_threshold=True, threshold_strategy="percentile")
# split_path = os.path.join(OUTPUT_DIR, "split_data.jsonl")
# save_results(split_data, split_path)




# shuffle_sentences = generate_all_shuffles(split_data)
# # shuffle_sentences=apply_tau_shuffle(split_data,tau=0.5)
# # shuffle_sentences = apply_random_shuffle(split_data)
# shuffle_path = os.path.join(OUTPUT_DIR, "shuffle_data.jsonl")
# save_results(shuffle_sentences, shuffle_path)



shuffle_sentences =read_jsonl("data/T5_evaluate/shuffle_data.jsonl")

reorder_results =[]
for item in tqdm(shuffle_sentences):
    inputs = []


    for key in item["shuffled"].keys():
        inputs.append(item["shuffled"][key])


    reordered_sents = reorder_one_by_one(model, tokenizer,device, inputs)


    reordered_shuffled = {key: reordered_sents[idx] for idx, key in enumerate(item["shuffled"].keys())}


    reordered_item = {
        "sentence_id": item["sentence_id"],
        "original": item["original"],
        "shuffled": item["shuffled"],
        "reorder": reordered_shuffled
    }


    reorder_results.append(reordered_item)

all_model_outputs = reorder_with_all_models(shuffle_sentences)
all_outputs_path = os.path.join(OUTPUT_DIR, "all_model_outputs.jsonl")
save_results(all_model_outputs, all_outputs_path)
print(f"✅ 重组数据完成，模型输出数量: {len(all_model_outputs)}")



eval_results = []
sub_outdir = OUTPUT_DIR +"/sub_outputs"
os.makedirs(sub_outdir, exist_ok=True)
for item in all_model_outputs:
    model_name=f"{item['model']}_{item['group_id']}"
    print(f"正在评估模型: {model_name}...")
    result = item["results"]
    medel_metrics = compute_metrics_for_batch(result)
    tmp ={
            "model": model_name,
            "metrics": medel_metrics
        }
    eval_results.append({
        "model": model_name,
        "metrics": medel_metrics,
    })


    save_results(tmp, os.path.join(sub_outdir, f"{model_name}_metrics.jsonl"))
    print(f"模型 {model_name} 评估完成，结果已保存。")
print(f"✅ 评估结果构造完成，模型数量: {len(eval_results)}")



eval_path = os.path.join(OUTPUT_DIR, "metrics_eval_results.jsonl")
save_results(eval_results, eval_path)
print(f"✅ 评估结果已保存至 {eval_path}")


