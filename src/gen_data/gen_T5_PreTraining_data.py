# python -m src.preprocess.sentence_shuffler_fn
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from nltk import ngrams
from nltk.metrics import edit_distance
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer, util

from src.utils.jsonl_handler import read_jsonl, save_results
from src.preprocess.sentence_splitter import split_sentences

# =========================

# =========================
MODEL_NAME = "models/instructor_large"
INPUT_PATH = "data/train_pairs/ieee-merged-balanced.jsonl"
OUTPUT_PATH = "data/train_pairs/grouped_shuffle_all.jsonl"
SEED = 42


def make_instance(item_id: str, shuffled: str, strategy: str, original: str, group_id: int) -> dict:

    hash_part = hashlib.md5(shuffled.encode()).hexdigest()[:8]
    return {
        "sentence_id": item_id,
        "instance_id": f"{item_id}_{strategy}_{hash_part}",
        "original": original,
        "shuffled": shuffled,
        "metadata": {"strategy": strategy, "group_id": group_id},
    }

# def _load_sbert(model_name: str):
#     if not os.path.isdir(model_name):

#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     print(f"[SBERT] Loading local model from {model_name} on {device}")
#     model= SentenceTransformer(model_name, device=device)
#     print(model)
#     return model


def _encode_cached(model, text: str, cache: Dict[str, torch.Tensor]):
    if text not in cache:
        cache[text] = model.encode(text, convert_to_tensor=True)
    return cache[text]

def evaluate_dissimilarity(original: str, shuffled: str, model, emb_cache: Dict[str, torch.Tensor]) -> Dict[str, float]:

    emb_orig = _encode_cached(model, original, emb_cache)
    emb_shuf = model.encode(shuffled, convert_to_tensor=True)

    semantic_sim = util.pytorch_cos_sim(emb_orig, emb_shuf).item()

    def ngram_overlap(a: str, b: str, n: int = 2):
        a_ngrams = Counter(ngrams(a.split(), n))
        b_ngrams = Counter(ngrams(b.split(), n))
        inter = sum((a_ngrams & b_ngrams).values())
        denom = max(len(a_ngrams), len(b_ngrams), 1)
        return inter / denom

    bigram_sim = ngram_overlap(original, shuffled, n=2)

    def norm_edit_distance(a: str, b: str):
        aw, bw = a.split(), b.split()
        m = max(len(aw), len(bw), 1)
        return edit_distance(aw, bw) / m

    ned = norm_edit_distance(original, shuffled)


    combined = 0.6 * (1 - semantic_sim) + 0.3 * (1 - bigram_sim) + 0.1 * ned
    return {"semantic_similarity": semantic_sim, "bigram_similarity": bigram_sim, "combined_score": combined}


def shuffle_with_target_tau(words: List[str], target_tau: float, rng: random.Random, max_trials: int = 100) -> List[str]:

    n = len(words)
    base = list(range(n))
    best_perm = base[:]
    best_tau = -2
    for _ in range(max_trials):
        perm = rng.sample(base, n)
        tau, _ = kendalltau(base, perm)
        if best_tau == -2 or abs(tau - target_tau) < abs(best_tau - target_tau):
            best_tau = tau
            best_perm = perm
        if abs(best_tau - target_tau) < 0.05:
            break
    return [words[i] for i in best_perm]

def apply_tau_shuffle(sentence: str, tau: float, rng: random.Random, max_trials: int = 100) -> str:
    words = sentence.split()
    if len(words) <= 1:
        return sentence
    shuffled = shuffle_with_target_tau(words, tau, rng, max_trials)
    return " ".join(shuffled)

def apply_random_shuffle(sentence: str, rng: random.Random) -> str:
    words = sentence.split()
    rng.shuffle(words)
    return " ".join(words)

def aggressive_shuffle_once(words: List[str], rng: random.Random, p_char: float = 0.2) -> List[str]:

    out = words[:]
    for i, w in enumerate(out):
        if len(w) > 3 and rng.random() < p_char:
            out[i] = "".join(rng.sample(list(w), len(w)))
    rng.shuffle(out)
    return out

def apply_aggressive_shuffle(sentence: str, model, emb_cache: Dict[str, torch.Tensor], rng: random.Random,
                             trials: int = 30, early_stop: float = 0.15) -> str:

    words = sentence.split()
    best_text = sentence
    best_score = float("inf")

    for _ in range(trials):
        cand = " ".join(aggressive_shuffle_once(words, rng))
        score = evaluate_dissimilarity(sentence, cand, model, emb_cache)["combined_score"]
        if score < best_score:
            best_score = score
            best_text = cand
        if best_score < early_stop:
            break
    return best_text


def segment_chunks_by_wordcount(data: List[dict]) -> List[List[dict]]:

    chunks, cur, last_wc = [], [], -1
    for item in data:
        wc = item.get("word_count", 0)
        if last_wc != -1 and wc < last_wc and cur:
            chunks.append(cur)
            cur = []
        cur.append(item)
        last_wc = wc
    if cur:
        chunks.append(cur)
    return chunks

def generate_grouped_shuffle(data: List[dict], model, rng: random.Random) -> List[dict]:

    grouped = segment_chunks_by_wordcount(data)
    assert len(grouped) >= 10, f"分组不足 10 组，当前：{len(grouped)}"

    base = ["tau_0.2"] * 2 + ["tau_0.5"] * 2 + ["tau_0.8"] * 2 + ["random"] * 2 + ["random"] * 2
    # base = ["tau_0.2"] * 2 + ["tau_0.5"] * 2 + ["tau_0.8"] * 2 + ["random"] * 2 + ["aggressive"] * 2
    num_groups = len(grouped) // 10
    strategies_per_group = [base.copy() for _ in range(num_groups)]

    emb_cache: Dict[str, torch.Tensor] = {}

    results = []
    for g in tqdm(range(num_groups), desc="Processing Groups"):
        chunk_indices = list(range(g * 10, (g + 1) * 10))
        strategy_assignment = strategies_per_group[g]
        rng.shuffle(strategy_assignment)

        for idx, ch_idx in enumerate(chunk_indices):
            if ch_idx >= len(grouped):
                continue
            strat = strategy_assignment[idx]
            data_chunk = grouped[ch_idx]

            for item in tqdm(data_chunk, leave=False, desc=f"Chunk {ch_idx} -> {strat}"):
                sent = item.get("sentence", "")
                sid = item.get("sentence_id", "")
                if not sent:
                    continue

                if strat.startswith("tau_"):
                    tau_val = float(strat.split("_")[1])
                    shuf = apply_tau_shuffle(sent, tau=tau_val, rng=rng)
                elif strat == "random":
                    shuf = apply_random_shuffle(sent, rng=rng)
                elif strat == "aggressive":
                    shuf = apply_aggressive_shuffle(sent, model=model, emb_cache=emb_cache, rng=rng)
                else:
                    continue

                results.append(make_instance(sid, shuf, strat, sent, group_id=g))

    return results



def generate_t5_training_dataset():

    # print(1)
    rng = random.Random(SEED)
    # print(11)
    # model = _load_sbert(MODEL_NAME)


    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # print(f"[SBERT] Loading local model from {MODEL_NAME} on {device}")
    model= SentenceTransformer(MODEL_NAME)
    # print(model)
    # print(2)
    # data = read_jsonl(INPUT_PATH)
    data=read_data()
    # print(3)
    results = generate_grouped_shuffle(data, model=model, rng=rng)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_results(results, OUTPUT_PATH)

    print(f"✅ 共生成训练样本 {len(results)} 条 -> {OUTPUT_PATH}")



def read_data():
    all_data=[]
    for fname in ["data/raw/CHEAT/ieee-init.jsonl", "data/raw/CHEAT/ieee-chatgpt-generation.jsonl"]:
        data = read_jsonl(fname,max_records=1000)
        results = split_sentences(data, auto_threshold=True, threshold_strategy="percentile")
        # output_file = fname.replace(".jsonl", "_split.jsonl").replace("data/raw", "data/preprocess")
        # save_results(results, output_file)

        # print(results)
        all_data.extend(results)
        # print(len(all_data))
    return all_data



if __name__ == "__main__":
    generate_t5_training_dataset()
