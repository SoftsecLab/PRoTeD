# python -m src.preprocess.sentence_shuffler
import random
import hashlib
from collections import Counter
from nltk import ngrams
from nltk.metrics import edit_distance
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from scipy.stats import kendalltau
from src.utils.pretty_print import pretty_print
from src.utils.jsonl_handler import read_jsonl, save_results
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_instance(item_id: str, shuffled: str, strategy: str, original: str,group_id: str) -> dict:

    hash_part = hashlib.md5(shuffled.encode()).hexdigest()[:8]
    return {
        "sentence_id": item_id,
        "instance_id": f"{item_id}_{strategy}_{hash_part}",
        "original": original,
        "shuffled": shuffled,
        "metadata": {
            "strategy": strategy,
            "group_id": group_id
            }
    }


class SentenceShuffler:
    def __init__(self, model_name="./models/instructor_large"):


        self.model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        # self.model = SentenceTransformer(model_name)
        """ device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device) """


    def shuffle_with_target_tau(self, words, target_tau, max_trials=100):

        best = words[:]
        best_tau = -1
        for _ in range(max_trials):
            shuffled = words[:]
            random.shuffle(shuffled)
            tau, _ = kendalltau(range(len(words)), [shuffled.index(w) for w in words])
            if abs(tau - target_tau) < abs(best_tau - target_tau):
                best = shuffled
                best_tau = tau
            if abs(best_tau - target_tau) < 0.05:
                break
        return best

    def aggressive_shuffle(self, text, p_char=0.2):

        words = text.split()
        for i in range(len(words)):
            if random.random() < p_char and len(words[i]) > 3:
                words[i] = ''.join(random.sample(words[i], len(words[i])))
        random.shuffle(words)
        return ' '.join(words)

    def run_best_shuffle(self, text):

        best_score = float("inf")
        best_output = text
        for _ in range(30):
            shuffled = self.aggressive_shuffle(text)
            score = self.evaluate_dissimilarity(text, shuffled)["combined_score"]
            if score < best_score:
                best_score = score
                best_output = shuffled
        return best_output

    def evaluate_dissimilarity(self, original, shuffled):

        emb_orig = self.model.encode(original, convert_to_tensor=True)
        emb_shuf = self.model.encode(shuffled, convert_to_tensor=True)
        semantic_sim = util.pytorch_cos_sim(emb_orig, emb_shuf).item()

        def ngram_overlap(a, b, n=2):
            a_ngrams = Counter(ngrams(a.split(), n))
            b_ngrams = Counter(ngrams(b.split(), n))
            intersection = sum((a_ngrams & b_ngrams).values())
            return intersection / max(len(a_ngrams), 1)

        bigram_sim = ngram_overlap(original, shuffled)

        def norm_edit_distance(a, b):
            m, n = len(a.split()), len(b.split())
            max_len = max(m, n)
            return edit_distance(a.split(), b.split()) / max_len if max_len > 0 else 0

        return {
            "semantic_similarity": semantic_sim,
            "bigram_similarity": bigram_sim,
            "combined_score": 0.6 * (1 - semantic_sim) + 0.3 * (1 - bigram_sim) + 0.1 * norm_edit_distance(original, shuffled)
        }


    def apply_tau_shuffle(self, sentence, tau=0.5):

        words = sentence.split()
        shuffled = " ".join(self.shuffle_with_target_tau(words, tau))
        return shuffled

    def apply_random_shuffle(self, sentence):

        words = sentence.split()
        random.shuffle(words)
        shuffled = " ".join(words)
        return shuffled

    def apply_aggressive_shuffle(self, sentence):

        return self.run_best_shuffle(sentence)



def generate_all_shuffles(data):

    print("generate_all_shuffles")
    shuffler = SentenceShuffler()
    print("generate_all_shuffles")
    all_results = []
    for item in tqdm(data, desc="Generating Shuffles"):
        sentence = item.get("sentence", "")
        results = {
            "sentence_id": item.get("sentence_id"),
            "original": sentence,
            "shuffled": {
                "tau_0.2": shuffler.apply_tau_shuffle(sentence, tau=0.2),
                "tau_0.5": shuffler.apply_tau_shuffle(sentence, tau=0.5),
                "tau_0.8": shuffler.apply_tau_shuffle(sentence, tau=0.8),
                "random": shuffler.apply_random_shuffle(sentence),
                "aggressive": shuffler.apply_aggressive_shuffle(sentence)
            }
        }
        all_results.append(results)
    return all_results



def generate_t5_training_dataset():


    def generate_grouped_shuffle(data, shuffler):






        grouped_data = segment_chunks_by_wordcount(data)
        print(f"分组数量: {len(grouped_data)}")
        assert len(grouped_data) >= 10, "分组不足 10 组，数据过少"
        total_results = []
        strategies_per_group = [
            ["tau_0.2"] * 2 + ["tau_0.5"] * 2 + ["tau_0.8"] * 2 + ["random"] * 2 + ["aggressive"] * 2
        ] * (len(grouped_data) // 10)

        for group_id in tqdm(range(len(strategies_per_group)), desc="Processing Groups"):
            print(f"正在处理第 {group_id} 组")

            chunk_indices = list(range(group_id * 10, (group_id + 1) * 10))
            strategy_assignment = strategies_per_group[group_id]
            random.shuffle(strategy_assignment)

            for idx, chunk_idx in enumerate(chunk_indices):
                if chunk_idx >= len(grouped_data):
                    continue
                strategy = strategy_assignment[idx]
                data_chunk = grouped_data[chunk_idx]
                # pretty_print(data_chunk[:5])
                shuffled_data = []
                if strategy.startswith("tau_"):
                    tau_val = float(strategy.split("_")[1])
                    for item in tqdm(data_chunk,desc=f"Processing Sentences {strategy}:"):
                        """ print(item)
                        input("Press Enter to continue...") """

                        tmp=make_instance(item["sentence_id"], shuffler.apply_tau_shuffle(item["sentence"], tau=tau_val), strategy, item["sentence"],group_id=group_id)
                        shuffled_data.append(tmp)
                        # pretty_print(shuffled_data)
                    # shuffled_data = shuffler.apply_tau_shuffle(data_chunk, tau=tau_val)
                elif strategy == "random":
                    for item in tqdm(data_chunk,desc=f"Processing Sentences {strategy}:"):

                        tmp=make_instance(item["sentence_id"], shuffler.apply_random_shuffle(item["sentence"]), strategy, item["sentence"],group_id=group_id)
                        shuffled_data.append(tmp)
                        # pretty_print(shuffled_data)

                elif strategy == "aggressive":
                    for item in tqdm(data_chunk,desc=f"Processing Sentences {strategy}:"):

                        tmp=make_instance(item["sentence_id"], shuffler.apply_aggressive_shuffle(item["sentence"]), strategy, item["sentence"],group_id=group_id)
                        shuffled_data.append(tmp)
                        # pretty_print(shuffled_data)

                else:
                    continue



                total_results.extend(shuffled_data)

        return total_results



    def segment_chunks_by_wordcount(data, target_chunks=100):






        chunks = []
        current_chunk = []
        last_wc = -1
        for item in data:
            wc = item.get("word_count", 0)
            if last_wc != -1 and wc < last_wc and len(current_chunk) > 0:
                chunks.append(current_chunk)
                current_chunk = []
            current_chunk.append(item)
            last_wc = wc
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    input_path = "data/train_pairs/ieee-merged-balanced.jsonl"
    output_path = "data/train_pairs/grouped_shuffle_all.jsonl"
    all_data = read_jsonl(input_path)
    # pretty_print(all_data[:5])
    shuffler = SentenceShuffler()
    results_grouped = generate_grouped_shuffle(all_data,shuffler)
    save_results(results_grouped, output_path)

    print(f"✅ 共生成训练样本 {len(results_grouped)} 条")
    for item in results_grouped[:10]:
        print(item)
def generate_classification_dataset():
    print("generate_classification_dataset")
    for fname in ["data/classification_dataset/processed/ieee-init_split.jsonl", "data/classification_dataset/processed/ieee-chatgpt-generation_split.jsonl",
                  #"data/classification_dataset/ieee-chatgpt-polish_split.jsonl","data/classification_dataset/ieee-chatgpt-fusion_split.jsonl"
                  ]:
        split_data = read_jsonl(fname,max_records=1000)
        shuffle_sentences = generate_all_shuffles(split_data)
        shuffle_path = fname.replace(".jsonl", "_shuffle.jsonl")
        # save_results(shuffle_sentences, shuffle_path)
        print(shuffle_sentences[:5])

if __name__ == "__main__":

    generate_classification_dataset()