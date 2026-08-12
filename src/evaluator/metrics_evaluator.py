import json
import os
import textstat
import math
import spacy
import nltk
nltk.data.path.append('models/nltk_data')
from nltk import ngrams
import Levenshtein
from torchmetrics.functional.text import bert_score
from bert_score import score as bert_score
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
from sentence_transformers import SentenceTransformer
from src.utils.gpt2_ppl import compute_llscore_ppl
from src.utils.pretty_print import pretty_print
from tqdm import tqdm
from src.utils.jsonl_handler import read_jsonl, save_results
from nltk.corpus import cmudict
import numpy as np
from src.preprocess.sentence_splitter import SentenceSegmenter
import time

d = cmudict.dict()

nlp = spacy.load("en_core_web_sm")
semantic_model = SentenceTransformer('./models/paraphrase-MiniLM-L6-v2')
segmenter = SentenceSegmenter()



# ============================

# ============================

def compute_sentence_length_variability(sentence_list):

    sentence_lengths = [len(sent.split()) for sent in sentence_list]
    if len(sentence_lengths) < 2:
        return 0.0
    return float(np.std(sentence_lengths) / (np.mean(sentence_lengths) + 1e-5))

def compute_avg_sentence_length(sentence_list):

    if not sentence_list:
        return 0.0
    return float(np.mean([len(sent.split()) for sent in sentence_list]))

def compute_sentence_complexity_index(sentence_list):

    total_complexity = 0
    for sentence in sentence_list:
        doc = nlp(sentence)
        clauses = [tok for tok in doc if tok.dep_ in {"ccomp", "acl", "advcl"}]
        total_complexity += len(clauses)
    return total_complexity / max(len(sentence_list), 1)

def compute_dependency_length(sentence_list):

    total_length = 0
    total_edges = 0
    for sent in sentence_list:
        doc = nlp(sent)
        total_length += sum(abs(tok.i - tok.head.i) for tok in doc if tok.head != tok)
        total_edges += sum(1 for tok in doc if tok.head != tok)
    return total_length / max(total_edges, 1)

def compute_semantic_cohesion(sentence_list):

    if len(sentence_list) <= 1:
        return 0
    embeddings = semantic_model.encode(sentence_list)
    cohesion = sum(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0] for i in range(len(embeddings) - 1))
    return float(cohesion / (len(embeddings) - 1))



# ============================

# ============================

def compute_entropy_density(text):

    prob_dist = Counter(text)
    total = len(text)
    return float(sum(-freq/total * math.log2(freq/total) for freq in prob_dist.values()))

def compute_compression_density(text):

    tokens = text.split()
    return len(set(tokens)) / (len(tokens) + 1e-5)

def compute_syntactic_density(text):

    doc = nlp(text)
    return sum(1 for tok in doc if not tok.is_punct) / max(len(list(doc.sents)), 1)

def flesch_kincaid_score(text):


    return textstat.flesch_reading_ease(text)

def gunning_fog_index(text):


    return textstat.gunning_fog(text)


# ============================

# ============================

def compute_hierarchical_density(text):

    doc = nlp(text)
    return sum(1 for tok in doc if tok.dep_ in {"prep", "relcl", "conj"}) / max(len(list(doc.sents)), 1)

def compute_syntactic_errors(text):

    doc = nlp(text)
    return sum(1 for tok in doc if tok.pos_ == "PUNCT" and tok.dep_ not in {"punct", "ROOT"})


# ============================

# ============================

def compute_lexical_density(text):

    doc = nlp(text)
    content_words = [tok for tok in doc if tok.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
    return len(content_words) / (len(doc) + 1e-5)

def compute_lexical_diversity(text):

    tokens = text.split()
    return len(set(tokens)) / (len(tokens) + 1e-5)

def compute_word_frequency_entropy(text):
    words = text.split()
    freq = Counter(words)
    probs = [count / len(words) for count in freq.values()]
    return -sum(p * math.log2(p) for p in probs)

def compute_syllable_density(text):


    tokens = text.split()
    return np.mean([textstat.syllable_count(word) for word in tokens])

def compute_perplexity(text):
    _, ppl = compute_llscore_ppl(text)
    return round(ppl, 2)



# ============================

# ============================

def compute_linguistic_metrics(text):
    sentence_list = [s for s, _ in segmenter.segment(text)]
    metrics = {
        # L1
        "SentenceLengthVariability": round(compute_sentence_length_variability(sentence_list),4),
        "AvgSentenceLength": round(compute_avg_sentence_length(sentence_list),4),
        "SentenceComplexityIndex": round(compute_sentence_complexity_index(sentence_list),4),
        "DependencyLength": round(compute_dependency_length(sentence_list),4),
        "SemanticCohesion": round(compute_semantic_cohesion(sentence_list),4),

        # L2
        "EntropyDensity": round(compute_entropy_density(text),4),
        "CompressionDensity": round(compute_compression_density(text),4),
        "SyntacticDensity": round(compute_syntactic_density(text),4),
        "FleschKincaidScore": round(flesch_kincaid_score(text),4),
        "GunningFogScore": round(gunning_fog_index(text),4),

        # L3
        "HierarchicalDensity": round(compute_hierarchical_density(text),4),
        "SyntacticErrors": round(compute_syntactic_errors(text),4),

        # L4
        "LexicalDensity": round(compute_lexical_density(text),4),
        "LexicalDiversity": round(compute_lexical_diversity(text),4),
        "WordFrequencyEntropy":round(compute_word_frequency_entropy(text),4),
        "SyllableDensity": round(compute_syllable_density(text),4),
        "PPL": round(compute_perplexity(text),4)
    }
    return metrics


def group_metrics_by_level(metrics):

    return {
        "L1": {k: metrics[k] for k in [
            "SentenceLengthVariability", "AvgSentenceLength",
            "SentenceComplexityIndex", "DependencyLength", "SemanticCohesion"]},
        "L2": {k: metrics[k] for k in [
            "EntropyDensity", "CompressionDensity", "SyntacticDensity",
            "FleschKincaidScore", "GunningFogScore"]},
        "L3": {k: metrics[k] for k in [
            "HierarchicalDensity", "SyntacticErrors"]},
        "L4": {k: metrics[k] for k in [
            "LexicalDensity", "LexicalDiversity", "WordFrequencyEntropy",
            "SyllableDensity", "PPL"]}
    }






def compute_text_match_metrics(pred, ref):
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
    meteor = meteor_score([ref.split()], pred.split())
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(ref, pred)

    bert_P, bert_R, bert_F1 = bert_score(
        [pred], [ref],
        lang='en',
        model_type='./models/bert-base-uncased',
        num_layers=12,
        verbose=False
    )


    return {
        "BLEU": round(bleu, 4),
        "METEOR": round(meteor, 4),
        "ROUGE-1": round(rouge_scores['rouge1'].fmeasure, 4),
        "ROUGE-2": round(rouge_scores['rouge2'].fmeasure, 4),
        "ROUGE-3": round(rouge_scores['rouge3'].fmeasure, 4),
        "ROUGE-L": round(rouge_scores['rougeL'].fmeasure, 4),
        "BERTScore": round(float(bert_F1[0]), 4)
    }



def compute_similarity_metrics(pred, ref):
    pred_vec= semantic_model.encode(pred)
    ref_vec=semantic_model.encode(ref)

    pred_vec_2d = pred_vec.reshape(1, -1)
    ref_vec_2d = ref_vec.reshape(1, -1)
    pred_vec_flat = pred_vec.flatten()
    ref_vec_flat = ref_vec.flatten()
    return {
        "CosineSim": round(float(cosine_similarity(pred_vec_2d, ref_vec_2d)[0][0]), 4),
        "EuclideanDist": round(float(euclidean(pred_vec_flat, ref_vec_flat)), 4),
        "ManhattanDist": round(float(cityblock(pred_vec_flat, ref_vec_flat)), 4)
    }


def compute_semantic_consistency(pred, ref):
    pred_embedding = semantic_model.encode([pred])
    ref_embedding = semantic_model.encode([ref])
    cosine_sim = cosine_similarity(pred_embedding, ref_embedding)[0][0]
    return float(cosine_sim)

def compute_structural_similarity_metrics(pred, ref):
    def tokenize(text):
        return [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_space]

    def pos_tags(text):
        return [token.pos_ for token in nlp(text)]

    def get_ngrams(tokens, n=2):
        return list(ngrams(tokens, n))

    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)


    token_set_pred = set(pred_tokens)
    token_set_ref = set(ref_tokens)
    jaccard_sim = len(token_set_pred & token_set_ref) / len(token_set_pred | token_set_ref) if token_set_pred | token_set_ref else 0


    common_tokens = len(set(pred_tokens) & set(ref_tokens))
    token_overlap = common_tokens / len(set(ref_tokens)) if ref_tokens else 0


    edit_distance = Levenshtein.distance(pred, ref)
    max_len = max(len(pred), len(ref))
    norm_edit_distance = edit_distance / max_len if max_len > 0 else 0


    pred_pos = pos_tags(pred)
    ref_pos = pos_tags(ref)
    common_pos = len(set(pred_pos) & set(ref_pos))
    pos_overlap = common_pos / len(set(ref_pos)) if ref_pos else 0


    pred_sent_count = len(list(nlp(pred).sents))
    ref_sent_count = len(list(nlp(ref).sents))
    sent_count_diff = abs(pred_sent_count - ref_sent_count)


    pred_sent_lens = [len(sent) for sent in nlp(pred).sents]
    ref_sent_lens = [len(sent) for sent in nlp(ref).sents]
    avg_len_diff = abs(np.mean(pred_sent_lens) - np.mean(ref_sent_lens)) if pred_sent_lens and ref_sent_lens else 0


    pred_bigrams = set(get_ngrams(pred_tokens, n=2))
    ref_bigrams = set(get_ngrams(ref_tokens, n=2))
    bigram_overlap = len(pred_bigrams & ref_bigrams) / len(ref_bigrams) if ref_bigrams else 0

    return {
        "JaccardSimilarity": round(jaccard_sim, 4),
        "TokenOverlap": round(token_overlap, 4),
        "EditDistance": round(norm_edit_distance, 4),
        "POSOverlap": round(pos_overlap, 4),
        "SentenceCountDiff": sent_count_diff,
        "AvgSentenceLenDiff": round(avg_len_diff, 4),
        "BigramOverlap": round(bigram_overlap, 4)
    }


def compute_metrics_for_pair(pred, ref):

    result = {}


    result.update(compute_text_match_metrics(pred, ref))


    result.update(compute_similarity_metrics(pred, ref))


    result["SemanticConsistency"] = compute_semantic_consistency(pred, ref)

    result.update(compute_structural_similarity_metrics(pred, ref))
    return result


def compute_metrics_for_batch(evaluation_data, use_ppl=True, save_path=None):
    results = []

    for i, item in enumerate(tqdm(evaluation_data, desc="Evaluating", unit="item")):

        pred = item["prediction"]
        ref = item["original"]

        result = {
            "sentence_id": item.get("sentence_id"),
            "instance_id": item.get("instance_id"),
            "pair_metrics": {},
            "single_metrics": {
                "prediction": {},
                "reference": {}
            }
        }

        metrics = compute_metrics_for_pair(pred, ref)

        result["pair_metrics"].update(metrics)
        result["single_metrics"]["prediction"].update(compute_linguistic_metrics(pred))
        result["single_metrics"]["reference"].update(compute_linguistic_metrics(ref))

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

        results.append(result)

    return results



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    original="In the era of the Internet of Things (IoT), communication and computation play essential roles in the performance and effectiveness of IoT systems. One promising technology for optimizing IoT systems is device-to-device (D2D) communication. In this paper, we focus on the optimization of D2D computation offloading for precedence-constrained tasks in information-centric IoT. To achieve this goal, we introduce a task analysis framework, which considers both the computational and communication aspects of IoT systems. By integrating this framework with a computational modeling approach, we propose an energy-efficient D2D computation offloading optimization algorithm. Specifically, the proposed algorithm aims to minimize the delay and energy consumption of computing tasks, while ensuring the precedence constraints are satisfied. Simulation results demonstrate the effectiveness of the proposed algorithm in improving the performance of D2D computation offloading for IoT systems, especially for mobile handsets with limited computational resources."
    shuffled="To achieve this goal, we propose a task analysis framework considering both communication and computational aspects of IoT. The effectiveness of the proposed D2D offloading algorithm is demonstrated by simulations. In this paper, we emphasize the optimization of D2D offloading for tasks with precedence constraints. Mobile devices with constrained computing resources benefit from the proposed solution. Communication and computation are key to IoT system performance. The D2D method is a promising direction for optimization in IoT systems. An energy-aware algorithm is proposed, minimizing delay and energy consumption while meeting task constraints."
    reordered="Communication and computation are critical to the performance of IoT systems. In this paper, we focus on optimizing D2D computation offloading for precedence-constrained tasks in information-centric IoT. A task analysis framework is introduced, addressing both communication and computation aspects. We integrate this framework with computational modeling to propose an energy-efficient offloading algorithm. The algorithm minimizes delay and energy consumption while satisfying task constraints. Simulation results show that the approach significantly improves offloading performance, especially on resource-limited mobile devices. D2D communication remains a promising optimization technology in the IoT domain."
    metrics=compute_linguistic_metrics(original)
    print(len(metrics))
    pretty_print(metrics)
    level=group_metrics_by_level(metrics)
    print(level)
    # pretty_print(compute_linguistic_metrics(shuffled))
    # pretty_print(compute_linguistic_metrics(reordered))
    # pretty_print(compute_metrics_for_pair(original, shuffled))
    # pretty_print(compute_metrics_for_pair(original, reordered))
    # pretty_print(compute_metrics_for_pair(shuffled, reordered))


    exit()
    data=read_jsonl("data/classification_dataset/processed/train_data.jsonl",max_records=16)
    texts=[]
    start_time = time.time()
    for item in data:
        text = item["abstract"]
        start_time2 = time.time()
        sentences = [s for s, _ in segmenter.segment(text)]
        end_time2 = time.time()
        elapsed_time2 = end_time2 - start_time2
        print(f"分句程序运行时间: {elapsed_time2}秒")
        metricses=[]
        start_time1 = time.time()
        for sentence in sentences:
            metrics=compute_linguistic_metrics(sentence)
            metricses.append(metrics)
        texts.append((sentences, metricses))
        end_time1 = time.time()
        elapsed_time1 = end_time1 - start_time1
        print(f"指标计算程序运行时间: {elapsed_time1}秒")
    # print(texts)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time}秒")
    exit()

    # demo_data = [
    #     {
    #         "sentence_id": "8600003_0",
    #         "instance_id": "8600003_0_abcd1234",
    #         "original":"In the era of the Internet of Things (IoT), communication and computation play essential roles in the performance and effectiveness of IoT systems. One promising technology for optimizing IoT systems is device-to-device (D2D) communication. In this paper, we focus on the optimization of D2D computation offloading for precedence-constrained tasks in information-centric IoT. To achieve this goal, we introduce a task analysis framework, which considers both the computational and communication aspects of IoT systems. By integrating this framework with a computational modeling approach, we propose an energy-efficient D2D computation offloading optimization algorithm. Specifically, the proposed algorithm aims to minimize the delay and energy consumption of computing tasks, while ensuring the precedence constraints are satisfied. Simulation results demonstrate the effectiveness of the proposed algorithm in improving the performance of D2D computation offloading for IoT systems, especially for mobile handsets with limited computational resources.",
    #         "shuffled": "cat mat the on the sat",
    #         "prediction": "The cat sat on the mat."
    #     }
    # ]
    # print("======"*30)
    # input("Press enter to continue...")


    # results = compute_metrics_for_batch(demo_data, use_ppl=True)
    # pretty_print(results)
    # pretty_print(compute_linguistic_metrics(demo_data[0]['original']))
    # pretty_print(compute_linguistic_metrics(demo_data[0]['shuffled']))

    # results = compute_metrics_for_pair(demo_data[0]['original'], demo_data[0]['prediction'])
    # pretty_print(results)





    # data=read_jsonl("data/raw/train.jsonl")
    # results=[]
    # for item in tqdm(data, desc="Evaluating", unit="item"):
    #     metrics=compute_linguistic_metrics(item['text'])
    #
    #     results.append({
    #         "text":item['text'],
    #         "label":item['label'],
    #         "metrics":metrics
    #     })
    # save_results(results,"src/evaluator/bisai_train_metrics.jsonl")
    # data=read_jsonl("data/classification_dataset/processed/train_data.jsonl")
    # results=[]
    # for item in tqdm(data,desc="Evaluating",unit="item"):
    #     metrics=compute_linguistic_metrics(item['abstract'])
    #     results.append({
    #         "abstract":item['abstract'],
    #         "source":item['source'],
    #         "metrics":metrics
    #     })
    # save_results(results,"src/evaluator/train_metrics.jsonl")