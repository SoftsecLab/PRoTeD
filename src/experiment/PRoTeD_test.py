# -*- coding: utf-8 -*-
"""
PRoTeD_test.py (STRICT train-consistent, including softmix inference)
+ auto load best_threshold.json
+ auto load best_temperature.json (temperature scaling)
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt

from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.utils.jsonl_handler import read_jsonl
from src.evaluator.metrics_evaluator import compute_linguistic_metrics
from src.preprocess.sentence_splitter import SentenceSegmenter
from src.experiment.classifier_modules import (
    RoBERTaWrapper, AttentionPooling, JointClassifier
)
from src.experiment.layered_perturbator import (
    load_or_init_model,
    parse_option,
    LAYER_PERTURB_OPTIONS,
    STRATEGY_FUNCTIONS,
    LEVEL_FEATURE_KEYS,
)

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_PERTURB_MODE = os.getenv("PROTED_PERTURB_MODE", "B")
DEFAULT_SOFTMIX_LAMBDA = float(os.getenv("PROTED_SOFTMIX_LAMBDA", "0.5"))
DEFAULT_GUMBEL_TAU = float(os.getenv("PROTED_GUMBEL_TAU", "1.0"))
DEFAULT_SOFTMIX_MAX_SENTENCES = int(os.getenv("PROTED_SOFTMIX_MAX_SENTENCES", "8"))
DEFAULT_T5_PREFIX = "reorder: "


class TextDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples = read_jsonl(jsonl_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        return x["text"], int(x["label"])


def load_best_threshold(model_dir: str, default_thr: float = 0.5) -> float:
    p = os.path.join(model_dir, "best_threshold.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return float(obj.get("threshold", default_thr))
        except Exception:
            return default_thr
    return default_thr


def load_best_temperature(model_dir: str, default_T: float = 1.0) -> float:
    p = os.path.join(model_dir, "best_temperature.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return float(obj.get("temperature", default_T))
        except Exception:
            return default_T
    return default_T


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _metric_keys_from_text(text: str) -> List[str]:
    md = compute_linguistic_metrics(text)
    return list(md.keys())


def _apply_temperature_np(probs: np.ndarray, T: float, eps: float = 1e-6) -> np.ndarray:
    if T is None or float(T) <= 0:
        T = 1.0
    p = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(p) - np.log(1.0 - p)
    cal = 1.0 / (1.0 + np.exp(-(logits / float(T))))
    return cal.astype(np.float32)


@torch.no_grad()
def encode_sent_list_with_t5_encoder(
    t5_model: T5ForConditionalGeneration,
    t5_tokenizer: T5Tokenizer,
    sent_list: List[str],
    device,
    max_length: int = 256,
    max_sentences: int = 8,
    prefix: str = DEFAULT_T5_PREFIX,
) -> torch.Tensor:
    if sent_list is None or len(sent_list) == 0:
        return torch.zeros((t5_model.config.d_model,), device=device)

    sents = list(sent_list)
    if len(sents) > max_sentences:
        sents = sents[:max_sentences]

    vecs = []
    for s in sents:
        enc = t5_tokenizer(
            prefix + s,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        enc_out = t5_model.encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        v = enc_out.last_hidden_state.mean(dim=1).squeeze(0)
        vecs.append(v)

    return torch.stack(vecs, dim=0).mean(dim=0)


def _gumbel_softmax_probs(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return F.gumbel_softmax(logits, tau=tau, hard=False)


@torch.no_grad()
def layered_perturb_train_consistent(
    text: str,
    segmenter: SentenceSegmenter,
    perturbator_models: Dict[str, torch.nn.Module],
    t5_model: T5ForConditionalGeneration,
    t5_tokenizer: T5Tokenizer,
    device,
    perturb_mode: str,
    softmix_lambda: float,
    gumbel_tau: float,
    softmix_max_sentences: int,
) -> Tuple[List[str], List[str], Optional[torch.Tensor]]:
    sentences = [s for s, _ in segmenter.segment(text)]
    sent_list = sentences.copy()
    o_shuffled = sentences.copy()

    enc_mix_layers: List[torch.Tensor] = []

    for layer in ["O", "S", "L"]:
        curr_text = " ".join(sent_list)
        metrics_now = compute_linguistic_metrics(curr_text)

        keys = LEVEL_FEATURE_KEYS[layer]
        pmodel = perturbator_models[layer]
        pmodel.eval()

        feat = torch.tensor([metrics_now.get(k, 0.0) for k in keys], dtype=torch.float).unsqueeze(0).to(device)
        logits = pmodel(feat)

        if perturb_mode.upper() == "B" and softmix_lambda > 0.0:
            probs_soft = _gumbel_softmax_probs(logits, tau=gumbel_tau)[0]

            cand_vecs = []
            cand_sent_lists = []
            for opt in LAYER_PERTURB_OPTIONS[layer]:
                strat, strength = parse_option(opt)
                cand_sents = STRATEGY_FUNCTIONS[strat](sent_list, strength)
                cand_sent_lists.append(cand_sents)

                v = encode_sent_list_with_t5_encoder(
                    t5_model=t5_model,
                    t5_tokenizer=t5_tokenizer,
                    sent_list=cand_sents,
                    device=device,
                    max_length=256,
                    max_sentences=softmix_max_sentences,
                    prefix=DEFAULT_T5_PREFIX,
                )
                cand_vecs.append(v)

            enc_stack = torch.stack(cand_vecs, dim=0)
            v_mix = (probs_soft.unsqueeze(-1) * enc_stack).sum(dim=0)
            enc_mix_layers.append(v_mix)

            hard_idx = int(torch.argmax(logits, dim=-1).item())
            sent_list = cand_sent_lists[hard_idx]
        else:
            hard_idx = int(torch.argmax(logits, dim=-1).item())
            opt = LAYER_PERTURB_OPTIONS[layer][hard_idx]
            strat, strength = parse_option(opt)
            sent_list = STRATEGY_FUNCTIONS[strat](sent_list, strength)

        if layer == "O":
            o_shuffled = sent_list.copy()

    disturbed_vec_soft = None
    if perturb_mode.upper() == "B" and softmix_lambda > 0.0 and len(enc_mix_layers) > 0:
        disturbed_vec_soft = torch.stack(enc_mix_layers, dim=0).mean(dim=0)

    return sent_list, o_shuffled, disturbed_vec_soft


@torch.no_grad()
def infer_one(
    text: str,
    roberta_model,
    t5_model,
    t5_tokenizer,
    reorder_pool,
    classifier,
    perturbator_models,
    device,
    metric_keys: List[str],
    perturb_mode: str,
    softmix_lambda: float,
    gumbel_tau: float,
    softmix_max_sentences: int,
    max_length: int = 512,
) -> float:
    segmenter = SentenceSegmenter()

    orig_vec = roberta_model([text])
    m0 = compute_linguistic_metrics(text)
    orig_metrics = torch.tensor([[m0.get(k, 0.0) for k in metric_keys]], dtype=torch.float, device=device)

    sent_list, o_shuffled, disturbed_vec_soft = layered_perturb_train_consistent(
        text=text,
        segmenter=segmenter,
        perturbator_models=perturbator_models,
        t5_model=t5_model,
        t5_tokenizer=t5_tokenizer,
        device=device,
        perturb_mode=perturb_mode,
        softmix_lambda=softmix_lambda,
        gumbel_tau=gumbel_tau,
        softmix_max_sentences=softmix_max_sentences,
    )

    disturbed_sent_vecs = []
    reordered_sent_vecs = []
    reordered_sents = []

    for shuffled_sent, target_sent in zip(sent_list, o_shuffled):
        inp = t5_tokenizer(
            f"{DEFAULT_T5_PREFIX}{shuffled_sent}",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        tgt = t5_tokenizer(
            target_sent,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        tgt.input_ids[tgt.input_ids == t5_tokenizer.pad_token_id] = -100

        out = t5_model(
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
            labels=tgt.input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        enc_vec = out.encoder_last_hidden_state.mean(dim=1).squeeze(0)
        dec_vec = out.decoder_hidden_states[-1].mean(dim=1).squeeze(0)

        disturbed_sent_vecs.append(enc_vec)
        reordered_sent_vecs.append(dec_vec)

        decoded = t5_tokenizer.decode(out.logits.argmax(dim=-1)[0], skip_special_tokens=True)
        reordered_sents.append(decoded)

    disturbed_vec_hard = reorder_pool(torch.stack(disturbed_sent_vecs).unsqueeze(0)).squeeze(0)
    reordered_vec = reorder_pool(torch.stack(reordered_sent_vecs).unsqueeze(0)).squeeze(0)

    if perturb_mode.upper() == "B" and softmix_lambda > 0.0 and disturbed_vec_soft is not None:
        disturbed_vec = (1.0 - softmix_lambda) * disturbed_vec_hard + softmix_lambda * disturbed_vec_soft
    else:
        disturbed_vec = disturbed_vec_hard

    m_d = compute_linguistic_metrics(" ".join(sent_list))
    m_r = compute_linguistic_metrics(" ".join(reordered_sents))

    disturbed_metrics = torch.tensor([[m_d.get(k, 0.0) for k in metric_keys]], dtype=torch.float, device=device)
    reordered_metrics = torch.tensor([[m_r.get(k, 0.0) for k in metric_keys]], dtype=torch.float, device=device)

    logits = classifier(
        orig_vec, orig_metrics,
        disturbed_vec.unsqueeze(0), disturbed_metrics,
        reordered_vec.unsqueeze(0), reordered_metrics
    )
    prob = torch.sigmoid(logits).view(-1)[0].item()
    return float(prob)


def evaluate_model(
    roberta_model,
    t5_model,
    t5_tokenizer,
    reorder_pool,
    classifier,
    perturbator_models,
    eval_path: str,
    device=default_device
):
    roberta_model.eval()
    t5_model.eval()
    reorder_pool.eval()
    classifier.eval()
    for m in perturbator_models.values():
        m.eval()

    dataset = TextDataset(eval_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_labels, all_probs = [], []

    perturb_mode = DEFAULT_PERTURB_MODE
    softmix_lambda = DEFAULT_SOFTMIX_LAMBDA
    gumbel_tau = DEFAULT_GUMBEL_TAU
    softmix_max_sentences = DEFAULT_SOFTMIX_MAX_SENTENCES

    first_text, _ = dataset[0]
    metric_keys = _metric_keys_from_text(first_text)

    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            text = texts[0]
            y = int(labels.item()) if torch.is_tensor(labels) else int(labels)
            p = infer_one(
                text=text,
                roberta_model=roberta_model,
                t5_model=t5_model,
                t5_tokenizer=t5_tokenizer,
                reorder_pool=reorder_pool,
                classifier=classifier,
                perturbator_models=perturbator_models,
                device=device,
                metric_keys=metric_keys,
                perturb_mode=perturb_mode,
                softmix_lambda=softmix_lambda,
                gumbel_tau=gumbel_tau,
                softmix_max_sentences=softmix_max_sentences,
            )
            all_labels.append(y)
            all_probs.append(p)

    labels_np = np.array(all_labels, dtype=np.int32)
    probs_np = np.array(all_probs, dtype=np.float32)
    preds_np = (probs_np > 0.5).astype(np.int32)
    return labels_np, preds_np, probs_np


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    preds = (probs > threshold).astype(np.int32)
    auc_roc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    acc = float(accuracy_score(labels, preds))
    pre = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    conf_matrix = confusion_matrix(labels, preds).tolist()

    fpr, tpr, _ = roc_curve(labels, probs)
    return auc_roc, {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "auc": auc_roc,
        "threshold": float(threshold),
        "conf_matrix": conf_matrix,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def plot_roc_curve(fpr, tpr, auc_roc, save_path: str, extra_text: str = ""):
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc_roc:.6f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (STRICT train-consistent)")
    if extra_text:
        plt.text(0.55, 0.15, extra_text, fontsize=10, bbox=dict(facecolor="white", alpha=0.7))
    plt.legend(loc="lower right")
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()


def load_model(model_dir: str, t5_model_path: str, roberta_model_path: str, device=default_device):
    roberta = RoBERTaWrapper(model_name=roberta_model_path).to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path).to(device)
    reorder_pool = AttentionPooling(hidden_dim=768).to(device)
    classifier = JointClassifier(input_dim_each=768 + 17).to(device)

    roberta.load_state_dict(torch.load(os.path.join(model_dir, "roberta.pt"), map_location=device, weights_only=False), strict=False)
    t5_model.load_state_dict(torch.load(os.path.join(model_dir, "t5.pt"), map_location=device, weights_only=False), strict=False)
    reorder_pool.load_state_dict(torch.load(os.path.join(model_dir, "reorder_pool.pt"), map_location=device, weights_only=False), strict=False)
    classifier.load_state_dict(torch.load(os.path.join(model_dir, "classifier.pt"), map_location=device, weights_only=False), strict=False)

    perturbator_models = {}
    for layer in LAYER_PERTURB_OPTIONS:
        m = load_or_init_model(layer).to(device)
        perturbator_models[layer] = m
        p = os.path.join(model_dir, f"perturbator_{layer}.pt")
        if os.path.exists(p):
            perturbator_models[layer].load_state_dict(torch.load(p, map_location=device, weights_only=False), strict=False)

    return roberta, t5_model, t5_tokenizer, reorder_pool, classifier, perturbator_models


def eval_jsonl_strict(
    model_dir: str,
    eval_path: str,
    out_dir: str,
    t5_model_path: str,
    roberta_model_path: str,
    threshold: Optional[float] = None,
    device=default_device
):
    if threshold is None:
        threshold = load_best_threshold(model_dir, default_thr=0.5)
    temperature = load_best_temperature(model_dir, default_T=1.0)

    roberta, t5_model, t5_tokenizer, reorder_pool, classifier, perturbator_models = load_model(
        model_dir=model_dir,
        t5_model_path=t5_model_path,
        roberta_model_path=roberta_model_path,
        device=device
    )

    labels, preds, probs = evaluate_model(
        roberta_model=roberta,
        t5_model=t5_model,
        t5_tokenizer=t5_tokenizer,
        reorder_pool=reorder_pool,
        classifier=classifier,
        perturbator_models=perturbator_models,
        eval_path=eval_path,
        device=device
    )

    probs_cal = _apply_temperature_np(probs, temperature)
    auc_roc, metrics = compute_metrics(labels, probs_cal, threshold=threshold)
    metrics["temperature"] = float(temperature)

    _ensure_dir(out_dir)
    with open(os.path.join(out_dir, "metrics_strict.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    extra = (
        f"Acc:{metrics['accuracy']:.4f}\nF1:{metrics['f1']:.4f}\n"
        f"Prec:{metrics['precision']:.4f}\nRec:{metrics['recall']:.4f}\n"
        f"Thr:{metrics['threshold']:.4f}\nT:{metrics['temperature']:.4f}"
    )
    plot_roc_curve(
        fpr=np.array(metrics["fpr"]),
        tpr=np.array(metrics["tpr"]),
        auc_roc=auc_roc,
        save_path=os.path.join(out_dir, "roc_strict.png"),
        extra_text=extra
    )

    pred_path = os.path.join(out_dir, "predictions_strict.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for y, p in zip(labels.tolist(), probs_cal.tolist()):
            pred = int(1 if float(p) >= float(threshold) else 0)
            f.write(json.dumps({"label": int(y), "prob": float(p), "pred": int(pred)}) + "\n")

    print(f"[OK] strict metrics saved to: {os.path.join(out_dir, 'metrics_strict.json')}")
    print(f"[OK] strict roc saved to: {os.path.join(out_dir, 'roc_strict.png')}")
    print(f"[OK] strict predictions saved to: {pred_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="eval", choices=["eval"])
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="saved_models/PRoTeD/test_outputs_strict")
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument("--t5_model_path", type=str, default="models/T5_base/t5-base_group_6")
    parser.add_argument("--roberta_model_path", type=str, default="models/roberta-base")

    parser.add_argument("--perturb_mode", type=str, default=DEFAULT_PERTURB_MODE, choices=["A", "B"])
    parser.add_argument("--softmix_lambda", type=float, default=DEFAULT_SOFTMIX_LAMBDA)
    parser.add_argument("--gumbel_tau", type=float, default=DEFAULT_GUMBEL_TAU)
    parser.add_argument("--softmix_max_sentences", type=int, default=DEFAULT_SOFTMIX_MAX_SENTENCES)

    args = parser.parse_args()

    DEFAULT_PERTURB_MODE = args.perturb_mode
    DEFAULT_SOFTMIX_LAMBDA = float(args.softmix_lambda)
    DEFAULT_GUMBEL_TAU = float(args.gumbel_tau)
    DEFAULT_SOFTMIX_MAX_SENTENCES = int(args.softmix_max_sentences)

    eval_jsonl_strict(
        model_dir=args.model_dir,
        eval_path=args.eval_path,
        out_dir=args.out_dir,
        t5_model_path=args.t5_model_path,
        roberta_model_path=args.roberta_model_path,
        threshold=args.threshold
    )
