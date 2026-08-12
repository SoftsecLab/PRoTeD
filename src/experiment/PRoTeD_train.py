# ========== PRoTeD_train.py (with train/val split + threshold selection + early stopping + resume checkpoint
#                             + temperature scaling + save labels_preds_probs_epochX.json + plot test ROC) ==========

import os
import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt

from src.utils.jsonl_handler import read_jsonl
from src.evaluator.metrics_evaluator import compute_linguistic_metrics
from src.experiment.classifier_modules import RoBERTaWrapper, AttentionPooling, JointClassifier
from src.preprocess.sentence_splitter import SentenceSegmenter
from src.experiment.layered_perturbator import (
    load_or_init_model,
    parse_option,
    LAYER_PERTURB_OPTIONS,
    STRATEGY_FUNCTIONS,
    LEVEL_FEATURE_KEYS,
)


from src.experiment.PRoTeD_test import evaluate_model


# -----------------------
# utils: write jsonl
# -----------------------
def write_jsonl(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def write_json_list(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# -----------------------
# dataset
# -----------------------
class TextDataset(Dataset):
    """Load jsonl with {"text":..., "label":...}"""
    def __init__(self, jsonl_path: str):
        self.samples = read_jsonl(jsonl_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["text"], self.samples[idx]["label"]


# -----------------------
# balanced dataloader (same as your old logic)
# -----------------------
def get_balanced_dataloader(dataset, batch_size: int, seed: int = 2025):

    rng = random.Random(seed)
    label_to_indices = defaultdict(list)

    for idx in range(len(dataset)):
        _, y = dataset[idx]
        label_to_indices[int(y)].append(idx)

    if 0 not in label_to_indices or 1 not in label_to_indices:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for y in label_to_indices:
        rng.shuffle(label_to_indices[y])

    half = batch_size // 2
    num_batches = min(len(label_to_indices[0]), len(label_to_indices[1])) // half

    all_indices = []
    for i in range(num_batches):
        pos = label_to_indices[1][i * half:(i + 1) * half]
        neg = label_to_indices[0][i * half:(i + 1) * half]
        batch = pos + neg
        rng.shuffle(batch)
        all_indices.extend(batch)

    subset = Subset(dataset, all_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


# -----------------------
# stratified split (train/val)
# -----------------------
def stratified_split_indices(labels: List[int], val_ratio: float, seed: int):
    rng = random.Random(seed)
    idx0 = [i for i, y in enumerate(labels) if int(y) == 0]
    idx1 = [i for i, y in enumerate(labels) if int(y) == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_val = int(round(len(idx0) * val_ratio))
    n1_val = int(round(len(idx1) * val_ratio))

    val_idx = idx0[:n0_val] + idx1[:n1_val]
    train_idx = idx0[n0_val:] + idx1[n1_val:]

    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx, val_idx


# -----------------------
# threshold selection on val (max F1)
# -----------------------
def find_best_threshold_by_f1(labels, probs) -> Tuple[float, float]:
    labels = [int(x) for x in labels]
    probs = [float(x) for x in probs]
    prec, rec, thrs = precision_recall_curve(labels, probs)

    best_thr = 0.5
    best_f1 = -1.0
    for i, thr in enumerate(thrs):
        p = float(prec[i + 1])
        r = float(rec[i + 1])
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


def compute_metrics_at_threshold(labels, probs, thr: float) -> Dict[str, Any]:
    labels = [int(x) for x in labels]
    probs = [float(x) for x in probs]
    preds = [1 if p >= thr else 0 for p in probs]

    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()

    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(pre),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(thr),
        "conf_matrix": cm,
    }


# -----------------------
# Temperature scaling (minimal change: infer logits from probs)
# -----------------------
def _probs_to_logits_torch(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.clamp(eps, 1.0 - eps)
    return torch.log(probs) - torch.log(1.0 - probs)


@torch.no_grad()
def apply_temperature_scaling_from_probs(probs, T: float) -> List[float]:
    """
    probs -> logits -> logits/T -> sigmoid
    (T>0, scalar)
    """
    if T is None or float(T) <= 0:
        T = 1.0
    p = torch.tensor(probs, dtype=torch.float32)
    logits = _probs_to_logits_torch(p)
    cal = torch.sigmoid(logits / float(T))
    return cal.cpu().numpy().astype(float).tolist()


def fit_temperature_from_probs(
    labels,
    probs,
    device,
    max_iter: int = 80,
    lr: float = 0.05,
) -> float:
    """
    Fit a single scalar Temperature T on VAL by minimizing NLL:
      loss = BCEWithLogits(logits/T, y)
    logits are recovered from probs to keep changes minimal.
    """
    y = torch.tensor([float(int(x)) for x in labels], dtype=torch.float32, device=device)
    p = torch.tensor([float(x) for x in probs], dtype=torch.float32, device=device)
    logits = _probs_to_logits_torch(p)

    # parameterize T = softplus(s) + 1e-6 to ensure >0
    s = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([s], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()

    best_T = 1.0
    best_loss = float("inf")

    for _ in range(max_iter):
        opt.zero_grad()
        T = F.softplus(s) + 1e-6
        loss = bce(logits / T, y)
        loss.backward()
        opt.step()

        lv = float(loss.item())
        if lv < best_loss:
            best_loss = lv
            best_T = float((F.softplus(s).detach() + 1e-6).item())

    # safety clamp
    if not (best_T > 0 and best_T < 1000):
        best_T = 1.0
    return float(best_T)


def plot_roc_and_save(labels, probs, save_path: str, title: str = "ROC Curve"):
    labels = [int(x) for x in labels]
    probs = [float(x) for x in probs]
    if len(set(labels)) <= 1:
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.6f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


# -----------------------
# gumbel helper
# -----------------------
def gumbel_softmax_sample(logits, tau=1.0):
    return F.gumbel_softmax(logits, tau=tau, hard=True)


# -----------------------
# optional: encode candidate sent list with T5 encoder (for softmix)
# -----------------------
def encode_sent_list_with_t5_encoder(
    t5_model: T5ForConditionalGeneration,
    t5_tokenizer: T5Tokenizer,
    sent_list: List[str],
    device,
    max_length: int = 256,
    max_sentences: int = 8,
    prefix: str = "reorder: ",
    use_grad: bool = False,
):
    if sent_list is None:
        return torch.zeros((t5_model.config.d_model,), device=device)

    sents = list(sent_list)
    if len(sents) == 0:
        return torch.zeros((t5_model.config.d_model,), device=device)
    if len(sents) > max_sentences:
        sents = sents[:max_sentences]

    ctx = torch.enable_grad() if use_grad else torch.no_grad()
    with ctx:
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


# -----------------------
# train one epoch
# -----------------------
def train_one_epoch(
    dataloader,
    roberta_model,
    t5_model,
    classifier,
    t5_tokenizer,
    reorder_pool,
    segmenter,
    device,
    optimizer,
    criterion,
    perturbator_models,
    alpha=1.0,
    beta=0.0,
    perturb_mode: str = "B",
    softmix_lambda: float = 0.5,
    gumbel_tau: float = 1.0,
    softmix_detach_enc: bool = True,
    softmix_max_sentences: int = 8,
):
    roberta_model.train()
    t5_model.train()
    reorder_pool.train()
    classifier.train()
    for m in perturbator_models.values():
        m.train()

    total_loss = 0.0

    for texts, labels in tqdm(dataloader, desc="Train", leave=False):
        labels = labels.to(device)

        # original
        original_vec = roberta_model(texts)
        metric_dicts = [compute_linguistic_metrics(text) for text in texts]
        original_metrics = torch.tensor(
            [[m[k] for k in metric_dicts[0].keys()] for m in metric_dicts],
            dtype=torch.float,
            device=device
        )

        disturbed_vecs, reordered_vecs = [], []
        disturbed_metrics_list, reordered_metrics_list = [], []
        t5_total_loss = 0.0
        t5_loss_count = 0

        for text in texts:
            sentences = [s for s, _ in segmenter.segment(text)]
            sent_list = sentences.copy()
            o_shuffled = sentences.copy()

            enc_mix_layers = []

            for layer in ["O", "S", "L"]:
                curr_text = " ".join(sent_list)
                metrics_now = compute_linguistic_metrics(curr_text)

                keys = LEVEL_FEATURE_KEYS[layer]
                pmodel = perturbator_models[layer].to(device)

                feat = torch.tensor([metrics_now.get(k, 0.0) for k in keys], dtype=torch.float).unsqueeze(0).to(device)
                logits = pmodel(feat)

                # if perturb_mode.upper() == "B":
                probs_soft = F.gumbel_softmax(logits, tau=gumbel_tau, hard=False)[0]

                cand_sent_lists = []
                cand_vecs = []
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
                        prefix="reorder: ",
                        use_grad=not softmix_detach_enc,
                    )
                    if softmix_detach_enc:
                        v = v.detach()
                    cand_vecs.append(v)

                enc_stack = torch.stack(cand_vecs, dim=0)
                v_mix = (probs_soft.unsqueeze(-1) * enc_stack).sum(dim=0)
                enc_mix_layers.append(v_mix)

                hard_idx = int(torch.argmax(logits, dim=-1).item())
                sent_list = cand_sent_lists[hard_idx]
                # else:
                #     probs_hard = gumbel_softmax_sample(logits, tau=gumbel_tau)
                #     hard_idx = int(torch.argmax(probs_hard).item())
                #     opt = LAYER_PERTURB_OPTIONS[layer][hard_idx]
                #     strat, strength = parse_option(opt)
                #     sent_list = STRATEGY_FUNCTIONS[strat](sent_list, strength)

                if layer == "O":
                    o_shuffled = sent_list.copy()

            disturbed_sent_vecs, reordered_sent_vecs, reordered_sents = [], [], []
            for shuffled_sent, target_sent in zip(sent_list, o_shuffled):
                input_enc = t5_tokenizer(
                    f"reorder: {shuffled_sent}",
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                target_enc = t5_tokenizer(
                    target_sent,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                target_enc.input_ids[target_enc.input_ids == t5_tokenizer.pad_token_id] = -100

                output = t5_model(
                    input_ids=input_enc.input_ids,
                    attention_mask=input_enc.attention_mask,
                    labels=target_enc.input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )

                encoder_vec = output.encoder_last_hidden_state.mean(dim=1).squeeze(0)
                decoder_vec = output.decoder_hidden_states[-1].mean(dim=1).squeeze(0)
                disturbed_sent_vecs.append(encoder_vec)
                reordered_sent_vecs.append(decoder_vec)

                decoded = t5_tokenizer.decode(output.logits.argmax(dim=-1)[0], skip_special_tokens=True)
                reordered_sents.append(decoded)

                t5_total_loss += output.loss
                t5_loss_count += 1

            disturbed_vec_hard = reorder_pool(torch.stack(disturbed_sent_vecs).unsqueeze(0)).squeeze(0)
            reordered_vec = reorder_pool(torch.stack(reordered_sent_vecs).unsqueeze(0)).squeeze(0)

            # if perturb_mode.upper() == "B" and len(enc_mix_layers) > 0:
            #     disturbed_vec_soft = torch.stack(enc_mix_layers, dim=0).mean(dim=0)
            #     disturbed_vec = (1.0 - softmix_lambda) * disturbed_vec_hard + softmix_lambda * disturbed_vec_soft
            # else:
            #     disturbed_vec = disturbed_vec_hard

            if len(enc_mix_layers) > 0:
                disturbed_vec_soft = torch.stack(enc_mix_layers, dim=0).mean(dim=0)
                disturbed_vec = (1.0 - softmix_lambda) * disturbed_vec_hard + softmix_lambda * disturbed_vec_soft

            disturbed_vecs.append(disturbed_vec)
            reordered_vecs.append(reordered_vec)

            disturbed_metrics = compute_linguistic_metrics(" ".join(sent_list))
            reordered_metrics = compute_linguistic_metrics(" ".join(reordered_sents))

            disturbed_metrics_list.append([disturbed_metrics[k] for k in disturbed_metrics])
            reordered_metrics_list.append([reordered_metrics[k] for k in reordered_metrics])

        disturbed_vecs = torch.stack(disturbed_vecs)
        reordered_vecs = torch.stack(reordered_vecs)
        disturbed_metrics = torch.tensor(disturbed_metrics_list, dtype=torch.float, device=device)
        reordered_metrics = torch.tensor(reordered_metrics_list, dtype=torch.float, device=device)

        logits = classifier(
            original_vec, original_metrics,
            disturbed_vecs, disturbed_metrics,
            reordered_vecs, reordered_metrics
        )

        cls_loss = criterion(logits, labels.float())
        t5_loss_value = (t5_total_loss / max(t5_loss_count, 1))

        perturbator_loss = torch.tensor(0.0, device=device)
        if beta > 0.0:
            for layer in ["O", "S", "L"]:
                pmodel = perturbator_models[layer]
                keys = LEVEL_FEATURE_KEYS[layer]
                batch_feats = torch.tensor(
                    [[m.get(k, 0.0) for k in keys] for m in metric_dicts],
                    dtype=torch.float,
                    device=device
                )
                logits_p = pmodel(batch_feats)
                probs_ = F.softmax(logits_p, dim=-1)
                entropy = -(probs_ * torch.log(probs_ + 1e-8)).sum(dim=-1).mean()
                perturbator_loss = perturbator_loss + (-entropy)

        total_loss_batch = cls_loss + alpha * t5_loss_value + beta * perturbator_loss

        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()

        total_loss += float(total_loss_batch.item())

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"📉 Avg loss: {avg_loss:.6f}")
    return avg_loss


# -----------------------
# early stopping
# -----------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = str(mode)
        self.best: Optional[float] = None
        self.bad_epochs: int = 0

    def step(self, value: float) -> Tuple[bool, bool]:
        if self.best is None:
            self.best = float(value)
            self.bad_epochs = 0
            return True, False

        improved = False
        if self.mode == "max":
            improved = value > (self.best + self.min_delta)
        else:
            improved = value < (self.best - self.min_delta)

        if improved:
            self.best = float(value)
            self.bad_epochs = 0
            return True, False

        self.bad_epochs += 1
        return False, (self.bad_epochs >= self.patience)


# -----------------------
# save best model
# -----------------------
def save_best_model(best_model_dir, original_encoder, t5_model, reorder_pool, classifier, perturbator_models):
    os.makedirs(best_model_dir, exist_ok=True)
    torch.save(original_encoder.state_dict(), os.path.join(best_model_dir, "roberta.pt"))
    torch.save(t5_model.state_dict(), os.path.join(best_model_dir, "t5.pt"))
    torch.save(reorder_pool.state_dict(), os.path.join(best_model_dir, "reorder_pool.pt"))
    torch.save(classifier.state_dict(), os.path.join(best_model_dir, "classifier.pt"))
    for layer, model in perturbator_models.items():
        torch.save(model.state_dict(), os.path.join(best_model_dir, f"perturbator_{layer}.pt"))


# -----------------------
# checkpoint (full resume)
# -----------------------
def save_checkpoint(
    path: str,
    epoch: int,
    best_auc: float,
    best_threshold: float,
    best_temperature: float,
    early_stopper: EarlyStopping,
    optimizer: torch.optim.Optimizer,
    original_encoder,
    t5_model,
    reorder_pool,
    classifier,
    perturbator_models,
):
    ckpt = {
        "epoch": int(epoch),
        "best_auc": float(best_auc),
        "best_threshold": float(best_threshold),
        "best_temperature": float(best_temperature),
        "early_best": None if early_stopper.best is None else float(early_stopper.best),
        "early_bad_epochs": int(early_stopper.bad_epochs),
        "optimizer": optimizer.state_dict(),
        "models": {
            "roberta": original_encoder.state_dict(),
            "t5": t5_model.state_dict(),
            "reorder_pool": reorder_pool.state_dict(),
            "classifier": classifier.state_dict(),
            "perturbators": {k: v.state_dict() for k, v in perturbator_models.items()},
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint_if_exists(
    path: str,
    device,
    optimizer: torch.optim.Optimizer,
    early_stopper: EarlyStopping,
    original_encoder,
    t5_model,
    reorder_pool,
    classifier,
    perturbator_models,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location=device, weights_only=False)

    original_encoder.load_state_dict(ckpt["models"]["roberta"], strict=False)
    t5_model.load_state_dict(ckpt["models"]["t5"], strict=False)
    reorder_pool.load_state_dict(ckpt["models"]["reorder_pool"], strict=False)
    classifier.load_state_dict(ckpt["models"]["classifier"], strict=False)

    for k, sd in ckpt["models"]["perturbators"].items():
        if k in perturbator_models:
            perturbator_models[k].load_state_dict(sd, strict=False)

    optimizer.load_state_dict(ckpt["optimizer"])

    early_stopper.best = ckpt.get("early_best", None)
    early_stopper.bad_epochs = ckpt.get("early_bad_epochs", 0)

    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "best_auc": float(ckpt.get("best_auc", 0.0)),
        "best_threshold": float(ckpt.get("best_threshold", 0.5)),
        "best_temperature": float(ckpt.get("best_temperature", 1.0)),
    }


# -----------------------
# main
# -----------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init modules
    t5_tokenizer = T5Tokenizer.from_pretrained(args.t5_model_path)
    original_encoder = RoBERTaWrapper(model_name=args.roberta_model_path).to(device)
    t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model_path).to(device)
    reorder_pool = AttentionPooling(hidden_dim=768).to(device)
    classifier = JointClassifier(input_dim_each=768 + 17, hidden_dim=256, fusion_dim=512).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # perturbators
    perturbator_models = {}
    for layer in LAYER_PERTURB_OPTIONS:
        model = load_or_init_model(layer).to(device)
        perturbator_models[layer] = model

    os.makedirs(args.save_dir, exist_ok=True)
    best_model_dir = os.path.join(args.save_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    # split train/val (deterministic) and dump val.jsonl
    full_dataset = TextDataset(args.data_path)
    labels = [int(full_dataset.samples[i]["label"]) for i in range(len(full_dataset.samples))]
    train_idx, val_idx = stratified_split_indices(labels, val_ratio=args.val_ratio, seed=args.split_seed)

    splits_dir = os.path.join(args.save_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    val_jsonl = os.path.join(splits_dir, "val.jsonl")
    split_meta = os.path.join(splits_dir, "split_meta.json")

    val_items = [full_dataset.samples[i] for i in val_idx]
    write_jsonl(val_jsonl, val_items)
    with open(split_meta, "w", encoding="utf-8") as f:
        json.dump({
            "data_path": args.data_path,
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
        }, f, indent=2)

    print(f"[Split] train={len(train_idx)} val={len(val_idx)} val_ratio={args.val_ratio}")
    print(f"[Split] val_jsonl: {val_jsonl}")

    # train loader (balanced)
    train_subset = Subset(full_dataset, train_idx)
    train_loader = get_balanced_dataloader(train_subset, batch_size=args.batch_size, seed=args.train_seed)

    segmenter = SentenceSegmenter()

    # optimizer
    optimizer_params = []
    optimizer_params += list(original_encoder.parameters())
    optimizer_params += list(t5_model.parameters())
    optimizer_params += list(reorder_pool.parameters())
    optimizer_params += list(classifier.parameters())
    for m in perturbator_models.values():
        optimizer_params += list(m.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)

    # early stopping
    early_stopper = EarlyStopping(patience=args.early_patience, min_delta=args.early_min_delta, mode="max")

    # best record files
    metrics_path = os.path.join(best_model_dir, "eval_metrics_val.json")
    thr_path = os.path.join(best_model_dir, "best_threshold.json")
    temp_path = os.path.join(best_model_dir, "best_temperature.json")

    best_auc = 0.0
    best_threshold = 0.5
    best_temperature = 1.0

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        best_auc = float(obj.get("auc", 0.0))
        print(f"[Init] found historical best val auc={best_auc:.6f}")

    if os.path.exists(thr_path):
        with open(thr_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        best_threshold = float(obj.get("threshold", 0.5))

    if os.path.exists(temp_path):
        with open(temp_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        best_temperature = float(obj.get("temperature", 1.0))

    # resume checkpoint
    ckpt_path = os.path.join(args.save_dir, "checkpoint_last.pt")
    resume = load_checkpoint_if_exists(
        ckpt_path,
        device=device,
        optimizer=optimizer,
        early_stopper=early_stopper,
        original_encoder=original_encoder,
        t5_model=t5_model,
        reorder_pool=reorder_pool,
        classifier=classifier,
        perturbator_models=perturbator_models,
    )

    start_epoch = 0
    if resume is not None:
        start_epoch = int(resume["epoch"]) + 1
        best_auc = float(resume["best_auc"])
        best_threshold = float(resume["best_threshold"])
        best_temperature = float(resume.get("best_temperature", 1.0))
        print(f"[Resume] start_epoch={start_epoch}, best_auc={best_auc:.6f}, best_thr={best_threshold:.6f}, best_T={best_temperature:.6f}")

    # train loop
    offset = int(args.offset)

    for epoch in range(start_epoch, args.epochs):
        epoch_id = epoch + offset
        print(f"\n=== Epoch {epoch_id}/{args.epochs + offset - 1} ===")

        alpha = 1.0
        beta = 0.01 if epoch < (args.epochs // 2 + 1) else 0.0

        train_one_epoch(
            dataloader=train_loader,
            roberta_model=original_encoder,
            t5_model=t5_model,
            classifier=classifier,
            t5_tokenizer=t5_tokenizer,
            reorder_pool=reorder_pool,
            segmenter=segmenter,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            perturbator_models=perturbator_models,
            alpha=alpha,
            beta=beta,
            perturb_mode=args.perturb_mode,
            softmix_lambda=args.softmix_lambda,
            gumbel_tau=args.gumbel_tau,
            softmix_detach_enc=(not args.softmix_t5_grad),
            softmix_max_sentences=args.softmix_max_sentences,
        )

        # ---- VAL evaluate (raw probs) ----
        val_labels, _, val_probs_raw = evaluate_model(
            roberta_model=original_encoder,
            t5_model=t5_model,
            classifier=classifier,
            reorder_pool=reorder_pool,
            t5_tokenizer=t5_tokenizer,
            eval_path=val_jsonl,
            perturbator_models=perturbator_models,
            device=device
        )

        # AUC unaffected by temperature scaling (monotonic), still compute on raw
        val_auc = float(roc_auc_score(val_labels, val_probs_raw)) if len(set(val_labels.tolist())) > 1 else 0.0

        # ---- Temperature scaling on VAL (fit T using raw probs -> logits) ----
        T_epoch = fit_temperature_from_probs(val_labels, val_probs_raw, device=device, max_iter=args.temp_max_iter, lr=args.temp_lr)
        val_probs = apply_temperature_scaling_from_probs(val_probs_raw.tolist(), T_epoch)

        # threshold selection on calibrated probs
        thr_star, best_f1_from_search = find_best_threshold_by_f1(val_labels, val_probs)
        val_metrics = compute_metrics_at_threshold(val_labels, val_probs, thr_star)
        val_metrics.update({
            "epoch": int(epoch_id),
            "best_f1_from_search": float(best_f1_from_search),
            "val_auc_raw": float(val_auc),
            "temperature_T": float(T_epoch),
        })

        with open(os.path.join(args.save_dir, f"metrics_val_epoch{epoch_id}.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)

        print(f"[VAL] epoch={epoch_id} auc(raw)={val_auc:.6f} T={T_epoch:.6f} thr*={thr_star:.6f} f1@thr*={val_metrics['f1']:.6f}")

        # ---- Per-epoch TEST dump like old version: labels_preds_probs_epoch{epoch_id}.json ----
        if args.save_epoch_test_dump:
            test_labels_ep, _, test_probs_raw_ep = evaluate_model(
                roberta_model=original_encoder,
                t5_model=t5_model,
                classifier=classifier,
                reorder_pool=reorder_pool,
                t5_tokenizer=t5_tokenizer,
                eval_path=args.eval_path,
                perturbator_models=perturbator_models,
                device=device
            )
            test_probs_ep = apply_temperature_scaling_from_probs(test_probs_raw_ep.tolist(), T_epoch)

            dump_thr = float(best_threshold) if best_threshold is not None else float(thr_star)
            dump_items = []
            for y, p in zip(test_labels_ep.tolist(), test_probs_ep):
                dump_items.append({
                    "label": int(y),
                    "pred": int(1 if float(p) >= dump_thr else 0),
                    "prob": float(p)
                })
            write_json_list(os.path.join(args.save_dir, f"labels_preds_probs_epoch{epoch_id}.json"), dump_items)

        # early stopping on val_auc (raw or calibrated is same for AUC)
        improved, should_stop = early_stopper.step(val_auc)

        # save best_model if improved (by val_auc)
        if improved and (val_auc >= best_auc + args.early_min_delta):
            best_auc = val_auc
            best_threshold = thr_star
            best_temperature = T_epoch

            print(f"🎯 Save best_model (epoch={epoch_id}) val_auc={best_auc:.6f} thr*={best_threshold:.6f} best_T={best_temperature:.6f}")

            save_best_model(best_model_dir, original_encoder, t5_model, reorder_pool, classifier, perturbator_models)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)
            with open(thr_path, "w", encoding="utf-8") as f:
                json.dump({"threshold": float(best_threshold), "epoch": int(epoch_id)}, f, indent=2)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({"temperature": float(best_temperature), "epoch": int(epoch_id)}, f, indent=2)

        # always save last checkpoint (resume)
        save_checkpoint(
            path=ckpt_path,
            epoch=epoch,
            best_auc=best_auc,
            best_threshold=best_threshold,
            best_temperature=best_temperature,
            early_stopper=early_stopper,
            optimizer=optimizer,
            original_encoder=original_encoder,
            t5_model=t5_model,
            reorder_pool=reorder_pool,
            classifier=classifier,
            perturbator_models=perturbator_models,
        )

        if should_stop:
            print(f"⏹️ EarlyStopping: no val_auc improvement for {args.early_patience} epochs.")
            break

    # -----------------------
    # Final test with best_model + best_threshold + best_temperature
    # -----------------------
    # reload best weights (safety)
    if os.path.exists(os.path.join(best_model_dir, "classifier.pt")):
        print(f"\n[Final Test] load best_model from {best_model_dir}")
        original_encoder.load_state_dict(torch.load(os.path.join(best_model_dir, "roberta.pt"), weights_only=False), strict=False)
        t5_model.load_state_dict(torch.load(os.path.join(best_model_dir, "t5.pt"), weights_only=False), strict=False)
        reorder_pool.load_state_dict(torch.load(os.path.join(best_model_dir, "reorder_pool.pt"), weights_only=False), strict=False)
        classifier.load_state_dict(torch.load(os.path.join(best_model_dir, "classifier.pt"), weights_only=False), strict=False)
        for layer in LAYER_PERTURB_OPTIONS:
            p = os.path.join(best_model_dir, f"perturbator_{layer}.pt")
            if os.path.exists(p):
                perturbator_models[layer].load_state_dict(torch.load(p, weights_only=False), strict=False)

    # read best threshold + temperature
    final_thr = best_threshold
    if os.path.exists(thr_path):
        with open(thr_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        final_thr = float(obj.get("threshold", final_thr))

    final_T = best_temperature
    if os.path.exists(temp_path):
        with open(temp_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        final_T = float(obj.get("temperature", final_T))

    test_labels, _, test_probs_raw = evaluate_model(
        roberta_model=original_encoder,
        t5_model=t5_model,
        classifier=classifier,
        reorder_pool=reorder_pool,
        t5_tokenizer=t5_tokenizer,
        eval_path=args.eval_path,
        perturbator_models=perturbator_models,
        device=device
    )
    test_probs = apply_temperature_scaling_from_probs(test_probs_raw.tolist(), final_T)

    test_metrics = compute_metrics_at_threshold(test_labels, test_probs, final_thr)
    test_metrics["threshold_from_val"] = float(final_thr)
    test_metrics["temperature_from_val"] = float(final_T)

    with open(os.path.join(args.save_dir, "metrics_test_best.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # save final test labels/preds/probs file (your requested format)
    final_dump = []
    for y, p in zip(test_labels.tolist(), test_probs):
        final_dump.append({
            "label": int(y),
            "pred": int(1 if float(p) >= float(final_thr) else 0),
            "prob": float(p)
        })
    write_json_list(os.path.join(args.save_dir, "labels_preds_probs_test_best.json"), final_dump)

    # plot ROC/AUROC image for test
    plot_roc_and_save(
        labels=test_labels.tolist(),
        probs=test_probs,
        save_path=os.path.join(args.save_dir, "test_roc_best.png"),
        title="Test ROC Curve (best_model, temperature-scaled)"
    )

    print(f"[TEST] auc={test_metrics['auc']:.6f} thr(val)={final_thr:.6f} T(val)={final_T:.6f} f1={test_metrics['f1']:.6f}")
    print(f"[TEST] saved: labels_preds_probs_test_best.json, test_roc_best.png")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_dir", type=str, default="saved_models/PRoTeD")
    parser.add_argument("--offset", type=int, default=1)

    # model paths
    parser.add_argument("--t5_model_path", type=str, default="models/T5_base/t5-base_group_6")
    parser.add_argument("--roberta_model_path", type=str, default="models/roberta-base")

    # perturb
    parser.add_argument("--perturb_mode", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--softmix_lambda", type=float, default=0.5)
    parser.add_argument("--gumbel_tau", type=float, default=1.0)
    parser.add_argument("--softmix_max_sentences", type=int, default=8)
    parser.add_argument("--softmix_t5_grad", action="store_true")

    # train/val split + resume stability
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=2025)
    parser.add_argument("--train_seed", type=int, default=2025)

    # early stopping
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--early_min_delta", type=float, default=0.0)

    # temperature scaling
    parser.add_argument("--temp_max_iter", type=int, default=80)
    parser.add_argument("--temp_lr", type=float, default=0.05)

    # save epoch test dump (labels_preds_probs_epochX.json)
    parser.add_argument("--save_epoch_test_dump", action="store_true",
                        help="If set, evaluate on TEST each epoch and save labels_preds_probs_epochX.json")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
