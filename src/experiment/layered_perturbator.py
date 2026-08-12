# -*- coding: utf-8 -*-


from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

import torch
import torch.nn as nn


# ============================

# ============================


#   PPL, SentenceLengthVariability, AvgSentenceLength, SentenceComplexityIndex,
#   DependencyLength, SemanticCohesion,
#   LexicalDensity, LexicalDiversity, WordFrequencyEntropy, SyllableDensity,
#   EntropyDensity, CompressionDensity,
#   SyntacticDensity, FleschKincaidScore, GunningFogScore,
#   HierarchicalDensity, SyntacticErrors

LEVEL_FEATURE_KEYS: Dict[str, List[str]] = {

    "O": [
        "SentenceLengthVariability",
        "AvgSentenceLength",
        "SentenceComplexityIndex",
        "DependencyLength",
        "SemanticCohesion",
    ],

    "S": [
        "SyntacticDensity",
        "FleschKincaidScore",
        "GunningFogScore",
        "HierarchicalDensity",
        "SyntacticErrors",
    ],

    "L": [
        "LexicalDensity",
        "LexicalDiversity",
        "WordFrequencyEntropy",
        "SyllableDensity",
        "PPL",
    ],
}


# ============================

# ============================

def _rng(seed: int = 42) -> random.Random:
    r = random.Random(seed)
    return r


def noop(sents: List[str], strength: float, seed: int = 42) -> List[str]:

    return list(sents)


def global_shuffle(sents: List[str], strength: float, seed: int = 42) -> List[str]:

    if len(sents) <= 1 or strength <= 0:
        return list(sents)
    r = _rng(seed)
    sents = list(sents)
    n = len(sents)
    k = max(1, int(round(n * min(1.0, strength))))
    idx = list(range(n))
    r.shuffle(idx)
    chosen = sorted(idx[:k])
    sub = [sents[i] for i in chosen]
    r.shuffle(sub)
    for j, i in enumerate(chosen):
        sents[i] = sub[j]
    return sents


def block_shuffle(sents: List[str], strength: float, seed: int = 42) -> List[str]:

    if len(sents) <= 2 or strength <= 0:
        return list(sents)
    r = _rng(seed)
    n = len(sents)

    block_size = max(1, int(round((1.0 - min(1.0, strength)) * n)))
    block_size = min(block_size, n)
    blocks = [sents[i : i + block_size] for i in range(0, n, block_size)]
    r.shuffle(blocks)
    out = []
    for b in blocks:
        out.extend(b)
    return out


def intra_shuffle(sents: List[str], strength: float, seed: int = 42) -> List[str]:

    if strength <= 0:
        return list(sents)
    r = _rng(seed)
    out = []
    for s in sents:
        toks = s.split()
        if len(toks) <= 4:
            out.append(s)
            continue

        win = max(2, int(round(len(toks) * min(0.5, strength))))
        start = r.randint(0, max(0, len(toks) - win))
        seg = toks[start : start + win]
        r.shuffle(seg)
        toks[start : start + win] = seg
        out.append(" ".join(toks))
    return out


# _FUNC_WORDS = {
#     "a", "an", "the", "and", "or", "but", "if", "then", "else",
#     "in", "on", "at", "to", "of", "for", "with", "without", "from",
#     "is", "are", "was", "were", "be", "been", "being",
#     "this", "that", "these", "those",
# }

def build_func_words(use_nltk: bool = True):
    base = {
        "a","an","the","and","or","but","if","then","else",
        "in","on","at","to","of","for","with","without","from",
        "is","are","was","were","be","been","being",
        "this","that","these","those",
    }

    if not use_nltk:
        return base

    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english"))
    except Exception:

        return base


    blacklist = {
        "no","not","nor","n't",
        "i","me","my","myself","we","our","ours","ourselves",
        "you","your","yours","yourself","yourselves",
        "he","him","his","himself","she","her","hers","herself",
        "it","its","itself","they","them","their","theirs","themselves",
        "what","which","who","whom","why","how","when","where",
    }


    sw = {w for w in sw if w.isalpha()}
    sw = sw - blacklist


    return base | sw

_FUNC_WORDS = build_func_words(use_nltk=True)


def function_word_substitution(sents: List[str], strength: float, seed: int = 42) -> List[str]:

    if strength <= 0:
        return list(sents)
    r = _rng(seed)
    repl = {
        "and": ["as well as", "along with"],
        "or": ["alternatively"],
        "but": ["however"],
        "because": ["since"],
        "with": ["using"],
        "of": ["regarding"],
        "in": ["within"],
    }
    out = []
    for s in sents:
        toks = s.split()
        new = []
        for t in toks:
            t_l = re.sub(r"[^A-Za-z]", "", t).lower()
            if t_l in _FUNC_WORDS and r.random() < strength:

                if t_l in repl and r.random() < 0.7:
                    cand = repl[t_l]
                    new.append(r.choice(cand))
                else:
                    # drop
                    continue
            else:
                new.append(t)
        out.append(" ".join(new))
    return out


def passive_transform(sents: List[str], strength: float, seed: int = 42) -> List[str]:

    if strength <= 0:
        return list(sents)
    r = _rng(seed)
    out = []
    for s in sents:
        if r.random() < strength and len(s.split()) > 6:
            out.append(s + " , which is reported by prior work .")
        else:
            out.append(s)
    return out


def nominalization_transform(sents: List[str], strength: float, seed: int = 42) -> List[str]:

    if strength <= 0:
        return list(sents)
    r = _rng(seed)
    out = []
    for s in sents:
        if r.random() < strength and len(s.split()) > 6:
            out.append("The analysis of " + s)
        else:
            out.append(s)
    return out


def synonym_substitution(sents: List[str], strength: float, seed: int = 42) -> List[str]:

    if strength <= 0:
        return list(sents)
    r = _rng(seed)
    syn = {
        "important": ["crucial", "significant"],
        "novel": ["new", "innovative"],
        "improve": ["enhance", "boost"],
        "method": ["approach", "technique"],
        "results": ["findings", "outcomes"],
        "demonstrate": ["show", "indicate"],
    }
    out = []
    for s in sents:
        toks = s.split()
        new = []
        for t in toks:
            key = re.sub(r"[^A-Za-z]", "", t).lower()
            if key in syn and r.random() < strength:
                rep = r.choice(syn[key])

                if t[:1].isupper():
                    rep = rep.capitalize()
                new.append(rep)
            else:
                new.append(t)
        out.append(" ".join(new))
    return out


STRATEGY_FUNCTIONS: Dict[str, Callable[[List[str], float], List[str]]] = {
    "none": lambda s, p: noop(s, p),
    "global_shuffle": lambda s, p: global_shuffle(s, p),
    "block_shuffle": lambda s, p: block_shuffle(s, p),
    "intra_shuffle": lambda s, p: intra_shuffle(s, p),
    "function_word_sub": lambda s, p: function_word_substitution(s, p),
    "passive": lambda s, p: passive_transform(s, p),
    "nominal": lambda s, p: nominalization_transform(s, p),
    "synonym": lambda s, p: synonym_substitution(s, p),
}


# ============================

# ============================

LAYER_PERTURB_OPTIONS: Dict[str, List[str]] = {

    "O": [
        "none_0.0",
        "global_shuffle_0.3",
        "global_shuffle_0.6",
        "block_shuffle_0.3",
        "block_shuffle_0.6",
        "intra_shuffle_0.2",
        "intra_shuffle_0.4",
    ],

    "S": [
        "none_0.0",
        "function_word_sub_0.10",
        "function_word_sub_0.20",
        "passive_0.10",
        "passive_0.20",
        "nominal_0.10",
        "nominal_0.20",
    ],

    "L": [
        "none_0.0",
        "synonym_0.10",
        "synonym_0.20",
        "synonym_0.30",
    ],
}


def parse_option(option: str) -> Tuple[str, float]:

    if option.count("_") == 0:
        return option, 0.0
    parts = option.rsplit("_", 1)
    if len(parts) != 2:
        return option, 0.0
    strat = parts[0]
    try:
        strength = float(parts[1])
    except Exception:
        strength = 0.0
    return strat, strength


# ============================

# ============================


class PerturbatorPolicy(nn.Module):


    def __init__(self, input_dim: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PerturbatorLoadConfig:


    save_dir: str | None = None
    hidden: int = 64


def load_or_init_model(layer: str, cfg: PerturbatorLoadConfig | None = None) -> PerturbatorPolicy:


    if layer not in LEVEL_FEATURE_KEYS:
        raise ValueError(f"Unknown layer={layer}. Expected one of {list(LEVEL_FEATURE_KEYS.keys())}")
    input_dim = len(LEVEL_FEATURE_KEYS[layer])
    num_classes = len(LAYER_PERTURB_OPTIONS[layer])
    hidden = 64 if cfg is None else cfg.hidden
    model = PerturbatorPolicy(input_dim=input_dim, num_classes=num_classes, hidden=hidden)

    if cfg is not None and cfg.save_dir:
        ckpt = os.path.join(cfg.save_dir, f"perturbator_{layer}.pt")
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)

    return model
