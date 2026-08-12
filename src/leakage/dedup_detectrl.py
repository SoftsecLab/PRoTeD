# -*- coding: utf-8 -*-


import os
import re
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

RAW_DIR = Path("data/raw/DetectRL_clear")
OUT_DIR = Path("data/raw/DetectRL")
REPORT_JSON = OUT_DIR / "dedup_report.json"
REPORT_CSV  = OUT_DIR / "dedup_report.csv"

# =========================
# normalize + hash
# =========================
_ws_re = re.compile(r"\s+")
_punc_re = re.compile(r"[^\w\s]", flags=re.UNICODE)

def normalize_text(text: str, mode: str = "light") -> str:
    """
    light : lower + collapse spaces
    strict: light + remove punctuation
    """
    if text is None:
        return ""
    t = text.strip().lower()
    t = _ws_re.sub(" ", t)
    if mode == "strict":
        t = _punc_re.sub("", t)
        t = _ws_re.sub(" ", t).strip()
    return t

def h_text(text: str, mode: str = "light") -> str:
    return hashlib.sha1(normalize_text(text, mode=mode).encode("utf-8")).hexdigest()

# =========================
# I/O
# =========================
def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =========================
# Detect format
# =========================
HUMAN_TYPES = {"abstract", "document", "story", "content"}

def is_flat_format(data) -> bool:
    if not isinstance(data, list) or len(data) == 0:
        return False
    x = data[0]
    return isinstance(x, dict) and ("text" in x) and ("label" in x) and not isinstance(x.get("label"), dict)

def is_nested_format(data) -> bool:
    if not isinstance(data, list) or len(data) == 0:
        return False
    x = data[0]
    return isinstance(x, dict) and isinstance(x.get("label", None), dict)

# =========================
# Near-duplicate: SimHash + LSH + Jaccard
# =========================
def token_ngrams(words: List[str], n: int):
    if len(words) < n:
        return []
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

def simhash64(text: str, ngram: int = 3) -> int:
    """
    64-bit SimHash on token n-grams (fast enough for filtering).
    """
    t = normalize_text(text, mode="light")
    words = t.split()
    feats = token_ngrams(words, ngram)
    if not feats:
        feats = words  # fallback

    v = [0] * 64
    for f in feats:
        h = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if ((h >> i) & 1) else -1

    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out

def lsh_keys(x: int, bands: int = 4, bits_per_band: int = 16):
    """
    64-bit -> 4 bands x 16-bit
    """
    keys = []
    mask = (1 << bits_per_band) - 1
    for b in range(bands):
        keys.append((b, (x >> (b * bits_per_band)) & mask))
    return keys

def char_ngrams(text: str, n: int = 5) -> set:
    t = normalize_text(text, mode="strict")
    if not t:
        return set()
    if len(t) < n:
        return {t}
    return {t[i:i+n] for i in range(len(t) - n + 1)}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

class NearDupIndex:

    def __init__(self, test_texts: List[str], jacc_thr: float = 0.90, max_cand: int = 2000):
        self.jacc_thr = jacc_thr
        self.max_cand = max_cand

        self.test_texts = test_texts
        self.test_sim = [simhash64(t) for t in test_texts]
        self.buckets = defaultdict(list)  # (band,val)-> [test_idx...]

        for i, sh in enumerate(self.test_sim):
            for k in lsh_keys(sh):
                self.buckets[k].append(i)

        self.test_grams_cache: Dict[int, set] = {}

    def is_near_dup_of_test(self, train_text: str) -> bool:

        tsh = simhash64(train_text)
        cand = set()
        for k in lsh_keys(tsh):
            for idx in self.buckets.get(k, []):
                cand.add(idx)

        if not cand:
            return False
        if len(cand) > self.max_cand:

            cand = set(list(cand)[:self.max_cand])

        tgrams = char_ngrams(train_text, n=5)
        for idx in cand:
            if idx not in self.test_grams_cache:
                self.test_grams_cache[idx] = char_ngrams(self.test_texts[idx], n=5)
            if jaccard(tgrams, self.test_grams_cache[idx]) >= self.jacc_thr:
                return True
        return False

# =========================
# Pair discovery: prefix_train.json & prefix_test.json
# =========================
def find_pairs(raw_dir: Path):
    pairs = defaultdict(dict)
    for p in raw_dir.glob("*.json"):
        name = p.name
        if name.endswith("_train.json"):
            pairs[name[:-10]]["train"] = p
        elif name.endswith("_test.json"):
            pairs[name[:-9]]["test"] = p
    return pairs

# =========================
# flat train filtering
# =========================
def extract_text_flat(item: Dict[str, Any]) -> str:
    return item.get("text", "") or ""

def build_test_hash_set_flat(test_data: List[Dict[str, Any]], mode: str) -> set:
    hs = set()
    for it in test_data:
        hs.add(h_text(extract_text_flat(it), mode=mode))
    return hs

def filter_train_flat(
    train_data: List[Dict[str, Any]],
    test_hashes: set,
    near_index: Optional[NearDupIndex],
    mode: str = "light",
):

    seen_train = set()
    filtered = []
    stats = {
        "train_before": len(train_data),
        "train_internal_dup_removed": 0,
        "train_test_exact_removed": 0,
        "train_test_near_removed": 0,
    }

    for it in train_data:
        text = extract_text_flat(it)
        key = h_text(text, mode=mode)


        if key in seen_train:
            stats["train_internal_dup_removed"] += 1
            continue
        seen_train.add(key)


        if key in test_hashes:
            stats["train_test_exact_removed"] += 1
            continue


        if near_index is not None and near_index.is_near_dup_of_test(text):
            stats["train_test_near_removed"] += 1
            continue

        filtered.append(it)

    stats["train_after"] = len(filtered)
    return filtered, stats

# =========================

# =========================
def filter_train_nested(
    train_data: List[Dict[str, Any]],
    test_hashes: set,
    near_index: Optional[NearDupIndex],
    mode: str = "light",
):

    stats = {
        "train_before": len(train_data),
        "human_internal_dup_removed": 0,
        "human_test_exact_removed": 0,
        "human_test_near_removed": 0,
        "gen_internal_dup_removed": 0,
        "gen_test_exact_removed": 0,
        "gen_test_near_removed": 0,
    }

    human_seen = set()
    new_data = []

    for item in train_data:
        dt = item.get("data_type", "")

        if dt in HUMAN_TYPES:
            text = item.get("text", "") or ""
            key = h_text(text, mode=mode)

            if key in human_seen:
                stats["human_internal_dup_removed"] += 1
                continue
            human_seen.add(key)

            if key in test_hashes:
                stats["human_test_exact_removed"] += 1
                continue

            if near_index is not None and near_index.is_near_dup_of_test(text):
                stats["human_test_near_removed"] += 1
                continue

            new_data.append(item)
            continue

        # gen part
        lbl = item.get("label", {})
        if not isinstance(lbl, dict):
            new_data.append(item)
            continue

        new_lbl = {}
        for llm_name, attack_dict in lbl.items():
            if not isinstance(attack_dict, dict):
                continue
            new_attack = {}
            for attack, texts in attack_dict.items():
                if not isinstance(texts, list):
                    continue
                seen_local = set()
                kept = []
                for t in texts:
                    if not isinstance(t, str):
                        continue
                    key = h_text(t, mode=mode)

                    if key in seen_local:
                        stats["gen_internal_dup_removed"] += 1
                        continue
                    seen_local.add(key)

                    if key in test_hashes:
                        stats["gen_test_exact_removed"] += 1
                        continue

                    if near_index is not None and near_index.is_near_dup_of_test(t):
                        stats["gen_test_near_removed"] += 1
                        continue

                    kept.append(t)
                new_attack[attack] = kept
            new_lbl[llm_name] = new_attack

        new_item = dict(item)
        new_item["label"] = new_lbl
        new_data.append(new_item)

    stats["train_after"] = len(new_data)
    return new_data, stats

# =========================
# CSV writer
# =========================
def write_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

# =========================
# Main
# =========================
def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)


    NEAR_JACC_THR = 0.95
    USE_NEAR_DUP = True


    HASH_MODE = "light"

    pairs = find_pairs(RAW_DIR)
    report = {
        "raw_dir": str(RAW_DIR),
        "out_dir": str(OUT_DIR),
        "settings": {
            "hash_mode": HASH_MODE,
            "use_near_dup": USE_NEAR_DUP,
            "near_jacc_thr": NEAR_JACC_THR,
            "note": "test files are NOT modified; only train is filtered/deduplicated.",
        },
        "datasets": {},
    }
    csv_rows = []

    prefixes = sorted(pairs.keys())
    for prefix in tqdm(prefixes, desc="Datasets"):
        train_path = pairs[prefix].get("train")
        test_path  = pairs[prefix].get("test")

        ds_info = {
            "train_path": str(train_path) if train_path else None,
            "test_path": str(test_path) if test_path else None,
            "format_train": None,
            "format_test": None,
            "stats": {},
            "errors": [],
        }

        if not train_path or not train_path.exists():
            ds_info["errors"].append("missing_train")
            report["datasets"][prefix] = ds_info
            continue

        if not test_path or not test_path.exists():
            ds_info["errors"].append("missing_test")

            test_data = []
            test_hashes = set()
            near_index = None
        else:
            test_data = read_json(test_path)


            if is_flat_format(test_data):
                ds_info["format_test"] = "flat"
                test_texts = [it.get("text", "") or "" for it in test_data]
                test_hashes = build_test_hash_set_flat(test_data, mode=HASH_MODE)
                near_index = NearDupIndex(test_texts, jacc_thr=NEAR_JACC_THR) if USE_NEAR_DUP else None
            elif is_nested_format(test_data):
                ds_info["format_test"] = "nested"

                test_texts = [(it.get("text", "") or "") for it in test_data if isinstance(it, dict)]
                test_hashes = {h_text(t, mode=HASH_MODE) for t in test_texts if t}
                near_index = NearDupIndex(test_texts, jacc_thr=NEAR_JACC_THR) if (USE_NEAR_DUP and test_texts) else None
            else:
                ds_info["format_test"] = "unknown"
                test_hashes = set()
                near_index = None

        train_data = read_json(train_path)


        out_train_path = OUT_DIR / train_path.name


        if is_flat_format(train_data):
            ds_info["format_train"] = "flat"


            stats = {
                "train_before": len(train_data),
                "train_internal_dup_removed": 0,
                "train_test_exact_removed": 0,
                "train_test_near_removed": 0,
            }
            seen_train = set()
            filtered = []
            iters = tqdm(train_data, desc=f"{prefix}: filter train", leave=False)

            for it in iters:
                text = it.get("text", "") or ""
                key = h_text(text, mode=HASH_MODE)

                if key in seen_train:
                    stats["train_internal_dup_removed"] += 1
                    continue
                seen_train.add(key)

                if key in test_hashes:
                    stats["train_test_exact_removed"] += 1
                    continue

                if near_index is not None and near_index.is_near_dup_of_test(text):
                    stats["train_test_near_removed"] += 1
                    continue

                filtered.append(it)


                iters.set_postfix({
                    "kept": len(filtered),
                    "dup_rm": stats["train_internal_dup_removed"],
                    "ex_rm": stats["train_test_exact_removed"],
                    "near_rm": stats["train_test_near_removed"],
                })

            stats["train_after"] = len(filtered)
            ds_info["stats"] = stats
            write_json(out_train_path, filtered)

        elif is_nested_format(train_data):
            ds_info["format_train"] = "nested"
            filtered, stats = filter_train_nested(train_data, test_hashes=test_hashes, near_index=near_index, mode=HASH_MODE)
            ds_info["stats"] = stats
            write_json(out_train_path, filtered)

        else:
            ds_info["format_train"] = "unknown"
            ds_info["errors"].append("unknown_train_format")

            write_json(out_train_path, train_data)
            ds_info["stats"] = {"train_before": "unknown", "train_after": "unknown"}

        report["datasets"][prefix] = ds_info


        st = ds_info["stats"]
        row = {
            "dataset": prefix,
            "train_format": ds_info["format_train"],
            "test_format": ds_info["format_test"],
            "train_before": st.get("train_before", ""),
            "train_after": st.get("train_after", ""),
            "train_internal_dup_removed": st.get("train_internal_dup_removed", st.get("human_internal_dup_removed", "")),
            "train_test_exact_removed": st.get("train_test_exact_removed", st.get("human_test_exact_removed", "")),
            "train_test_near_removed": st.get("train_test_near_removed", st.get("human_test_near_removed", "")),
            "out_train_path": str(out_train_path),
        }
        csv_rows.append(row)

    write_json(REPORT_JSON, report)
    write_csv(csv_rows, REPORT_CSV)

    print(f"\n[OK] Train-only dedup done.")
    print(f"[OK] Output train dir: {OUT_DIR}")
    print(f"[OK] Report JSON: {REPORT_JSON}")
    print(f"[OK] Report CSV : {REPORT_CSV}")
    print(f"[NOTE] Test files were NOT modified.")

if __name__ == "__main__":
    main()
