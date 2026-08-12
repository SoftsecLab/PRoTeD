# -*- coding: utf-8 -*-


import os
import re
import json
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable

# -------------------------

# -------------------------
ROOT_DIR = Path("data/final_training_data_stage1")
SCAN_DIRS = [
    ROOT_DIR / "CHEAT",
    ROOT_DIR / "DetectRL",
]
OUT_DIR = ROOT_DIR / "leakage_reports"
OUT_JSON = OUT_DIR / "leakage_report_stage1.json"
OUT_CSV = OUT_DIR / "leakage_report_stage1_summary.csv"

# -------------------------

# -------------------------
_ws_re = re.compile(r"\s+")
_punc_re = re.compile(r"[^\w\s]", flags=re.UNICODE)

def normalize_text(text: str, mode: str = "light") -> str:

    if text is None:
        return ""
    t = text.strip().lower()
    t = _ws_re.sub(" ", t)
    if mode == "strict":
        t = _punc_re.sub("", t)
        t = _ws_re.sub(" ", t).strip()
    return t

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -------------------------

# -------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def get_text(item: Dict[str, Any]) -> str:

    if "text" in item and item["text"]:
        return item["text"]
    if "abstract" in item and item["abstract"]:
        return item["abstract"]
    return ""

# -------------------------


#   - CHEAT: ieee-generation-test.jsonl / ieee-generation-seed11-train.jsonl
#   - DetectRL: xsum-test.jsonl / xsum-seed11-train.jsonl
# -------------------------
def parse_name(fn: str) -> Tuple[str, str, str]:

    name = fn.replace(".jsonl", "")
    # test: xxx-test
    if name.endswith("-test"):
        dataset = name[:-5]
        return dataset, "fixed", "test"
    # train: xxx-seed11-train
    m = re.match(r"^(.*)-seed(\d+)-train$", name)
    if m:
        return m.group(1), m.group(2), "train"
    return name, None, "other"

def discover_files() -> Dict[str, Dict[str, Any]]:

    datasets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"test": None, "trains": {}, "others": []})
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for p in d.glob("*.jsonl"):
            ds, seed, split = parse_name(p.name)
            if split == "test":
                datasets[ds]["test"] = p
            elif split == "train" and seed is not None:
                datasets[ds]["trains"][seed] = p
            else:
                datasets[ds]["others"].append(p)
    return datasets

# -------------------------
# Exact duplicate statistics
# -------------------------
def build_hash_index(texts: List[str], mode: str) -> Dict[str, List[int]]:
    """
    hash -> list of indices
    """
    mp = defaultdict(list)
    for i, t in enumerate(texts):
        h = sha1(normalize_text(t, mode=mode))
        mp[h].append(i)
    return mp

def count_internal_duplicates(hash_index: Dict[str, List[int]]) -> Tuple[int, int]:

    dup_groups = 0
    dup_items = 0
    for _, idxs in hash_index.items():
        if len(idxs) > 1:
            dup_groups += 1
            dup_items += (len(idxs) - 1)
    return dup_groups, dup_items

def exact_overlap(train_hash: Dict[str, List[int]], test_hash: Dict[str, List[int]]) -> Tuple[int, List[str]]:

    common = set(train_hash.keys()) & set(test_hash.keys())
    overlap = 0
    examples = []
    for h in common:
        overlap += min(len(train_hash[h]), len(test_hash[h]))
        if len(examples) < 20:
            examples.append(h)
    return overlap, examples

# -------------------------
# Near-duplicate: SimHash + LSH buckets + Jaccard verify
# -------------------------
def token_ngrams(words: List[str], n: int) -> Iterable[str]:
    if len(words) < n:
        return []
    for i in range(len(words) - n + 1):
        yield " ".join(words[i:i+n])

def simhash64(text: str, ngram: int = 3) -> int:
    """
    64-bit SimHash on token n-grams.
    """
    t = normalize_text(text, mode="light")
    words = t.split()
    feats = list(token_ngrams(words, ngram))
    if not feats:
        feats = words  # fallback

    v = [0] * 64
    for f in feats:
        h = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out

def lsh_keys(x: int, bands: int = 4, bits_per_band: int = 16) -> List[Tuple[int, int]]:

    keys = []
    mask = (1 << bits_per_band) - 1
    for b in range(bands):
        val = (x >> (b * bits_per_band)) & mask
        keys.append((b, val))
    return keys

def char_ngrams(text: str, n: int = 5) -> set:
    t = normalize_text(text, mode="strict")
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i+n] for i in range(len(t) - n + 1)}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def near_duplicate_stats(
    train_texts: List[str],
    test_texts: List[str],
    jacc_thrs: Tuple[float, float] = (0.90, 0.80),
    max_candidates_per_test: int = 2000,
    max_examples: int = 20,
) -> Dict[str, Any]:


    train_sim = [simhash64(t) for t in train_texts]
    buckets = defaultdict(list)  # (band, val) -> list of train_idx
    for i, sh in enumerate(train_sim):
        for k in lsh_keys(sh):
            buckets[k].append(i)


    near_counts = {thr: 0 for thr in jacc_thrs}
    examples = []
    candidates_total = 0



    train_grams_cache: Dict[int, set] = {}

    for j, tt in enumerate(test_texts):
        tsh = simhash64(tt)
        cand = set()
        for k in lsh_keys(tsh):
            for i in buckets.get(k, []):
                cand.add(i)

        if len(cand) > max_candidates_per_test:
            cand = set(list(cand)[:max_candidates_per_test])

        candidates_total += len(cand)

        tgrams = char_ngrams(tt, n=5)
        for i in cand:
            if i not in train_grams_cache:
                train_grams_cache[i] = char_ngrams(train_texts[i], n=5)
            score = jaccard(train_grams_cache[i], tgrams)


            for thr in jacc_thrs:
                if score >= thr:
                    near_counts[thr] += 1


            if score >= max(jacc_thrs) and len(examples) < max_examples:
                examples.append({"train_idx": i, "test_idx": j, "jaccard_char5": round(score, 4)})

    return {
        "candidates_total": candidates_total,
        "near_ge_%.2f" % jacc_thrs[0]: near_counts[jacc_thrs[0]],
        "near_ge_%.2f" % jacc_thrs[1]: near_counts[jacc_thrs[1]],
        "examples": examples,
    }

# -------------------------
# Main report
# -------------------------
def summarize_lengths(texts: List[str]) -> Dict[str, float]:
    lens = [len(t or "") for t in texts]
    if not lens:
        return {"n": 0}
    lens_sorted = sorted(lens)
    def pct(p):
        k = int(round((p/100.0) * (len(lens_sorted)-1)))
        return float(lens_sorted[k])
    return {
        "n": len(lens),
        "len_mean": float(sum(lens)/len(lens)),
        "len_p50": pct(50),
        "len_p90": pct(90),
        "len_p95": pct(95),
        "len_max": float(max(lens)),
    }

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = discover_files()

    report: Dict[str, Any] = {
        "root_dir": str(ROOT_DIR),
        "scanned_dirs": [str(x) for x in SCAN_DIRS],
        "datasets_found": sorted(datasets.keys()),
        "datasets": {},
        "notes": [
            "Exact overlap uses normalized hash (light/strict).",
            "Near-duplicate uses SimHash(64) LSH candidates + char5gram Jaccard verify.",
            "Near-dup counts are number of candidate pairs above threshold (not unique items).",
        ],
    }

    summary_rows = []

    for ds in sorted(datasets.keys()):
        info = datasets[ds]
        test_path = info["test"]
        trains = info["trains"]

        ds_out = {
            "test_path": str(test_path) if test_path else None,
            "train_paths": {k: str(v) for k, v in sorted(trains.items())},
            "errors": [],
            "splits": {},
        }

        if test_path is None:
            ds_out["errors"].append("Missing test file")
            report["datasets"][ds] = ds_out
            continue


        test_data = read_jsonl(test_path)
        test_texts = [get_text(x) for x in test_data]
        ds_out["splits"]["test"] = {
            "n": len(test_texts),
            "lengths": summarize_lengths(test_texts),
        }


        for mode in ["light", "strict"]:
            hidx = build_hash_index(test_texts, mode=mode)
            g, it = count_internal_duplicates(hidx)
            ds_out["splits"]["test"][f"internal_dups_{mode}"] = {"dup_groups": g, "dup_items": it}


        for seed, tr_path in sorted(trains.items(), key=lambda x: int(x[0])):
            tr_data = read_jsonl(tr_path)
            tr_texts = [get_text(x) for x in tr_data]

            split_key = f"train_seed{seed}"
            ds_out["splits"][split_key] = {
                "path": str(tr_path),
                "n": len(tr_texts),
                "lengths": summarize_lengths(tr_texts),
            }


            for mode in ["light", "strict"]:
                tr_hidx = build_hash_index(tr_texts, mode=mode)
                g, it = count_internal_duplicates(tr_hidx)
                ds_out["splits"][split_key][f"internal_dups_{mode}"] = {"dup_groups": g, "dup_items": it}

            # train-test exact overlap
            for mode in ["light", "strict"]:
                tr_hidx = build_hash_index(tr_texts, mode=mode)
                te_hidx = build_hash_index(test_texts, mode=mode)
                ov, ex = exact_overlap(tr_hidx, te_hidx)
                ds_out["splits"][split_key][f"exact_overlap_{mode}"] = {
                    "overlap_items": ov,
                    "example_hashes": ex,
                }

            # train-test near duplicate
            nd = near_duplicate_stats(tr_texts, test_texts, jacc_thrs=(0.90, 0.80))
            ds_out["splits"][split_key]["near_duplicate"] = nd


            summary_rows.append({
                "dataset": ds,
                "train_seed": seed,
                "train_n": len(tr_texts),
                "test_n": len(test_texts),
                "train_internal_dup_items_light": ds_out["splits"][split_key]["internal_dups_light"]["dup_items"],
                "test_internal_dup_items_light": ds_out["splits"]["test"]["internal_dups_light"]["dup_items"],
                "exact_overlap_light": ds_out["splits"][split_key]["exact_overlap_light"]["overlap_items"],
                "exact_overlap_strict": ds_out["splits"][split_key]["exact_overlap_strict"]["overlap_items"],
                "near_ge_0.90_pairs": nd.get("near_ge_0.90", 0),
                "near_ge_0.80_pairs": nd.get("near_ge_0.80", 0),
                "near_candidates_total": nd.get("candidates_total", 0),
            })

        report["datasets"][ds] = ds_out


    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)



    if summary_rows:
        cols = list(summary_rows[0].keys())
        with OUT_CSV.open("w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in summary_rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    print(f"[OK] JSON report saved to: {OUT_JSON}")
    print(f"[OK] CSV  summary saved to: {OUT_CSV}")
    print(f"[INFO] datasets found: {len(datasets)}")


if __name__ == "__main__":
    run()
