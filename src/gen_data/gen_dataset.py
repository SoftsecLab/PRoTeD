# -*- coding: utf-8 -*-


import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# =========================

# =========================
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(path: str, data: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

# =========================

# =========================
def build_label_indices(samples: List[Dict]) -> Tuple[List[int], List[int]]:
    idx0, idx1 = [], []
    for i, s in enumerate(samples):
        y = int(s["label"])
        (idx0 if y == 0 else idx1).append(i)
    return idx0, idx1

def stratified_split(samples: List[Dict], ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    idx0, idx1 = build_label_indices(samples)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_a = int(round(len(idx0) * ratio))
    n1_a = int(round(len(idx1) * ratio))

    a_idx = idx0[:n0_a] + idx1[:n1_a]
    b_idx = idx0[n0_a:] + idx1[n1_a:]
    rng.shuffle(a_idx)
    rng.shuffle(b_idx)

    A = [samples[i] for i in a_idx]
    B = [samples[i] for i in b_idx]
    return A, B

def make_cheat_datasets(
    cheat_dir: str,
    out_dir: str,
    total_samples: int = 6000,
    test_ratio: float = 0.4,
    test_seed: int = 202401,
    data_seeds: List[int] = None,
):
    if data_seeds is None:
        data_seeds = [11, 22, 33]

    ieee_init   = os.path.join(cheat_dir, "ieee-init.jsonl")
    ieee_gen    = os.path.join(cheat_dir, "ieee-chatgpt-generation.jsonl")
    ieee_polish = os.path.join(cheat_dir, "ieee-chatgpt-polish.jsonl")

    human  = read_jsonl(ieee_init)
    gen    = read_jsonl(ieee_gen)
    polish = read_jsonl(ieee_polish)

    out_root = Path(out_dir) / "CHEAT"
    out_root.mkdir(parents=True, exist_ok=True)

    def get_cheat_text(item: Dict) -> str:
        if "abstract" in item and item["abstract"]:
            return item["abstract"]
        return item.get("text", "") or ""

    def build_pool(human_list: List[Dict], other_list: List[Dict], seed_for_pool: int) -> List[Dict]:
        max_each = total_samples // 2
        rng = random.Random(seed_for_pool)
        h, o = human_list[:], other_list[:]
        rng.shuffle(h)
        rng.shuffle(o)
        h_sel, o_sel = h[:max_each], o[:max_each]

        pool = [{"text": get_cheat_text(x), "label": 0} for x in h_sel] + \
               [{"text": get_cheat_text(x), "label": 1} for x in o_sel]
        rng.shuffle(pool)
        return pool

    for tag, other in [("ieee-generation", gen), ("ieee-polish", polish)]:
        pool = build_pool(human, other, seed_for_pool=test_seed)


        test_set, train_rest = stratified_split(pool, ratio=test_ratio, seed=test_seed)
        write_jsonl(str(out_root / f"{tag}-test.jsonl"), test_set)


        for ds in data_seeds:
            rng = random.Random(ds)
            train_set = train_rest[:]
            rng.shuffle(train_set)
            write_jsonl(str(out_root / f"{tag}-seed{ds}-train.jsonl"), train_set)

        print(f"[CHEAT] {tag}: test={len(test_set)}, train={len(train_rest)}, seeds={data_seeds}")


# =========================

# =========================
HUMAN_TYPES = {"abstract", "document", "story", "content"}

def normalize_label(x: Any) -> Optional[int]:
    if x is None: return None
    if isinstance(x, bool): return 1 if x else 0
    if isinstance(x, (int, float)):
        xi = int(x)
        return xi if xi in (0, 1) else None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"0", "human", "real", "human-written", "human_written"}: return 0
        if s in {"1", "llm", "ai", "machine", "generated"}: return 1
    return None

def get_detectrl_text(item: Dict[str, Any]) -> str:
    for k in ["text", "content", "document", "abstract", "story"]:
        if k in item and item.get(k) is not None:
            v = item.get(k)
            return v if isinstance(v, str) else str(v)
    return ""

def infer_label(item: Dict[str, Any]) -> int:
    for k in ["label", "y", "gold", "target", "class"]:
        if k in item:
            y = normalize_label(item.get(k))
            if y is not None: return y
    for k in ["data_type", "type", "source_type", "category"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip().lower() in HUMAN_TYPES: return 0
    return 1

def is_human_sample(item: Dict[str, Any]) -> bool:
    lab = normalize_label(item.get("label"))
    if lab is not None: return lab == 0
    dt = item.get("data_type", "") or item.get("type", "") or ""
    return isinstance(dt, str) and dt.strip().lower() in HUMAN_TYPES

def segment_signature(item: Dict[str, Any]) -> Tuple[str, str, str, str]:
    def _str(v): return str(v).strip() if v is not None else ""
    dt = _str(item.get("data_type", "unknown"))
    llm_type = _str(item.get("llm_type", ""))
    domain = _str(item.get("domain", ""))
    extra = _str(item.get("generator", "") or item.get("model", ""))
    return (dt or "unknown", llm_type, domain, extra)

def split_into_segments_llm(data: List[Dict[str, Any]]) -> List[List[str]]:
    segments, cur = [], []
    prev_sig = None
    for item in data:
        if is_human_sample(item): continue
        txt = get_detectrl_text(item)
        if not txt: continue
        sig = segment_signature(item)
        if prev_sig is None:
            prev_sig, cur = sig, [txt]
        elif sig == prev_sig:
            cur.append(txt)
        else:
            if cur: segments.append(cur)
            prev_sig, cur = sig, [txt]
    if cur: segments.append(cur)
    return segments

def collect_human_texts(data: List[Dict[str, Any]]) -> List[str]:
    return [get_detectrl_text(item) for item in data if is_human_sample(item) and get_detectrl_text(item)]


# =========================

# =========================
def generate_detectrl_test_data(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith("_test.json"): continue
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname.replace("_test.json", "-test.jsonl"))

        data = read_json(in_path)
        rows, empty_text, missing_label = [], 0, 0

        for it in data:
            if not isinstance(it, dict):
                rows.append({"text": str(it) if it is not None else "", "label": 1})
                missing_label += 1
                continue

            txt = get_detectrl_text(it)
            if not txt: empty_text += 1

            y0 = normalize_label(it.get("label")) if "label" in it else None
            if y0 is None:
                y = infer_label(it)
                if "label" not in it: missing_label += 1
            else:
                y = y0

            rows.append({"text": txt, "label": int(y)})

        write_jsonl(out_path, rows)
        print(f"[DetectRL-Test] {fname} -> {os.path.basename(out_path)} n={len(rows)} (empty_text={empty_text})")


# =========================

# =========================
def generate_detectrl_train_data_seeded(input_dir: str, output_dir: str, seed: int, sample_size: int = 50):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    for fname in os.listdir(input_dir):
        if not fname.endswith("_train.json"): continue
        in_path = os.path.join(input_dir, fname)
        base_prefix = fname.replace("_train.json", "")
        out_path = os.path.join(output_dir, f"{base_prefix}-seed{seed}-train.jsonl")

        data = read_json(in_path)
        human_texts = collect_human_texts(data)
        segments = split_into_segments_llm(data)

        llm_texts = []
        for seg in segments:
            seg = [t for t in seg if t and t.strip()]
            if not seg: continue
            k = min(sample_size, len(seg))
            llm_texts.extend(rng.sample(seg, k) if len(seg) >= k else seg[:])


        rng.shuffle(human_texts)
        rng.shuffle(llm_texts)
        target = min(len(human_texts), len(llm_texts))
        human_texts = human_texts[:target]
        llm_texts = llm_texts[:target]

        final_rows = [{"text": t, "label": 0} for t in human_texts] + \
                     [{"text": t, "label": 1} for t in llm_texts]
        rng.shuffle(final_rows)
        write_jsonl(out_path, final_rows)

        print(f"[DetectRL-Train] {fname} -> {os.path.basename(out_path)} total={len(final_rows)} (segments={len(segments)})")


# =========================

# =========================
if __name__ == "__main__":

    OUT_DIR = "data/final_training_data"
    CHEAT_DIR = "data/raw/CHEAT"
    DETECTRL_DIR = "data/raw/DetectRL"

    DATA_SEEDS = [11, 22, 33]
    TEST_SEED = 202401

    print("=== 开始生成 CHEAT 数据集 ===")
    if os.path.exists(CHEAT_DIR):
        make_cheat_datasets(
            cheat_dir=CHEAT_DIR,
            out_dir=OUT_DIR,
            total_samples=6000,
            test_ratio=0.4,
            test_seed=TEST_SEED,
            data_seeds=DATA_SEEDS,
        )
    else:
        print(f"找不到 CHEAT 目录: {CHEAT_DIR}，跳过。")

    print("\n=== 开始生成 DetectRL 数据集 ===")
    if os.path.exists(DETECTRL_DIR):
        detectrl_out = os.path.join(OUT_DIR, "DetectRL")


        generate_detectrl_test_data(DETECTRL_DIR, detectrl_out)


        for s in DATA_SEEDS:
            generate_detectrl_train_data_seeded(
                input_dir=DETECTRL_DIR,
                output_dir=detectrl_out,
                sample_size=50,
                seed=s,
            )
    else:
        print(f"找不到 DetectRL 目录: {DETECTRL_DIR}，跳过。")

    print(f"\n全部处理完成，输出路径: {OUT_DIR}")