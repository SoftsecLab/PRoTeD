# -*- coding: utf-8 -*-
"""
Directory Comparison Tool for JSON/JSONL datasets.
Validates if the merged pipeline output matches the sequential pipeline output exactly.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Dict, Tuple, Set

# ----------------------------------------------------------------------
# Helper: Load Data
# ----------------------------------------------------------------------
def detect_format(path: Path) -> str:
    return "jsonl" if path.suffix.lower() == ".jsonl" else "json"

def load_records(path: Path) -> List[Any]:
    """
    Load data from file, normalizing everything to a List of records.
    Handles JSONL, JSON Lists, and JSON Dict wrappers ({"data": [...]}).
    """
    fmt = detect_format(path)
    try:
        if fmt == "jsonl":
            data = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        else:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            # Normalize structure
            if isinstance(raw, list):
                return raw
            elif isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
                return raw["data"]
            elif isinstance(raw, dict):
                # Single record case, wrap in list
                return [raw]
            else:
                return []
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return []

# ----------------------------------------------------------------------
# Helper: Deep Diff
# ----------------------------------------------------------------------
def find_first_diff(list_a: List[Any], list_b: List[Any]) -> str:
    """
    Compare two lists of records and return a string describing the first difference found.
    """
    if len(list_a) != len(list_b):
        return f"Count Mismatch: Base has {len(list_a)} records, Target has {len(list_b)} records."

    for i, (rec_a, rec_b) in enumerate(zip(list_a, list_b)):
        if rec_a != rec_b:
            # Attempt to find specific field mismatch if they are dicts
            if isinstance(rec_a, dict) and isinstance(rec_b, dict):
                keys_a = set(rec_a.keys())
                keys_b = set(rec_b.keys())

                if keys_a != keys_b:
                    return f"Record #{i}: Key mismatch. Base keys={keys_a}, Target keys={keys_b}"

                for k in keys_a:
                    if rec_a[k] != rec_b[k]:
                        val_a_short = str(rec_a[k])[:50] + "..." if len(str(rec_a[k])) > 50 else str(rec_a[k])
                        val_b_short = str(rec_b[k])[:50] + "..." if len(str(rec_b[k])) > 50 else str(rec_b[k])
                        return (f"Record #{i} mismatch in field '{k}':\n"
                                f"    Base:   {repr(val_a_short)}\n"
                                f"    Target: {repr(val_b_short)}")

            return f"Record #{i} mismatch (full object differs)."

    return "Identical"

# ----------------------------------------------------------------------
# Main Logic
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare two directories of JSON/JSONL files for logical equivalence.")
    parser.add_argument("--base_dir", required=True, help="The ground truth directory (e.g., clear6 output)")
    parser.add_argument("--target_dir", required=True, help="The new directory to check (e.g., merged output)")
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    target_path = Path(args.target_dir)

    if not base_path.exists():
        print(f"Error: Base directory does not exist: {base_path}")
        sys.exit(1)
    if not target_path.exists():
        print(f"Error: Target directory does not exist: {target_path}")
        sys.exit(1)

    # 1. Gather files
    print("Scanning files...")
    files_base = {p.relative_to(base_path) for p in base_path.rglob("*") if p.is_file() and p.suffix.lower() in ['.json', '.jsonl']}
    files_target = {p.relative_to(target_path) for p in target_path.rglob("*") if p.is_file() and p.suffix.lower() in ['.json', '.jsonl']}

    all_files = sorted(files_base | files_target)

    missing_in_target = files_base - files_target
    missing_in_base = files_target - files_base
    common_files = files_base & files_target

    print(f"Found {len(all_files)} unique files.")
    print(f"  - In Base only:   {len(missing_in_target)}")
    print(f"  - In Target only: {len(missing_in_base)}")
    print(f"  - Common files:   {len(common_files)}")
    print("-" * 60)

    # 2. Check structure
    if missing_in_target:
        print("[FAIL] The following files are missing in Target:")
        for f in sorted(missing_in_target):
            print(f"  x {f}")

    if missing_in_base:
        print("[WARN] Target has extra files (not necessarily an error, but check):")
        for f in sorted(missing_in_base):
            print(f"  + {f}")

    if not common_files:
        print("No common files to compare. Exiting.")
        sys.exit(1)

    # 3. Compare Content
    print(f"Deep comparing {len(common_files)} common files...")

    passed_count = 0
    failed_count = 0

    for rel_path in all_files:
        if rel_path not in common_files:
            continue

        p_base = base_path / rel_path
        p_target = target_path / rel_path

        # Load
        data_base = load_records(p_base)
        data_target = load_records(p_target)

        # Compare
        if data_base == data_target:
            passed_count += 1
            # Optional: Print progress for large datasets
            # print(f"[OK] {rel_path}")
        else:
            failed_count += 1
            diff_msg = find_first_diff(data_base, data_target)
            print(f"[MISMATCH] {rel_path}")
            print(f"  -> {diff_msg}")
            print("-" * 40)

    # 4. Summary
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Total Files Checked: {len(common_files)}")
    print(f"Identical:           {passed_count}")
    print(f"Mismatches:          {failed_count}")

    if len(missing_in_target) > 0:
        print(f"Missing Files:       {len(missing_in_target)} (FAILURE)")

    if failed_count == 0 and len(missing_in_target) == 0:
        print("\nResult: SUCCESS. The merged script output is identical to the baseline.")
        sys.exit(0)
    else:
        print("\nResult: FAILURE. Differences detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()