# python -m scripts.visual.plot_model_metrics
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.cm import get_cmap

import re
def natural_key(s: str):

    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


INPUT_FILE = "data/T5_evaluate/metrics_eval_results.jsonl"
SAVE_DIR = "src/T5_Evaluate/visual/"
os.makedirs(SAVE_DIR, exist_ok=True)
def plot_model_single_metrics():

    delta_values = defaultdict(lambda: defaultdict(list))


    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            group = json.loads(line)
            model_id = group['model']
            for item in group["metrics"]:
                pred = item["single_metrics"]["prediction"]
                ref = item["single_metrics"]["reference"]
                for metric in pred:
                    delta = pred[metric] - ref[metric]
                    if metric == "HierarchicalDensity":
                        delta *= 10
                    delta_values[metric][model_id].append(delta)


    delta_avg = {
        metric: {
            model: np.mean(vals)
            for model, vals in models.items()
        }
        for metric, models in delta_values.items()
    }


    selected_metrics = list(delta_avg.keys())[:6]


    fig, axes = plt.subplots(3, 2, figsize=(20, 15))  # 3 rows, 2 columns
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    cmap = plt.get_cmap("tab10")


    for idx, metric in enumerate(selected_metrics):
        ax = axes[idx]
        values = delta_avg[metric]
        model_ids = sorted(values.keys())
        y = [values[mid] for mid in model_ids]
        x = np.arange(len(model_ids))
        colors = [cmap(i % 10) for i in range(len(model_ids))]

        formatted_labels = [
            mid.replace("t5-small_group_", "S")
            .replace("t5-base_group_", "B")
            .replace("t5-large_group_", "L")
            .replace("t5-base_random_", "R")
            .replace("t5-base_reorder_", "r")
            .replace("t5-base_tau_", "b")
            for mid in model_ids
        ]

        ax.plot(x, y, linestyle='-', marker='o', color='tab:blue', label=metric)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f"Δ {metric} (Prediction - Reference)")
        ax.set_xticks(x)
        ax.set_xticklabels(formatted_labels, rotation=0, fontsize=8)
        ax.set_ylabel("Δ Value")

        for xi, yi in zip(x, y):
            ax.text(xi, yi, f"{yi:.2f}", ha='center', va='bottom' if yi >= 0 else 'top', fontsize=7)


    for j in range(len(selected_metrics), 6):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "single_metrics_deltas_line.svg")
    plt.savefig(save_path,format='svg')
    plt.close()
    print(f"✅ 折线图已保存至：{save_path}")


def plot_model_pair_metrics():

    metric_groups = {
        "ROUGE": ["ROUGE-1", "ROUGE-2", "ROUGE-3", "ROUGE-L"],
        "Match": ["BLEU", "METEOR", "BERTScore"],
        "Distance": ["CosineSim", "EuclideanDist", "ManhattanDist"]
    }
    group_markers = {"ROUGE": "o", "Match": "s", "Distance": "^"}


    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]


    metric_matrix = defaultdict(lambda: defaultdict(list))  # {metric: {model_id: [values]}}
    for group in raw_data:
        model_id = group['model']
        for item in group["metrics"]:
            for metric, value in item["pair_metrics"].items():

                if metric == "ManhattanDist":
                    value /= 100.0
                elif metric =="EuclideanDist":
                    value /= 10.0
                elif metric =="AvgSentenceLenDiff":
                    value /= 30.0

                if metric =="AvgSentenceLenDiff" or metric == "SentenceCountDiff" or metric=="TokenOverlap" or metric=="POSOverlap":
                    continue
                metric_matrix[metric][model_id].append(value)

    metric_avg = {
        metric: {mid: np.mean(vals) for mid, vals in models.items()}
        for metric, models in metric_matrix.items()
    }


    all_model_ids = sorted(
        {mid for m in metric_avg.values() for mid in m.keys()},
        key=natural_key
    )
    x = np.arange(len(all_model_ids))
    formatted_labels = [mid.replace("t5-small_group_", "Small")
                            .replace("t5-base_group_", "Base")
                            .replace("t5-large_group_", "Large")
                            .replace("t5-base_random_", "R")
                            .replace("t5-base_reorder_", "r")
                            .replace("t5-base_tau_", "b")
                        for mid in all_model_ids]


    plt.figure(figsize=(18, 9))
    cmap = get_cmap("tab20")
    metric_list = sorted(metric_avg.keys())
    color_map = {m: cmap(i % 20) for i, m in enumerate(metric_list)}


    ref_model_full = "t5-base_group_5"
    if ref_model_full in all_model_ids:
        idx = all_model_ids.index(ref_model_full)

        plt.axvline(x=idx, linestyle='--', linewidth=1, color='gray', alpha=0.7, label="B5 position")


    for metric in metric_list:

        group_name = next((g for g, mlist in metric_groups.items() if metric in mlist), "Other")
        marker = group_markers.get(group_name, "x")

        y = [metric_avg[metric].get(mid, np.nan) for mid in all_model_ids]
        label = metric + (" (÷100)" if metric == "ManhattanDist" else "")
        label = metric + (" (÷10)" if metric == "EuclideanDist" else "")
        label = metric + (" (÷30)" if metric == "AvgSentenceLenDiff" else "")
        plt.plot(
            x, y,
            marker=marker,
            linewidth=1.5,
            markersize=5,
            label=label,
            color=color_map[metric]
        )

    plt.title("Model Comparison on Pair Metrics (All-in-One)")
    plt.xlabel("Model ID")
    plt.ylabel("Metric Value")
    plt.xticks(ticks=x, labels=formatted_labels, rotation=0, ha='center')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.5)


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "pair_metrics_all_in_one.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"✅ 图已保存至：{save_path}")


if __name__ == "__main__":
    plot_model_single_metrics()
    plot_model_pair_metrics()