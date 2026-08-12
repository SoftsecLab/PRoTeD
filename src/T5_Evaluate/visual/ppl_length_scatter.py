# python -m scripts.visual.ppl_length_scatter
# modules/visual/ppl_length_scatter.py


import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.jsonl_handler import read_jsonl


def extract_metrics(data):
    lengths, ppls, lls = [], [], []
    for item in data:
        if "PPL" in item and "word_count" in item and "LLScore" in item:
            lengths.append(item["word_count"])
            ppls.append(item["PPL"])
            lls.append(item["LLScore"])
    return lengths, ppls, lls

def remove_outliers(lengths, ppls, lls, iqr_scale=1.5):

    ppls_arr = np.array(ppls)
    lengths_arr = np.array(lengths)
    lls_arr = np.array(lls)


    mask_valid = np.isfinite(ppls_arr)
    ppls_arr = ppls_arr[mask_valid]
    lengths_arr = lengths_arr[mask_valid]
    lls_arr = lls_arr[mask_valid]

    print(f"[IQR] 合法样本数: {len(ppls_arr)}")

    q1 = np.percentile(ppls_arr, 25)
    q3 = np.percentile(ppls_arr, 75)
    iqr = q3 - q1
    lower = q1 - iqr_scale * iqr
    upper = q3 + iqr_scale * iqr

    print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}, Lower: {lower}, Upper: {upper}")

    mask = (ppls_arr >= lower) & (ppls_arr <= upper)

    print(f"原始样本: {len(lengths)}, 清洗后: {len(ppls_arr)}, 保留: {mask.sum()} 条")

    return (
        lengths_arr[mask].astype(int).tolist(),
        ppls_arr[mask].tolist(),
        lls_arr[mask].tolist()
    )

def plot_ppl_summary(input_path, output_path, filter_outliers=False):
    data = read_jsonl(input_path)
    lengths, ppls, lls = extract_metrics(data)
    print(f"PPL长度: {len(ppls)}, 示例: {ppls[:5]}")

    if filter_outliers:
        lengths, ppls, lls = remove_outliers(lengths, ppls, lls, iqr_scale=1.5)

    if not lengths:
        print("未找到包含 PPL、LLScore 和 word_count 的有效句子。")
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    title_prefix = "Filtered " if filter_outliers else ""
    fig.suptitle(f"{title_prefix}Sentence Statistics Summary", fontsize=16)


    axs[0, 0].scatter(lengths, ppls, alpha=0.5, edgecolors='k', s=40)
    axs[0, 0].set_title("Sentence Length vs PPL")
    axs[0, 0].set_xlabel("Word Count")
    axs[0, 0].set_ylabel("PPL")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)


    try:
        clean_data = [(l, p) for l, p in zip(lengths, ppls) if np.isfinite(l) and np.isfinite(p)]
        if len(clean_data) >= 2:
            x_vals, y_vals = zip(*clean_data)
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            axs[0, 0].plot(sorted(x_vals), p(sorted(x_vals)), color='red', label='Trend')
            axs[0, 0].legend()
    except Exception as e:
        print(f"⚠️ 趋势线绘制失败: {e}")
    ppls_cleaned = [ppl for ppl in ppls if not np.isnan(ppl)]
    ppls=[]
    ppls=ppls_cleaned

    axs[0, 1].hist(ppls, bins=50, alpha=0.7, color='red', edgecolor='black')
    axs[0, 1].axvline(np.mean(ppls), color='blue', linestyle='--', label=f"Mean={np.mean(ppls):.2f}")
    axs[0, 1].set_title("PPL Distribution")
    axs[0, 1].set_xlabel("PPL")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].legend()

    lls_cleaned = [ll for ll in lls if not np.isnan(ll)]
    lls=[]
    lls=lls_cleaned

    axs[1, 0].hist(lls, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axs[1, 0].axvline(np.mean(lls), color='red', linestyle='--', label=f"Mean={np.mean(lls):.2f}")
    axs[1, 0].set_title("LLScore Distribution")
    axs[1, 0].set_xlabel("LLScore")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    axs[1, 0].legend()


    axs[1, 1].hist(lengths, bins=range(int(min(lengths)), int(max(lengths)) + 1), alpha=0.7, color='green', edgecolor='black')
    axs[1, 1].axvline(np.mean(lengths), color='orange', linestyle='--', label=f"Mean={np.mean(lengths):.2f}")
    axs[1, 1].set_title("Sentence Length Distribution")
    axs[1, 1].set_xlabel("Word Count")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path,format='svg')
    print(f"图像已保存至 {output_dir}")


if __name__ == "__main__":
    input_dir = "data/processed"
    output_dir = "data/visual/split"
    plot_ppl_summary(
        input_path=f"{input_dir}/ieee-init-split.jsonl",
        output_path=f"{output_dir}/ppl_summary_human.svg"
    )
    plot_ppl_summary(
        input_path=f"{input_dir}/ieee-chatgpt-generation-split.jsonl",
        output_path=f"{output_dir}/ppl_summary_gpt.svg"
    )

    plot_ppl_summary(
        input_path=f"{input_dir}/ieee-init-split.jsonl",
        output_path=f"{output_dir}/ppl_summary_human_filtered.svg",
        filter_outliers=True
    )
    plot_ppl_summary(
        input_path=f"{input_dir}/ieee-chatgpt-generation-split.jsonl",
        output_path=f"{output_dir}/ppl_summary_gpt_filtered.svg",
        filter_outliers=True
    )
    plot_ppl_summary(
        input_path=f"{input_dir}/ieee-merged.jsonl",
        output_path=f"{output_dir}/ppl_summary_merged.svg",
        # filter_outliers=True
    )
    plot_ppl_summary(
        input_path=f"{input_dir}/ieee-merged-balanced.jsonl",
        output_path=f"{output_dir}/ppl_summary_merged-balanced.svg",
        # filter_outliers=True
    )
    """ input_dir = "/home/jxy/Data/ReoraganizationData/init/split"
    plot_ppl_summary(
        input_path=f"{input_dir}/ieee-init-split.jsonl",
        output_path=f"{output_dir}/ppl_summary_human_old.png"
    )
    plot_ppl_summary(
        input_path=f"{input_dir}/ieee-chatgpt-generation-split.jsonl",
        output_path=f"{output_dir}/ppl_summary_gpt_old.png"
    ) """