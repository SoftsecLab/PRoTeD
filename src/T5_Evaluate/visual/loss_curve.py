#python -m scripts.visual.loss_curve
import json
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
from matplotlib.ticker import MultipleLocator
def find_trainer_state(model_dir):
    checkpoint_dirs = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    print(model_dir)
    print(checkpoint_dirs)
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort()
    trainer_file = os.path.join(checkpoint_dirs[-1], "trainer_state.json")
    return trainer_file if os.path.exists(trainer_file) else None

def load_log_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    log_history = data.get("log_history", [])
    epochs, steps, losses = [], [], []
    for record in log_history:
        if "loss" in record and "epoch" in record and "step" in record:
            epochs.append(record["epoch"])
            steps.append(record["step"])
            losses.append(record["loss"])
    return epochs, steps, losses


def plot_loss_all(base_path, save_path=None):
    models = {
        "T5_small": "t5-small_group_{}",
        "T5_base1": "t5-base_group_{}",
        "T5_large": "t5-large_group_{}",
    }
    group_ids = list(range(10))
    rows, cols = 3, 4

    def draw_plot(fig_title, x_type="epoch", filename="output.svg", baseline_points=5):
        fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= len(group_ids):
                fig.delaxes(ax)
                continue

            for model_name, model_pattern in models.items():
                model_dir = os.path.join(base_path, model_name, model_pattern.format(i))
                trainer_file = find_trainer_state(model_dir)
                print(trainer_file)
                # input("Press")
                if not trainer_file:
                    continue
                epochs, steps, losses = load_log_data(trainer_file)
                x_data = epochs if x_type == "epoch" else steps
                ax.plot(x_data, losses, label=model_name, marker='o', linewidth=1.5, markersize=3)
                if len(losses) >= baseline_points:
                    baseline = np.mean(losses[-baseline_points:])
                    ax.axhline(y=baseline, linestyle='--', color='gray', linewidth=1.2, label=f"{model_name} baseline ({baseline:.3f})")
                    ax.text(x_data[-1], baseline, f"{baseline:.3f}", fontsize=8, color='gray',
                            verticalalignment='bottom', horizontalalignment='right',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6))

            if i in [0, 5, 2]:
                large_path = os.path.join(base_path, "T5_large", f"trainer_state{i}.json")
                if os.path.exists(large_path):
                    epochs, steps, losses = load_log_data(large_path)
                    x_data = epochs if x_type == "epoch" else steps
                    ax.plot(x_data, losses, label="T5_large", marker='s', linewidth=1.5, markersize=3)
                    if len(losses) >= baseline_points:
                        baseline = np.mean(losses[-baseline_points:])
                        ax.axhline(y=baseline, linestyle='--', color='black', linewidth=1.2, label=f"T5_large baseline ({baseline:.3f})")
                        ax.text(x_data[-1], baseline, f"{baseline:.3f}", fontsize=8, color='black',
                                verticalalignment='bottom', horizontalalignment='right',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6))

            ax.set_title(f"Group {i}", fontsize=11)
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.legend(fontsize=8)
            ax.set_xlabel("Epoch" if x_type == "epoch" else "Step")
            ax.set_ylabel("Loss")


            if save_path:
                single_dir = os.path.join(save_path, "group_plots")
                os.makedirs(single_dir, exist_ok=True)
                fig_single = plt.figure(figsize=(6, 4))
                ax_s = fig_single.add_subplot(111)
                for line in ax.get_lines():
                    ax_s.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(),
                              marker=line.get_marker(), linewidth=1.5, markersize=3)
                ax_s.set_title(f"Group {i}")
                ax_s.set_xlabel(ax.get_xlabel())
                ax_s.set_ylabel(ax.get_ylabel())
                ax_s.grid(True, linestyle='--', linewidth=0.5)
                ax_s.legend(fontsize=8)
                fig_single.savefig(os.path.join(single_dir, f"group_{i}.svg"), format='svg')
                plt.close(fig_single)

        fig.suptitle(fig_title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            svg_path = os.path.join(save_path, filename)
            fig.savefig(svg_path, format='svg')
            print(f"[✓] 总图已保存: {svg_path}")
        plt.close(fig)

    draw_plot("Loss vs Epoch with Baseline", x_type="epoch", filename="loss_epoch_with_baseline.svg")
    #draw_plot("Loss vs Step with Baseline", x_type="step", filename="loss_step_with_baseline.svg")

def plot_loss_all_broken_and_zoomed(base_path, save_path=None):
    models = {
        "T5_small": "t5-small_group_{}",
        "T5_base1": "t5-base_group_{}",
        "T5_large": "t5-large_group_{}",
    }
    group_ids = list(range(10))
    rows, cols = 3, 4

    def draw_plot(fig_title, x_type="epoch", filename="output_zoomed.svg", baseline_points=5):
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        axes = axes.flatten()
        fig.subplots_adjust(left=0.03, right=0.99, top=0.94, bottom=0.06, wspace=0.15, hspace=0.25)

        for i, ax in enumerate(axes):
            if i >= len(group_ids):
                fig.delaxes(ax)
                continue

            for model_name, model_pattern in models.items():
                model_dir = os.path.join(base_path, model_name, model_pattern.format(i))
                trainer_file = find_trainer_state(model_dir)
                if not trainer_file:
                    continue

                epochs, steps, losses = load_log_data(trainer_file)
                x_data = epochs if x_type == "epoch" else steps
                ax.plot(x_data, losses, label=model_name, marker='o', linewidth=1.5, markersize=3)
                if len(losses) >= baseline_points:
                    baseline = np.mean(losses[-baseline_points:])
                    ax.axhline(y=baseline, linestyle='--', color='gray', linewidth=1.5, label=f"{model_name} baseline ({baseline:.3f})")
                    ax.text(x_data[-1], baseline, f"{baseline:.3f}", fontsize=8, color='gray',
                            verticalalignment='bottom', horizontalalignment='right',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6))

            ax.set_ylim(0.1, 0.3)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
            ax.set_xlabel("Epoch" if x_type == "epoch" else "Step")
            ax.set_title(f"Group {i}", fontsize=11)
            if i == 0:
                ax.legend(fontsize=8)


            if save_path:
                zoom_dir = os.path.join(save_path, "group_plots_zoomed")
                os.makedirs(zoom_dir, exist_ok=True)
                fig_single = plt.figure(figsize=(6, 4))
                ax_s = fig_single.add_subplot(111)
                for line in ax.get_lines():
                    ax_s.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(),
                              marker=line.get_marker(), linewidth=1.5, markersize=3)
                ax_s.set_ylim(0.1, 0.3)
                ax_s.set_title(f"Group {i}")
                ax_s.set_xlabel(ax.get_xlabel())
                ax_s.set_ylabel("Loss")
                ax_s.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                ax_s.legend(fontsize=8)
                fig_single.savefig(os.path.join(zoom_dir, f"group_{i}.svg"), format='svg')
                plt.close(fig_single)

        fig.suptitle(fig_title, fontsize=16)
        if save_path:
            svg_path = os.path.join(save_path, filename)
            fig.savefig(svg_path, format='svg')
            print(f"[✓] Zoom 总图已保存: {svg_path}")
        plt.close(fig)

    draw_plot("Loss vs Epoch (Zoomed Y ∈ [0.1, 0.3])", x_type="epoch", filename="loss_epoch_y_0.1_0.3.svg")
    #draw_plot("Loss vs Step (Zoomed Y ∈ [0.1, 0.3])", x_type="step", filename="loss_step_y_0.1_0.3.svg")


if __name__ == "__main__":

    base_path = "models"
    save_path = "data/T5_evaluate/loss"
    os.makedirs(save_path, exist_ok=True)
    plot_loss_all(base_path, save_path)
    plot_loss_all_broken_and_zoomed(base_path, save_path)
