import os, json
import matplotlib.pyplot as plt
from transformers import TrainerCallback

def plot_and_save_loss(logs, save_path_svg):

    if not logs:
        print("[loss] 没有可绘制的 loss 日志")
        return
    steps = [x.get("step", i) for i, x in enumerate(logs)]
    losses = [x["loss"] for x in logs if "loss" in x]
    os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(steps[:len(losses)], losses)
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(save_path_svg, format="svg")
    plt.close()

def save_loss_json(logs, save_path_json):

    os.makedirs(os.path.dirname(save_path_json), exist_ok=True)
    with open(save_path_json, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"[loss] loss_log 已保存 -> {save_path_json}")


class LossLogger(TrainerCallback):
    def __init__(self):
        self.loss_log = []

    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs is None:
            return
        if "loss" in logs:
            step = int(state.global_step) if state and state.global_step is not None else len(self.loss_log)
            self.loss_log.append({"step": step, "loss": float(logs["loss"])})
