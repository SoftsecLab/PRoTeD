#python -m modules.plotter
import numpy as np
import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self):
        pass

    def plot_llscore_ppl(self, results, file_path):

        llscores = [res["LLScore"] for res in results]
        ppls = [res["PPL"] for res in results]

        plt.figure(figsize=(12, 5))


        plt.subplot(1, 2, 1)
        plt.hist(llscores, bins=50, color="blue", alpha=0.7, label="LLScore")
        plt.xlabel("LLScore")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of LLScore ({os.path.basename(file_path)})")
        plt.legend()


        plt.subplot(1, 2, 2)
        plt.hist(ppls, bins=50, color="red", alpha=0.7, label="PPL")
        plt.xlabel("Perplexity (PPL)")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Perplexity (PPL) ({os.path.basename(file_path)})")
        plt.legend()


        image_path = f"data/tmp/{os.path.basename(os.path.splitext(file_path)[0])}_llscore_ppl.svg"
        os.makedirs("results", exist_ok=True)
        plt.savefig(image_path, format='svg')
        print(f"统计图已保存至 {image_path}")
if __name__ == "__main__":

    dummy_results = [
        {"LLScore": np.random.randn(), "PPL": np.random.rand() * 100}
        for _ in range(1000)
    ]
    plotter = Plotter()
    plotter.plot_llscore_ppl(dummy_results, "dummy_file.jsonl")
