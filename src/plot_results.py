"""
Generate evaluation comparison bar chart for the report.
Usage:
    python src/plot_results.py
Output:
    data/eval_comparison.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = {
    "Semantic RAG\n(proposed)": "data/eval_results.json",
    "Fixed-size RAG\n(baseline)": "data/eval_results_fixed500.json",
    "No-retrieval\n(baseline)": "data/eval_results_no_rag.json",
}

METRICS = {
    "correctness":      "Correctness",
    "faithfulness":     "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "source_hit_rate":  "Source Hit Rate",
}

COLORS = ["#2c6e49", "#e76f51", "#a8b0c0"]


def load_summaries(results):
    summaries = {}
    for label, path in results.items():
        with open(path) as f:
            summaries[label] = json.load(f)["summary"]
    return summaries


def plot(summaries, output_path="data/eval_comparison.png"):
    configs = list(summaries.keys())
    metric_keys = list(METRICS.keys())
    metric_labels = list(METRICS.values())

    n_metrics = len(metric_keys)
    n_configs = len(configs)
    x = np.arange(n_metrics)
    width = 0.22
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (config, color) in enumerate(zip(configs, COLORS)):
        values = [summaries[config][k] for k in metric_keys]
        bars = ax.bar(x + offsets[i], values, width, label=config,
                      color=color, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=7.5, color="#222222"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Score (0–1)", fontsize=10)
    ax.set_title("Evaluation Results by Configuration", fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc",
              loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    summaries = load_summaries(RESULTS)
    plot(summaries)
