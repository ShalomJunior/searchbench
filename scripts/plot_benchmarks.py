import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results/plots", exist_ok=True)

# 1. Latency Plot (Python vs Elastic)
labels = ["BM25 Engine", "Hybrid (RRF)"]
python_latency = [6935.50, 7008.65]
elastic_latency = [24.03, 60.58]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(
    x - width / 2, python_latency, width, label="Pure Python (Dict)", color="#e74c3c"
)
rects2 = ax.bar(
    x + width / 2, elastic_latency, width, label="Elasticsearch (C++)", color="#2ecc71"
)

ax.set_ylabel("Latency (ms) - Log Scale")
ax.set_title("TREC-COVID: Search Latency Bottleneck Resolution")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale("log")
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.0f} ms",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("results/plots/latency_comparison.png", dpi=300)
plt.close()

# 2. Catastrophic Forgetting Plot
datasets = [
    "SciFact\n(In-Domain)",
    "FIQA\n(Out-of-Domain)",
    "TREC-COVID\n(Out-of-Domain)",
    "ArguAna\n(Out-of-Domain)",
]
base_scores = [0.6888, 0.3696, 0.7387, 0.3092]
ft_scores = [0.7303, 0.3399, 0.6959, 0.2780]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(
    x - width / 2, base_scores, width, label="Base Reranker (General)", color="#3498db"
)
rects2 = ax.bar(
    x + width / 2,
    ft_scores,
    width,
    label="Fine-Tuned Reranker (SciFact)",
    color="#9b59b6",
)

ax.set_ylabel("NDCG@10 Score")
ax.set_title("Catastrophic Forgetting: Cross-Encoder Generalization")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc="lower center")


def autolabel_scores(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel_scores(rects1)
autolabel_scores(rects2)

# Draw an arrow showing degradation vs improvement
ax.annotate(
    "Improvement",
    xy=(0.17, 0.73),
    xytext=(0.17, 0.65),
    arrowprops=dict(facecolor="green", shrink=0.05),
    color="green",
    fontweight="bold",
    ha="center",
)
ax.annotate(
    "Degradation",
    xy=(1.17, 0.33),
    xytext=(1.17, 0.25),
    arrowprops=dict(facecolor="red", shrink=0.05),
    color="red",
    fontweight="bold",
    ha="center",
)
ax.annotate(
    "Degradation",
    xy=(2.17, 0.69),
    xytext=(2.17, 0.61),
    arrowprops=dict(facecolor="red", shrink=0.05),
    color="red",
    fontweight="bold",
    ha="center",
)
ax.annotate(
    "Degradation",
    xy=(3.17, 0.27),
    xytext=(3.17, 0.19),
    arrowprops=dict(facecolor="red", shrink=0.05),
    color="red",
    fontweight="bold",
    ha="center",
)

plt.tight_layout()
plt.savefig("results/plots/catastrophic_forgetting.png", dpi=300)
plt.close()

print("Plots generated successfully in results/plots/")
