"""Summary overview figure: all methods at matched coverage on three benchmarks."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("/Users/aayanalwani/MATHAI/MATHAI/paper/figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

methods = [
    "X-SGRV (Llama)",
    "X-SGRV consensus",
    "Qwen-PRM (best)",
    "Skywork-PRM (best)",
    "SE (SymPy)",
    "SE (NLI)",
    "p(True)",
]
math500 = [0.947, 1.000, 0.927, 0.938, 0.902, 0.870, 0.50]
aime = [0.50, 1.00, 0.50, 0.50, 0.00, 0.0625, 0.50]
cleanmath = [1.00, 1.00, 0.50, 1.00, 0.167, 0.085, 0.50]

colors = {
    "X-SGRV (Llama)": "#1f77b4",
    "X-SGRV consensus": "#17becf",
    "Qwen-PRM (best)": "#ff7f0e",
    "Skywork-PRM (best)": "#d62728",
    "SE (SymPy)": "#2ca02c",
    "SE (NLI)": "#98df8a",
    "p(True)": "#7f7f7f",
}

fig, ax = plt.subplots(figsize=(7.0, 3.8))
n_methods = len(methods)
n_bench = 3
x = np.arange(n_bench)
width = 0.11

for i, method in enumerate(methods):
    vals = [math500[i], aime[i], cleanmath[i]]
    offset = (i - (n_methods - 1) / 2) * width
    ax.bar(x + offset, vals, width, label=method, color=colors[method],
           edgecolor="black", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels([
    "MATH-500\n(n=498, contamination-\nassisted)",
    "AIME 2025\n(n=30, clean)",
    "CleanMath\n(n=125, clean)"
])
ax.set_ylabel("Top-tier precision at matched coverage")
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)
ax.grid(True, axis="y", alpha=0.3)
ax.set_title("Matched-coverage precision across methods and benchmarks")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), framealpha=0.9)
fig.savefig(FIG_DIR / "fig_method_summary.pdf")
plt.close(fig)
print(f"wrote {FIG_DIR/'fig_method_summary.pdf'}")
