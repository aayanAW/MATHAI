"""Full 2k x 2k pairwise dependency heatmap.

Renders a 12 x 12 (k=6 LLMs x 2 modalities) Cohen's kappa heatmap
showing the block-diagonal structure of within-modality dependence
contrasted with the off-block-diagonal independence (cross-modality).

Output: paper/figures/fig_cross_modality_heatmap.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.dependency.kappa import cohen_kappa
from verifyensemble.utils.io import align_extractor_caches

GROUP = {
    "E05": HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06": HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09": HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
    "E08S": HERE.parent / "../MATHAI/results/exp55_gpt4o_extractor.json",
    "E04S": HERE.parent / "../MATHAI/results/exp56_gpt41_extractor.json",
}
# Auto-extend with gpt-5 if complete
_GPT5 = HERE.parent / "../MATHAI/results/exp57_gpt5_extractor.json"
if _GPT5.exists():
    try:
        import json as _j
        if len(_j.load(open(_GPT5)).get("results", [])) >= 300:
            GROUP["E10S"] = _GPT5
    except Exception:
        pass
# Auto-extend with Anthropic Opus 4.7 if complete
_OPUS47 = HERE.parent / "../MATHAI/results/exp58_opus47_extractor.json"
if _OPUS47.exists():
    try:
        import json as _j
        if len(_j.load(open(_OPUS47)).get("results", [])) >= 300:
            GROUP["E01A"] = _OPUS47
    except Exception:
        pass
_OPUS46 = HERE.parent / "../MATHAI/results/exp59_opus46_extractor.json"
if _OPUS46.exists():
    try:
        import json as _j
        if len(_j.load(open(_OPUS46)).get("results", [])) >= 300:
            GROUP["E02A"] = _OPUS46
    except Exception:
        pass
_HAIKU45 = HERE.parent / "../MATHAI/results/exp61_haiku45_extractor.json"
if _HAIKU45.exists():
    try:
        import json as _j
        if len(_j.load(open(_HAIKU45)).get("results", [])) >= 300:
            GROUP["E03A"] = _HAIKU45
    except Exception:
        pass
_LLAMA33 = HERE.parent / "../MATHAI/results/exp62_llama33_70b_extractor.json"
if _LLAMA33.exists():
    try:
        import json as _j
        if len(_j.load(open(_LLAMA33)).get("results", [])) >= 300:
            GROUP["E13"] = _LLAMA33
    except Exception:
        pass
_QWEN235 = HERE.parent / "../MATHAI/results/exp63_qwen3_235b_extractor.json"
if _QWEN235.exists():
    try:
        import json as _j
        if len(_j.load(open(_QWEN235)).get("results", [])) >= 300:
            GROUP["E14"] = _QWEN235
    except Exception:
        pass

FIG_DIR = HERE.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    aligned = align_extractor_caches(GROUP, bench=None)
    k = len(aligned["extractor_ids"])
    wrong_idx = [j for j, c in enumerate(aligned["solver_correct"]) if not c]

    struct = [[(aligned["classification"][i][j] == "working")
               for j in wrong_idx] for i in range(k)]
    exec_ = [[(aligned["candidate_verdict"][i][j] is True)
              for j in wrong_idx] for i in range(k)]

    K = 2 * k
    signals = struct + exec_
    labels = [f"S:{e}" for e in aligned["extractor_ids"]] + \
             [f"X:{e}" for e in aligned["extractor_ids"]]
    M = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if i == j:
                M[i][j] = 1.0
            else:
                M[i][j] = cohen_kappa(signals[i], signals[j])

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    # Draw the block boundary (between struct and exec halves)
    ax.axhline(k - 0.5, color="black", lw=1.2)
    ax.axvline(k - 0.5, color="black", lw=1.2)

    # Annotate cells
    for i in range(K):
        for j in range(K):
            color = "white" if abs(M[i][j]) > 0.55 else "black"
            ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center",
                    fontsize=4.5, color=color)

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label="Cohen's $\\kappa$")
    ax.set_title("Pairwise $\\kappa$ between modalities ($k=6$ ext., "
                 f"$n_w={len(wrong_idx)}$)\nS = structural (script works),  "
                 "X = executable (script accepts)",
                 fontsize=8)
    fig.tight_layout()
    out_path = FIG_DIR / "fig_cross_modality_heatmap.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
