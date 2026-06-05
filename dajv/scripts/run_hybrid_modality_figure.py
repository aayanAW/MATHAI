"""Render the cross-modality independence figure for the paper.

Reads artifacts/hybrid_modality.json and saves
paper/figures/fig_cross_modality.pdf.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ART = HERE.parent / "artifacts" / "hybrid_modality.json"
FIG_DIR = HERE.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    data = json.load(open(ART))
    benches = [r for r in data if isinstance(r, dict) and not r.get("skipped")]
    if not benches:
        print("no benches in artifact")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    bucket_order = [
        ("within_modality_struct", "within-mod\nstruct"),
        ("within_modality_exec",   "within-mod\nexec"),
        ("within_llm_cross_modality", "within-LLM\ncross-mod"),
        ("cross_llm_cross_modality",  "cross-LLM\ncross-mod"),
    ]
    bucket_colors = ["#4c72b0", "#dd8452", "#55a467", "#c44e52"]

    for ax, r in zip(axes, benches):
        positions = list(range(len(bucket_order)))
        all_vals = []
        for (key, _label) in bucket_order:
            vals = []
            for pd in r.get("pair_details", []):
                if pd.get("type") == key:
                    vals.append(pd.get("kappa"))
            all_vals.append([v for v in vals if v is not None])

        bp = ax.boxplot(all_vals, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        medianprops=dict(color="black", linewidth=1.2))
        for patch, color in zip(bp["boxes"], bucket_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels([lbl for _, lbl in bucket_order], fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_ylim(-0.15, 1.05)
        ax.axhline(0.0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"{r['tag']} ($n_w={r.get('n_wrong')}$)", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Cohen's $\\kappa$", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    fig.suptitle("Cross-modality is essentially independent ($\\kappa \\!\\approx\\! 0$)\nwhile within-modality is strongly dependent",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    out_path = FIG_DIR / "fig_cross_modality.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
