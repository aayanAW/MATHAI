"""Render H6 within-lab vs cross-lab dependency figure with CIs.

Reads artifacts/h6_significance.json and saves
paper/figures/fig_h6_within_vs_cross.pdf.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ART = HERE.parent / "artifacts" / "h6_significance.json"
FIG_DIR = HERE.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    data = json.load(open(ART))
    benches = [r for r in data if isinstance(r, dict) and not r.get("skipped")]
    if not benches:
        print("no benches")
        return

    fig, axes = plt.subplots(1, len(benches), figsize=(3.3 * len(benches), 2.8),
                             sharey=True)
    if len(benches) == 1:
        axes = [axes]

    for ax, r in zip(axes, benches):
        wvals = r["within_lab"]["values"]
        cvals = r["cross_lab"]["values"]
        bp = ax.boxplot([wvals, cvals], positions=[0, 1], widths=0.55,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        medianprops=dict(color="black", linewidth=1.2))
        bp["boxes"][0].set_facecolor("#4c72b0")
        bp["boxes"][1].set_facecolor("#dd8452")
        for box in bp["boxes"]:
            box.set_alpha(0.6)

        # Overlay individual points
        for x, vals in zip([0, 1], [wvals, cvals]):
            for v in vals:
                ax.scatter([x + 0.05], [v], color="black", s=8, alpha=0.6,
                           zorder=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"within-lab\n($n={r['within_lab']['n_pairs']}$)",
             f"cross-lab\n($n={r['cross_lab']['n_pairs']}$)"],
            fontsize=7
        )
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="both", labelsize=7)
        pval = r.get("permutation_test", {}).get("p_value")
        obs = r.get("permutation_test", {}).get("observed_diff")
        pstr = f"p={pval:.3f}" if pval is not None else "p=n/a"
        obsstr = f"$\\Delta$={obs:.3f}" if obs is not None else ""
        ax.set_title(f"{r['tag']} (k={r['k_extractors']})\n{obsstr}  {pstr}",
                     fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Cohen's $\\kappa$", fontsize=8)

    fig.suptitle("H6: within-lab vs cross-lab dependency",
                 fontsize=9, y=1.04)
    fig.tight_layout()
    out_path = FIG_DIR / "fig_h6_within_vs_cross.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
