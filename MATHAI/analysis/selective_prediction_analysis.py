"""Final analysis of selective prediction results (exp25 + exp26).

Produces:
- AUROC (tie-insensitive) with bootstrap CIs
- Tier-based precision-coverage analysis (no tie-breaking needed)
- Expected RC-AUC under random tie-breaking (with std over 50 seeds)
- Risk-coverage curves figure
- Cross-model table

Tie handling: the three methods produce heavily-tied confidence scores
(Verbalized 166/175 at 1.0, SC 118/175 at 1.0, SGRV 59/175 at 1.0).
Reporting a single Acc@k% value is misleading because the top-k may cut
mid-tier. We report (a) AUROC (tie-aware), (b) top-tier accuracy and
coverage (exact), and (c) RC-AUC expected over random tie-breakings.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def ci(k, n):
    if n == 0:
        return 0.0, 0.0
    c = binomtest(k, n).proportion_ci(method="exact")
    return c.low, c.high


def bootstrap_auroc(labels, scores, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            aurocs.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            pass
    aurocs = np.array(aurocs)
    return aurocs.mean(), np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)


def rc_auc_single(labels, scores):
    order = np.argsort(-scores)
    y = labels[order]
    n = len(y)
    cumsum = np.cumsum(y)
    cov = np.arange(1, n + 1) / n
    risk = 1 - cumsum / np.arange(1, n + 1)
    return float(np.trapezoid(risk, cov))


def rc_auc_expected(labels, scores, n_seeds=50):
    """Expected RC-AUC under random tie-breaking (mean ± std over seeds)."""
    vals = []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        jitter = rng.normal(0, 1e-8, size=len(labels))
        vals.append(rc_auc_single(labels, scores + jitter))
    return float(np.mean(vals)), float(np.std(vals))


def tier_analysis(labels, scores):
    """Group by confidence score, report top-tier stats (no tie-breaking needed)."""
    tiers = defaultdict(list)
    for i, s in enumerate(scores):
        tiers[float(s)].append(i)
    sorted_keys = sorted(tiers.keys(), reverse=True)
    out = []
    cum_n = 0
    cum_correct = 0
    for k in sorted_keys:
        idxs = tiers[k]
        tier_correct = int(labels[idxs].sum())
        tier_n = len(idxs)
        cum_n += tier_n
        cum_correct += tier_correct
        lo, hi = ci(cum_correct, cum_n)
        out.append({
            "tier_score": k,
            "tier_n": tier_n,
            "tier_acc": tier_correct / tier_n,
            "cum_n": cum_n,
            "cum_coverage": cum_n / len(labels),
            "cum_acc": cum_correct / cum_n,
            "cum_ci_low": lo,
            "cum_ci_high": hi,
        })
    return out


def top_tier_stats(labels, scores):
    """Accuracy and coverage of the top-confidence tier (exact, tie-free)."""
    t = tier_analysis(labels, scores)
    return t[0]  # top tier


def main():
    exp25_path = RESULTS_DIR / "exp25_selective_prediction.json"
    exp26_path = RESULTS_DIR / "exp26_crossmodel_selective.json"

    print("=" * 70)
    print("SELECTIVE PREDICTION ANALYSIS")
    print("=" * 70)

    primary_metrics = {}
    if exp25_path.exists():
        with open(exp25_path) as f:
            exp25 = json.load(f)

        results = exp25["raw_results"] if "raw_results" in exp25 else exp25
        labels = np.array([r["sample0_correct"] for r in results], dtype=int)
        n = len(labels)
        print(f"\n--- Primary (Qwen2.5-7B-Instruct-Turbo, n={n}) ---")
        print(f"  Baseline accuracy: {labels.mean():.3f}")

        methods = {
            "Self-Consistency": np.array([r["sc_confidence"] for r in results]),
            "Verbalized Confidence": np.array([r["verb_confidence"] for r in results]),
            "SGRV (ours)": np.array([r["sgrv_confidence"] for r in results]),
        }

        print(f"\n  {'Method':<22} {'AUROC [95% CI]':<25} {'E[RC-AUC]':<18} {'Top-tier':<28}")
        print("  " + "-" * 95)
        for name, scores in methods.items():
            mean, lo, hi = bootstrap_auroc(labels, scores)
            rc_mean, rc_std = rc_auc_expected(labels, scores)
            top = top_tier_stats(labels, scores)
            top_str = f"n={top['tier_n']:3d} ({top['cum_coverage']*100:4.1f}%) acc={top['cum_acc']:.3f} [{top['cum_ci_low']:.3f},{top['cum_ci_high']:.3f}]"
            print(f"  {name:<22} {mean:.3f} [{lo:.3f}-{hi:.3f}]     {rc_mean:.3f}±{rc_std:.3f}       {top_str}")
            primary_metrics[name] = {
                "auroc": float(mean),
                "auroc_ci_low": float(lo),
                "auroc_ci_high": float(hi),
                "rc_auc_expected": rc_mean,
                "rc_auc_std": rc_std,
                "top_tier": top,
            }

        # Full tier analysis for each method (for paper table)
        print("\n--- Full tier analysis ---")
        primary_metrics["_tiers"] = {}
        for name, scores in methods.items():
            tiers = tier_analysis(labels, scores)
            primary_metrics["_tiers"][name] = tiers
            print(f"\n  {name}:")
            for t in tiers:
                print(f"    score={t['tier_score']:.2f}  tier_n={t['tier_n']:3d}  "
                      f"cum_cov={t['cum_coverage']*100:5.1f}%  "
                      f"cum_acc={t['cum_acc']:.3f} [{t['cum_ci_low']:.3f},{t['cum_ci_high']:.3f}]")

        # Risk-coverage curve figure (using random tie-broken seed=0 for plotting)
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {"Self-Consistency": "#3498db", "Verbalized Confidence": "#e67e22", "SGRV (ours)": "#2ecc71"}
        for name, scores in methods.items():
            # Average curve over 20 seeds
            n_total = len(labels)
            all_risks = []
            for s in range(20):
                rng = np.random.default_rng(s)
                jitter = rng.normal(0, 1e-8, size=n_total)
                order = np.argsort(-(scores + jitter))
                y_sorted = labels[order]
                cumsum = np.cumsum(y_sorted)
                risks = (1 - cumsum / np.arange(1, n_total + 1)) * 100
                all_risks.append(risks)
            mean_risk = np.mean(all_risks, axis=0)
            coverages = np.arange(1, n_total + 1) / n_total * 100
            ax.plot(coverages, mean_risk, label=name, linewidth=2, color=colors[name])

        ax.set_xlabel("Coverage (%)", fontsize=12)
        ax.set_ylabel("Risk (% incorrect)", fontsize=12)
        ax.set_title("Risk-Coverage Curves: Selective Prediction on MATH-500\n(Qwen2.5-7B-Instruct-Turbo, n=175)", fontsize=13)
        ax.legend(frameon=False, loc="upper left", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 100)
        for fmt in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"fig27_risk_coverage_curves.{fmt}", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"\n  Saved fig27_risk_coverage_curves")

    # Cross-model (exp26)
    cross_model_metrics = {}
    if exp26_path.exists():
        with open(exp26_path) as f:
            exp26 = json.load(f)

        all_results = exp26.get("raw_results", exp26)
        if "summary" in exp26:
            # Already has summary structure
            all_results = exp26.get("raw_results", exp26)
        if isinstance(all_results, dict) and "summary" in all_results:
            all_results = all_results.get("raw_results", {})

        print(f"\n--- Cross-Model Results ---")
        for model_name, results in all_results.items():
            if not isinstance(results, list) or len(results) < 5:
                continue
            short = model_name.split("/")[-1][:30]
            labels = np.array([r["sample0_correct"] for r in results], dtype=int)
            n = len(labels)
            print(f"\n  {short} (n={n}, acc={labels.mean():.2f})")
            model_entry = {"n": n, "accuracy": float(labels.mean()), "methods": {}}
            for key, name in [("sc_confidence", "SC"), ("verb_confidence", "Verb"), ("sgrv_confidence", "SGRV")]:
                if not all(key in r for r in results):
                    continue
                scores = np.array([r[key] for r in results])
                try:
                    auroc = roc_auc_score(labels, scores)
                except ValueError:
                    auroc = 0.5
                rc_mean, rc_std = rc_auc_expected(labels, scores)
                top = top_tier_stats(labels, scores)
                print(f"    {name:<6} AUROC={auroc:.3f}  E[RC-AUC]={rc_mean:.3f}±{rc_std:.3f}  "
                      f"top-tier: n={top['tier_n']} ({top['cum_coverage']*100:.1f}%) "
                      f"acc={top['cum_acc']:.3f}")
                model_entry["methods"][name] = {
                    "auroc": float(auroc),
                    "rc_auc_expected": rc_mean,
                    "rc_auc_std": rc_std,
                    "top_tier": top,
                }
            cross_model_metrics[model_name] = model_entry

    output = {
        "primary": primary_metrics,
        "cross_model": cross_model_metrics,
    }

    with open(RESULTS_DIR / "selective_prediction_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/selective_prediction_analysis.json")


if __name__ == "__main__":
    main()
