"""Compute calibration metrics for ExeVer and SGRV on MATH-500.

Metrics:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier score
- AUROC (verdict as score for correctness prediction)
- Precision-recall curve
- Risk-coverage curve area

Framing: treat verification as a confidence calibration problem.
ExeVer and SGRV both provide signals that predict whether a solution is correct.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve

RESULTS_DIR = Path(__file__).parent.parent / "results"


def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(y_true)
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_prob[in_bin].mean()
        bin_weight = in_bin.sum() / n
        ece_val += bin_weight * abs(bin_acc - bin_conf)
    return ece_val


def mce(y_true, y_prob, n_bins=10):
    """Maximum Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    max_err = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_prob[in_bin].mean()
        max_err = max(max_err, abs(bin_acc - bin_conf))
    return max_err


def risk_coverage_auc(y_true, confidence_scores):
    """Area under the risk-coverage curve (lower is better)."""
    # Sort by confidence descending
    order = np.argsort(-confidence_scores)
    y_sorted = y_true[order]
    n = len(y_sorted)

    # At each threshold, compute coverage and risk (1 - accuracy)
    cumsum_correct = np.cumsum(y_sorted)
    coverages = np.arange(1, n + 1) / n
    accuracies = cumsum_correct / np.arange(1, n + 1)
    risks = 1 - accuracies

    # AUC via trapezoidal rule (np.trapezoid in NumPy 2.0+)
    auc = np.trapezoid(risks, coverages)
    return auc


def score_exever(exp5_results):
    """Get ExeVer confidence scores and correctness labels.

    Confidence score: continuous signal derived from verdict
    - ALL_PASS: score = 1.0 (high confidence)
    - REPAIRED: score = 0.8
    - REPAIRED_UNVERIFIED: score = 0.6
    - FAIL_STEP_k: score = 0.3
    - RUNTIME_ERROR / SYNTAX_ERROR / TIMEOUT: score = 0.5 (no signal)
    """
    score_map = {
        "ALL_PASS": 1.0,
        "REPAIRED": 0.8,
        "REPAIRED_UNVERIFIED": 0.6,
        "TIMEOUT": 0.5,
        "RUNTIME_ERROR": 0.5,
        "SYNTAX_ERROR": 0.5,
        "NO_SCRIPT": 0.4,
    }
    scores = []
    labels = []
    for r in exp5_results:
        verdict = r.get("verdict", "")
        if "FAIL_STEP" in verdict:
            scores.append(0.3)
        else:
            scores.append(score_map.get(verdict, 0.5))
        labels.append(int(r["answer_correct"]))
    return np.array(scores), np.array(labels)


def score_sgrv(exp15_results):
    """Get SGRV confidence scores and correctness labels.

    Confidence score: fraction of tested steps that passed.
    - all_tested_pass && n_testable > 0: score = 1.0
    - no testable steps: score = 0.5 (abstain)
    - mixed pass/fail: proportional
    """
    scores = []
    labels = []
    for r in exp15_results:
        n_tested = r.get("n_testable", 0)
        if n_tested == 0:
            scores.append(0.5)  # abstain
        else:
            n_passed = r.get("n_passed", 0)
            # Raw pass fraction, but if all passed give max score
            if r.get("all_tested_pass", False):
                scores.append(1.0)
            else:
                scores.append(n_passed / n_tested * 0.6)  # max 0.6 if some failed
        labels.append(int(r["answer_correct"]))
    return np.array(scores), np.array(labels)


def main():
    print("=" * 60)
    print("CALIBRATION METRICS FOR VERIFICATION METHODS")
    print("=" * 60)

    with open(RESULTS_DIR / "exp5_math500_full.json") as f:
        exp5 = json.load(f)
    with open(RESULTS_DIR / "exp15_pbt_math500.json") as f:
        exp15 = json.load(f)

    # ExeVer scores
    exever_scores, exever_labels = score_exever(exp5["exever_results"])
    print(f"\n--- ExeVer ---")
    print(f"  n = {len(exever_labels)}")
    print(f"  Accuracy (baseline): {exever_labels.mean():.3f}")

    ece_val = ece(exever_labels, exever_scores)
    mce_val = mce(exever_labels, exever_scores)
    brier = brier_score_loss(exever_labels, exever_scores)
    auroc = roc_auc_score(exever_labels, exever_scores)
    rc_auc = risk_coverage_auc(exever_labels, exever_scores)

    print(f"  ECE: {ece_val:.4f}")
    print(f"  MCE: {mce_val:.4f}")
    print(f"  Brier score: {brier:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Risk-coverage AUC: {rc_auc:.4f} (lower = better)")

    exever_metrics = {
        "n": len(exever_labels),
        "accuracy": float(exever_labels.mean()),
        "ece": float(ece_val),
        "mce": float(mce_val),
        "brier": float(brier),
        "auroc": float(auroc),
        "risk_coverage_auc": float(rc_auc),
    }

    # SGRV scores
    sgrv_scores, sgrv_labels = score_sgrv(exp15["detailed_results"])
    print(f"\n--- SGRV ---")
    print(f"  n = {len(sgrv_labels)}")
    print(f"  Accuracy (baseline): {sgrv_labels.mean():.3f}")

    ece_val = ece(sgrv_labels, sgrv_scores)
    mce_val = mce(sgrv_labels, sgrv_scores)
    brier = brier_score_loss(sgrv_labels, sgrv_scores)
    auroc = roc_auc_score(sgrv_labels, sgrv_scores)
    rc_auc = risk_coverage_auc(sgrv_labels, sgrv_scores)

    print(f"  ECE: {ece_val:.4f}")
    print(f"  MCE: {mce_val:.4f}")
    print(f"  Brier score: {brier:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Risk-coverage AUC: {rc_auc:.4f} (lower = better)")

    sgrv_metrics = {
        "n": len(sgrv_labels),
        "accuracy": float(sgrv_labels.mean()),
        "ece": float(ece_val),
        "mce": float(mce_val),
        "brier": float(brier),
        "auroc": float(auroc),
        "risk_coverage_auc": float(rc_auc),
    }

    # Comparison
    print(f"\n--- Comparison ---")
    print(f"{'Metric':<25} {'ExeVer':>10} {'SGRV':>10} {'Delta':>10}")
    print(f"{'ECE':<25} {exever_metrics['ece']:>10.4f} {sgrv_metrics['ece']:>10.4f} "
          f"{sgrv_metrics['ece'] - exever_metrics['ece']:>+10.4f}")
    print(f"{'Brier score':<25} {exever_metrics['brier']:>10.4f} {sgrv_metrics['brier']:>10.4f} "
          f"{sgrv_metrics['brier'] - exever_metrics['brier']:>+10.4f}")
    print(f"{'AUROC':<25} {exever_metrics['auroc']:>10.4f} {sgrv_metrics['auroc']:>10.4f} "
          f"{sgrv_metrics['auroc'] - exever_metrics['auroc']:>+10.4f}")
    print(f"{'Risk-coverage AUC':<25} {exever_metrics['risk_coverage_auc']:>10.4f} "
          f"{sgrv_metrics['risk_coverage_auc']:>10.4f} "
          f"{sgrv_metrics['risk_coverage_auc'] - exever_metrics['risk_coverage_auc']:>+10.4f}")

    # Save
    output = {
        "exever": exever_metrics,
        "sgrv": sgrv_metrics,
        "comparison": {
            "ece_improvement": exever_metrics["ece"] - sgrv_metrics["ece"],
            "brier_improvement": exever_metrics["brier"] - sgrv_metrics["brier"],
            "auroc_improvement": sgrv_metrics["auroc"] - exever_metrics["auroc"],
            "rc_auc_improvement": exever_metrics["risk_coverage_auc"] - sgrv_metrics["risk_coverage_auc"],
        },
    }

    out_path = RESULTS_DIR / "calibration_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
