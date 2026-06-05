"""Monte Carlo validation of Theorem 1 (DAJV joint-FP bound).

Generates synthetic Bernoulli ensembles with known pairwise correlations
via a Gaussian-copula construction, computes the empirical joint-all-1
probability, and compares against:

    independence_lower_bound:  prod(pi)
    dajv_upper_bound:          prod(pi) + sum rho^+ * sqrt(...)
    union_upper_bound:         1 - prod(1 - pi)

For each (k, rho) combination, prints
    empirical, indep_bound, dajv_bound, union_bound
    and verifies empirical <= dajv_bound up to MC noise.

Saves artifacts/theorem1_validation.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.theory import (
    dajv_upper_bound,
    independence_lower_bound,
    required_n,
    union_upper_bound,
)


def _phi(x: float) -> float:
    """Standard normal CDF, via erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def gaussian_copula_bernoulli(
    pi: list[float],
    rho: list[list[float]],
    n_samples: int,
    seed: int = 42,
) -> list[list[bool]]:
    """Generate correlated Bernoulli samples via Gaussian copula.

    Each sample is a length-k binary vector. The marginal of X_i is pi_i,
    and Corr(X_i, X_j) approximates rho[i][j] (the Gaussian-copula
    correlation differs slightly from the Bernoulli correlation, but the
    deviation is small for moderate rho).
    """
    import random as rnd
    k = len(pi)
    rng = rnd.Random(seed)
    thresholds = [_inverse_phi(1 - p) for p in pi]

    # Cholesky factor of covariance matrix
    L = _cholesky([[rho[i][j] if i != j else 1.0 for j in range(k)] for i in range(k)])

    samples = []
    for _ in range(n_samples):
        z = [rng.gauss(0, 1) for _ in range(k)]
        # Apply Cholesky to get correlated Gaussians
        z_cor = [sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(k)]
        sample = [bool(z_cor[i] > thresholds[i]) for i in range(k)]
        samples.append(sample)
    return samples


def _inverse_phi(p: float) -> float:
    """Inverse normal CDF (probit). Bisection."""
    if p <= 0.5:
        return -_inverse_phi(1 - p)
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cholesky(A: list[list[float]]) -> list[list[float]]:
    """Lower-triangular Cholesky factor."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v < 0:
                    v = 1e-9
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def run_validation(
    k_values: list[int] = (2, 5, 12),
    rho_values: list[float] = (0.0, 0.1, 0.2, 0.4, 0.6),
    pi_values: list[float] = (0.05, 0.10, 0.20),
    n_samples: int = 20000,
) -> list[dict]:
    """Sweep (k, rho, pi) and verify Theorem 1 holds empirically."""
    results = []
    for k in k_values:
        for rho_val in rho_values:
            for pi_val in pi_values:
                pi = [pi_val] * k
                rho = [[rho_val if i != j else 1.0 for j in range(k)] for i in range(k)]
                samples = gaussian_copula_bernoulli(pi, rho, n_samples, seed=42)
                empirical = sum(1 for s in samples if all(s)) / n_samples
                indep = independence_lower_bound(pi)
                bound = dajv_upper_bound(pi, rho)
                union = union_upper_bound(pi)
                # Approximate MC error
                mc_err = math.sqrt(empirical * (1 - empirical) / n_samples) * 2
                holds = (empirical <= bound + mc_err)
                results.append({
                    "k": k, "rho": rho_val, "pi": pi_val,
                    "empirical": empirical,
                    "independence_bound": indep,
                    "dajv_bound": bound,
                    "union_bound": union,
                    "mc_2sigma": mc_err,
                    "bound_holds": holds,
                    "gap_below_bound": bound - empirical,
                })
    return results


def main() -> None:
    print("=== Theorem 1 (DAJV joint-FP bound) Monte Carlo validation ===\n")
    print(f"{'k':>3} {'rho':>5} {'pi':>6} {'empirical':>10} {'indep':>10} "
          f"{'DAJV':>10} {'union':>10} {'holds':>6}")
    results = run_validation()
    for r in results:
        flag = "OK" if r["bound_holds"] else "VIOLATE"
        print(f"{r['k']:>3} {r['rho']:>5.2f} {r['pi']:>6.3f} "
              f"{r['empirical']:>10.6f} {r['independence_bound']:>10.6f} "
              f"{r['dajv_bound']:>10.6f} {r['union_bound']:>10.6f} {flag:>6}")
    n_viol = sum(1 for r in results if not r["bound_holds"])
    n_total = len(results)
    print(f"\nTheorem 1 violations: {n_viol} / {n_total} (within 2sigma MC noise)")

    out = HERE.parent / "artifacts" / "theorem1_validation.json"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        json.dump({"results": results, "n_violations": n_viol,
                   "n_total": n_total}, f, indent=2)
    print(f"\nWrote {out}")

    print("\n=== Theorem 2 (sample complexity) ===")
    print("Required n at delta=0.05 for varying (k, eps):")
    for k in [4, 12, 20]:
        for eps in [0.05, 0.10, 0.20]:
            n_req = required_n(k, eps, delta=0.05)
            print(f"  k={k:3d} eps={eps:.2f}  ->  n >= {n_req}")


if __name__ == "__main__":
    main()
