"""DependencyMatrix: estimates the pairwise dependency tensor across k verifiers.

Stores three k x k matrices:
    kappa[i,j]      Cohen's kappa on the wrong-candidate subset
    joint_fp[i,j]   empirical P(both accept | wrong)
    cig[i,j]        Cumulative Information Gain on the wrong-candidate subset

Plus per-verifier marginal FP rates pi[i] = P(i accepts | wrong) and the
matrix of independence-bound predictions pi[i] * pi[j].

Bootstrap CIs are produced by resampling problems with replacement.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from verifyensemble.dependency.cig import cig as cig_metric
from verifyensemble.dependency.joint_fp import joint_fp_rate
from verifyensemble.dependency.kappa import cohen_kappa


@dataclass
class DependencyMatrix:
    extractor_ids: list[str]
    n_problems: int
    n_wrong: int
    pi: list[float]                # k
    kappa: list[list[float]]        # k x k
    joint_fp: list[list[float]]     # k x k
    indep_bound: list[list[float]]  # k x k
    cig: list[list[float]]          # k x k
    ratio: list[list[float]]        # joint_fp / indep_bound

    @classmethod
    def from_accept(
        cls,
        accept: Sequence[Sequence[bool]],
        problem_correct: Sequence[bool],
        extractor_ids: list[str],
    ) -> "DependencyMatrix":
        """Build a DependencyMatrix from a k x n acceptance tensor.

        Args:
            accept:           shape (k, n_problems), boolean
            problem_correct:  length n_problems, True iff gold matches candidate
            extractor_ids:    length k
        """
        k = len(accept)
        if k != len(extractor_ids):
            raise ValueError(
                f"accept has {k} rows, extractor_ids has {len(extractor_ids)}"
            )
        n = len(problem_correct)
        for i in range(k):
            if len(accept[i]) != n:
                raise ValueError(
                    f"accept[{i}] has length {len(accept[i])}, "
                    f"expected {n} (== len(problem_correct))"
                )
        wrong = [j for j, c in enumerate(problem_correct) if not c]
        n_wrong = len(wrong)

        pi = []
        for i in range(k):
            if n_wrong == 0:
                pi.append(0.0)
            else:
                pi.append(sum(1 for j in wrong if accept[i][j]) / n_wrong)

        kappa_m = [[0.0] * k for _ in range(k)]
        jfp_m = [[0.0] * k for _ in range(k)]
        ind_m = [[0.0] * k for _ in range(k)]
        cig_m = [[0.0] * k for _ in range(k)]
        ratio_m = [[1.0] * k for _ in range(k)]

        for i in range(k):
            for j in range(k):
                if i == j:
                    kappa_m[i][j] = 1.0
                    jfp_m[i][j] = pi[i]
                    ind_m[i][j] = pi[i] * pi[i]
                    cig_m[i][j] = 0.0
                    ratio_m[i][j] = 1.0 if pi[i] == 0 else 1.0 / pi[i]
                    continue
                a = accept[i]
                b = accept[j]
                kappa_m[i][j] = cohen_kappa(
                    [a[t] for t in wrong],
                    [b[t] for t in wrong],
                )
                stats = joint_fp_rate(a, b, problem_correct)
                jfp_m[i][j] = stats["joint_observed"]
                ind_m[i][j] = stats["indep_bound"]
                ratio_m[i][j] = stats["ratio"] if stats["ratio"] != float("inf") else 1e9
                cig_m[i][j] = cig_metric(a, b, problem_correct)

        return cls(
            extractor_ids=list(extractor_ids),
            n_problems=n,
            n_wrong=n_wrong,
            pi=pi,
            kappa=kappa_m,
            joint_fp=jfp_m,
            indep_bound=ind_m,
            cig=cig_m,
            ratio=ratio_m,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> dict:
        """Return summary statistics across all off-diagonal pairs."""
        k = len(self.extractor_ids)
        offdiag_kappa = []
        offdiag_ratio = []
        offdiag_cig = []
        offdiag_excess = []
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                offdiag_kappa.append(self.kappa[i][j])
                offdiag_ratio.append(self.ratio[i][j])
                offdiag_cig.append(self.cig[i][j])
                offdiag_excess.append(self.joint_fp[i][j] - self.indep_bound[i][j])
        if not offdiag_kappa:
            return {"n_pairs": 0}
        def _stats(xs):
            xs = sorted(xs)
            n = len(xs)
            return {
                "min": xs[0], "max": xs[-1],
                "median": xs[n // 2],
                "mean": sum(xs) / n,
                "q25": xs[max(0, int(0.25 * n) - 1)],
                "q75": xs[min(n - 1, int(0.75 * n))],
            }
        return {
            "n_pairs": len(offdiag_kappa),
            "kappa": _stats(offdiag_kappa),
            "joint_fp_over_indep_ratio": _stats(offdiag_ratio),
            "excess_over_indep": _stats(offdiag_excess),
            "cig": _stats(offdiag_cig),
            "marginals_pi": {
                "min": min(self.pi),
                "max": max(self.pi),
                "mean": sum(self.pi) / len(self.pi) if self.pi else 0.0,
            },
        }


def bootstrap_ci(
    accept: Sequence[Sequence[bool]],
    problem_correct: Sequence[bool],
    extractor_ids: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap CIs on every entry of the dependency matrix.

    Resamples problem indices with replacement. Returns 2.5th and 97.5th
    percentile of each pair's metric.
    """
    rng = random.Random(seed)
    k = len(accept)
    n = len(problem_correct)

    bs_kappa: list[list[list[float]]] = [[[] for _ in range(k)] for _ in range(k)]
    bs_jfp: list[list[list[float]]] = [[[] for _ in range(k)] for _ in range(k)]
    bs_ratio: list[list[list[float]]] = [[[] for _ in range(k)] for _ in range(k)]
    bs_cig: list[list[list[float]]] = [[[] for _ in range(k)] for _ in range(k)]

    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        accept_b = [[accept[i][t] for t in idx] for i in range(k)]
        pc_b = [problem_correct[t] for t in idx]
        m = DependencyMatrix.from_accept(accept_b, pc_b, extractor_ids)
        for i in range(k):
            for j in range(k):
                bs_kappa[i][j].append(m.kappa[i][j])
                bs_jfp[i][j].append(m.joint_fp[i][j])
                bs_ratio[i][j].append(m.ratio[i][j])
                bs_cig[i][j].append(m.cig[i][j])

    def pctile(arr, q):
        a = sorted(arr)
        if not a:
            return None
        return a[int(q * (len(a) - 1))]

    out = {
        "kappa_lo": [[pctile(bs_kappa[i][j], 0.025) for j in range(k)] for i in range(k)],
        "kappa_hi": [[pctile(bs_kappa[i][j], 0.975) for j in range(k)] for i in range(k)],
        "joint_fp_lo": [[pctile(bs_jfp[i][j], 0.025) for j in range(k)] for i in range(k)],
        "joint_fp_hi": [[pctile(bs_jfp[i][j], 0.975) for j in range(k)] for i in range(k)],
        "ratio_lo": [[pctile(bs_ratio[i][j], 0.025) for j in range(k)] for i in range(k)],
        "ratio_hi": [[pctile(bs_ratio[i][j], 0.975) for j in range(k)] for i in range(k)],
        "cig_lo": [[pctile(bs_cig[i][j], 0.025) for j in range(k)] for i in range(k)],
        "cig_hi": [[pctile(bs_cig[i][j], 0.975) for j in range(k)] for i in range(k)],
        "n_bootstrap": n_bootstrap,
    }
    return out
