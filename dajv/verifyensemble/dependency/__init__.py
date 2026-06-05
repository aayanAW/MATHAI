"""Dependency subpackage: estimate pairwise dependency between verifiers.

Public API:
    cohen_kappa(a, b)                            -> float
    joint_fp_rate(a, b, gold_correct)            -> dict
    independence_bound_fp(margin_a, margin_b)    -> float
    cig(a, b, errors)                            -> float
    DependencyMatrix.from_accept(accept_mat,
                                 problem_correct)-> DependencyMatrix
    DependencyMatrix.bootstrap_ci(...)           -> dict
"""
from verifyensemble.dependency.cig import cig
from verifyensemble.dependency.joint_fp import (
    independence_bound_fp,
    joint_fp_rate,
)
from verifyensemble.dependency.kappa import cohen_kappa
from verifyensemble.dependency.matrix import DependencyMatrix

__all__ = [
    "cohen_kappa",
    "joint_fp_rate",
    "independence_bound_fp",
    "cig",
    "DependencyMatrix",
]
