"""Aggregation subpackage: rules for combining k verifier verdicts.

Each aggregator takes a vote vector ``v in {0, 1, 'abstain'}^k`` and
returns ``(P_correct, lower, upper, recommendation)``.

Aggregators:
    naive_unanimous(v)    -> commit iff all working verifiers accept
    naive_majority(v)     -> commit iff >= k/2 working verifiers accept
    care(v, D)            -> CARE-style confounder-aware (Zhao et al. 2026)
    dajv(v, D, marginals) -> dependency-aware copula posterior

Helpers:
    clopper_pearson(k_success, n) -> (lower, upper)
"""
from verifyensemble.aggregate.care import care_aggregate
from verifyensemble.aggregate.dajv import dajv_aggregate
from verifyensemble.aggregate.naive import naive_majority, naive_unanimous
from verifyensemble.aggregate.posterior import clopper_pearson

__all__ = [
    "naive_unanimous",
    "naive_majority",
    "clopper_pearson",
    "care_aggregate",
    "dajv_aggregate",
]
