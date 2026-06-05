"""Evaluation subpackage: risk-coverage, ECE, Brier, reliability, bootstrap."""
from verifyensemble.evaluation.bootstrap import bootstrap_metric
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error, reliability_diagram
from verifyensemble.evaluation.mcnemar import mcnemar_mid_p
from verifyensemble.evaluation.risk_coverage import (
    risk_coverage_auc,
    risk_coverage_curve,
)

__all__ = [
    "risk_coverage_curve",
    "risk_coverage_auc",
    "expected_calibration_error",
    "reliability_diagram",
    "brier_score",
    "bootstrap_metric",
    "mcnemar_mid_p",
]
