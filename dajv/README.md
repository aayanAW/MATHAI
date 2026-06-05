# verifyensemble: Executable Spec-Grounded LLM-Jury Verification with Cross-Modality Calibration

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Status: Pre-release](https://img.shields.io/badge/Status-Pre--release%20v0.1-orange)]()

## What this is

`verifyensemble` is a calibrated jury-aggregation library for
LLM-emitted Python verifiers on math reasoning. Each LLM extractor
emits a deterministic Python `verify(answer) -> bool` function; the
library aggregates their executable verdicts (and the LLMs'
self-classifications of those verifiers) into a calibrated
$P(\text{correct})$ with a Clopper--Pearson coverage band.

The empirical headline finding is that the **script-execution
modality** (Python verdict) is essentially independent of the
**script-writing modality** (LLM self-classification) across all
$42$ cross-LLM pairs in a $7$-extractor frontier ensemble
(cross-modality $\kappa \approx 0.02$--$0.06$ vs within-modality
$\kappa \approx 0.70$). Pure LLM-jury aggregators in the concurrent
literature cannot exploit this axis by construction.

This is the reference implementation for the paper:

> "Two Modalities Are Better Than One: Cross-Modality Independence
> in Executable Spec-Grounded LLM-Jury Verification for Math
> Reasoning."  Anonymous (2027).
> [Paper PDF](paper/main.pdf).
> [Pre-registration v2](PRE_REGISTRATION_v2.md).

## Why you might use it

Common LLM-jury setups (majority vote, unanimous consensus, learned
PRMs) implicitly assume the verifiers fail independently. Recent
literature (Kim 2025, Kuai 2026, CARE 2026) establishes this is
empirically false: frontier LLMs share substantial training data and
exhibit correlated errors. `verifyensemble` measures the residual
correlation on your calibration set and provides a calibrated
aggregation rule that is robust to it.

## Headline empirical findings (this repo)

**Cross-modality independence (key new result).** On the 7-extractor
cache, within-modality executable κ has median **0.72** across 21 LLM
pairs; cross-LLM cross-modality κ is **0.02–0.06** across 42 pairs.
The executable-grounding modality is an axis of independence that
pure LLM-jury aggregation rules (Kuai 2026, CARE 2026, Balasubramanian
Ising 2026) cannot exploit by construction. See §Cross-modality
independence in the paper.

**H7' attempts: all negative.** Four pre-registered aggregators that
attempt to convert cross-modality independence into a calibrated gain
(factorized Bayes, agreement gating, per-LLM joint Naive-Bayes,
structural-gated exec-DAJV) all fail to improve over default DAJV at
the default operating threshold. Independence is necessary but not
sufficient. See Appendix B.10.

**VERGE head-to-head (faithful Z3 replication).** DAJV math175 (cov
0.62 / prec 0.97) Pareto-dominates VERGE-Z3 (cov 0.49 / prec 0.96)
and VERGE-Z3-MCS (cov 0.55 / prec 0.97). See Appendix B.13.

**4-extractor calibration result.** On the 4-extractor baseline
ensemble (gpt-oss-120B, gpt-5-mini, claude-sonnet-4-6, Qwen3-Coder-480B)
evaluated on $n=330$ math-reasoning problems. Values shown as **mean
$\pm$ std across 10 calibration/test splits**:

| Quantity | Value |
|---|---|
| Median pairwise Cohen's $\kappa$ on wrong subset | **0.797** |
| Median joint-FP / independence-bound ratio | **15.7$\times$** |
| DAJV precision @ commit | $0.985 \pm 0.022$ |
| DAJV coverage | $0.579 \pm 0.065$ |
| DAJV ECE | $0.120 \pm 0.051$ |
| Naive unanimous precision | $0.980 \pm 0.028$ |
| Naive unanimous coverage | $0.460 \pm 0.061$ |
| Naive unanimous ECE | $0.215 \pm 0.037$ |
| **ECE reduction (DAJV vs naive)** | **44%** |
| Theorem 1 Monte Carlo violations | **0 / 45** |
| Pre-registration H4 (cross-benchmark transfer) | **FAIL** → Branch B |

## Quick start

### Install (dev mode)

```bash
git clone <repo-url>
cd dajv
pip install -e .
```

Optional extras for API extractors:

```bash
pip install together openai anthropic google-genai
```

### Run a calibrated jury verify (cached data, no API needed)

```python
import json
from pathlib import Path
from verifyensemble.utils.io import align_extractor_caches
from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_unanimous

# Load 4 cached extractor caches from prior X-SGRV work
caches = {
    "E05_gpt_oss": "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt5":    "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude":  "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3":   "../MATHAI/results/exp48_qwen3coder_extractor.json",
}
aligned = align_extractor_caches(caches, bench="math175")

# 70/30 split
import random; random.Random(42)
idx = list(range(len(aligned["problem_ids"])))
random.Random(42).shuffle(idx)
cal_n = int(0.7 * len(idx))
cal_idx, test_idx = idx[:cal_n], idx[cal_n:]

# Fit DAJV calibration
accept = aligned["accept"]; correct = aligned["solver_correct"]
accept_cal = [[accept[i][j] for j in cal_idx] for i in range(4)]
correct_cal = [correct[j] for j in cal_idx]
calibration = DajvCalibration.fit(accept_cal, correct_cal, aligned["extractor_ids"])

# Apply to a test problem
votes = [accept[i][test_idx[0]] for i in range(4)]
out = dajv_aggregate(votes, calibration)
print(out)
# -> {'P_correct': 0.97..., 'lower': 0.92..., 'upper': 1.00, 'recommendation': 'COMMIT', ...}
```

### Run on your own extractors (needs API keys)

```python
import os
os.environ["TOGETHER_API_KEY"] = "..."
os.environ["ANTHROPIC_API_KEY"] = "..."

from verifyensemble.extractors.api_wrappers import make_extractor_call
from verifyensemble.extractors.parser import extract_verifier
from verifyensemble.sandbox.executor import execute_verifier
from verifyensemble.sandbox.adversarial import deployment_time_filter

extractor = make_extractor_call("E07_claude_sonnet_4_6")
ext_result = extract_verifier("Find the sum of all integer roots of x^2 - 5x + 6 = 0", extractor)

if not ext_result.unverifiable:
    candidate = "5"
    broken, _ = deployment_time_filter(ext_result.script, candidate)
    if broken:
        print("verifier rejected by adversarial filter; abstain")
    else:
        result = execute_verifier(ext_result.script, candidate)
        print(result.verdict)   # True / False / None
```

## Reproducing the paper

```bash
# 1. Compute dependency matrix on cached extractor outputs
PYTHONPATH=. python3 scripts/run_dependency_mapping.py

# 2. Validate Theorem 1 via Monte Carlo
PYTHONPATH=. python3 scripts/run_synthetic_validation.py

# 3. Compare aggregation methods (naive vs DAJV vs CARE)
PYTHONPATH=. python3 scripts/run_aggregation_comparison.py

# 4. Compile paper
cd paper && tectonic main.tex
```

All scripts write outputs to `artifacts/`. Reproducibility:
fixed seed = 42 everywhere.

## Repository structure

```
dajv/
├── verifyensemble/             core library
│   ├── sandbox/                subprocess executor + adversarial filter
│   ├── extractors/             prompt, parser, API wrappers
│   ├── dependency/             kappa, joint-FP, CIG, matrix builder
│   ├── aggregate/              naive, DAJV copula, CARE, Clopper-Pearson
│   ├── theory/                 Theorem 1 + 2 numerical implementations
│   ├── evaluation/             risk-coverage, ECE, Brier, McNemar
│   └── utils/                  cache loaders, I/O helpers
├── scripts/                    end-to-end experiments
├── tests/                      pytest test suite
├── paper/                      LaTeX source + .bib + sections
├── artifacts/                  generated outputs (JSONs, figures)
├── PRE_REGISTRATION_v1.md      pre-committed hypotheses + branches
└── README.md                   this file
```

## Citation

```bibtex
@inproceedings{anonymous2027dajv,
  title={Calibrated LLM-Jury Verification: A Dependency-Aware
         Aggregation Framework for Selective Prediction on Math Reasoning},
  author={Anonymous},
  booktitle={ICML},
  year={2027},
  note={under review}
}
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).

## Status

Pre-release v0.1. Tests are minimal (synthetic-data validation only).
Real LLM API integration tested on 4 frontier extractors via the
cached X-SGRV outputs; head-to-head testing on fresh API calls is
gated on API budget approval.

## Acknowledgements

The sandbox harness, frozen extractor prompt, and adversarial probe set
are directly ported from the X-SGRV project (anonymous, prior work
cited in the paper). The empirical baseline on math175, AIME 2025, and
CleanMath uses the cached extractor outputs from that project.
