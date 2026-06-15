# MATHAI — LLM Verification & Selective Prediction for Math Reasoning

Research monorepo on a single question: **when an LLM checks its own math
reasoning, why does the check fail, and what can you trust instead?**

It holds two linked research tracks. The first _measures_ the failure
(correlated-verification collapse) and proposes a high-precision signal that
dodges it. The second _calibrates_ an ensemble of such signals into a
deployment-ready selective-prediction rule with valid coverage guarantees.

| Track                       | Directory           | What it is                                                                     | Target venue                       | Status                                  |
| --------------------------- | ------------------- | ------------------------------------------------------------------------------ | ---------------------------------- | --------------------------------------- |
| **X-SGRV / SGRV**           | [`MATHAI/`](MATHAI) | Correlated-verification collapse + spec-grounded randomized verification       | NeurIPS 2026 main                  | Results measured, paper compiled        |
| **DAJV / `verifyensemble`** | [`dajv/`](dajv)     | Dependency-aware calibrated LLM-jury verification; cross-modality independence | ICML 2026 Workshop (Muslims in ML) | Library + paper, k=12 four-lab ensemble |

The pivot from the first track to the second is documented in
[`ARCHITECTURE_PIVOT_A.md`](ARCHITECTURE_PIVOT_A.md). `dajv` is **not** a
rewrite — it ports the sandbox harness, the frozen extractor prompt, the
adversarial probe set, and the cached extractor outputs from X-SGRV and treats
them as Stage-0 evidence.

## The throughline

1. **The failure** — same-model self-verification has a structural false-positive
   floor. A model that writes both a solution and the checks for it produces
   _correlated_ errors: bugs in the reasoning leak into the assertions. FPVR
   tracks the solver's own error rate and rises with difficulty.

2. **The first fix (X-SGRV)** — generate property tests from the _problem
   specification_, not the solution. A 2×2 ablation shows **independence**, not
   randomization, is the operative mechanism (Fisher p < 10⁻⁸). SGRV reaches
   100% precision on its top confidence tier at 33.7% coverage.

3. **The catch** — the literature (Kuai 2026, CARE 2026, Kim 2025) shows
   cross-family LLM verifiers are _also_ correlated; naïve consensus does not
   bound joint false positives by the product of marginals.

4. **The second fix (DAJV)** — measure the residual pairwise dependency and feed
   it into a calibrated aggregation rule with a Clopper–Pearson coverage band.
   The new empirical result: the **script-execution** modality (Python verdict)
   is near-independent of the **script-writing** modality (LLM self-rating) —
   cross-modality κ ≈ 0.02–0.06 vs within-modality κ ≈ 0.72 — an axis pure
   LLM-jury aggregators cannot exploit.

## Headline numbers

| Finding                                              | Track  | Number                   |
| ---------------------------------------------------- | ------ | ------------------------ |
| Same-model (ExeVer) FPVR, MATH-500 / Qwen2.5-Math-7B | X-SGRV | 13.8% [10.2–18.0]        |
| SGRV FPVR, same setting                              | X-SGRV | 0.0% [0.0–2.2]           |
| Independence is the operative factor (2×2 ablation)  | X-SGRV | Fisher p < 10⁻⁸          |
| SGRV top-tier selective prediction (MATH-500, n=175) | X-SGRV | 33.7% cov @ 100% acc     |
| Cross-modality κ (exec vs LLM self-rating)           | DAJV   | 0.02–0.06 vs 0.72 within |
| Calibration error reduction, DAJV vs naïve unanimous | DAJV   | **44%** ECE reduction    |

**Honest caveat that gates everything:** MATH-500 is contaminated for
Qwen2.5-Math-7B (~54.6% verbatim memorization). On contamination-clean **AIME
2025**, every top-tier signal tested collapses and baseline accuracy drops
76.6% → 10.0%. AIME 2025 + CleanMath are the realistic operating points. See
[`MATHAI/README.md`](MATHAI/README.md) for the full caveat list.

## Repository layout

```
.
├── MATHAI/                  # Track 1 — X-SGRV / SGRV / ExeVer (see MATHAI/README.md)
│   ├── src/ experiments/ analysis/ training/
│   ├── results/ figures/ paper/
│   ├── PROPOSAL.md          # technical proposal v8
│   └── ARCHITECTURE.md      # architecture v10
├── dajv/                    # Track 2 — verifyensemble library (see dajv/README.md)
│   ├── verifyensemble/      # core library: sandbox, extractors, dependency, aggregate, theory
│   ├── scripts/ tests/ artifacts/ paper/
│   ├── PRE_REGISTRATION_v2.md
│   └── NOVELTY_REPORT_2026-05-25.md
├── src/xsgrv/               # legacy root-level extractor scaffolding (superseded by MATHAI/src)
├── experiments/             # legacy root-level pilot (run_exp29_xsgrv_pilot.py)
└── ARCHITECTURE_PIVOT_A.md  # why and how Track 1 became Track 2
```

> Note: the top-level `src/` and `experiments/` are early scaffolding kept for
> provenance. Current code for Track 1 lives under `MATHAI/`.

## Getting started

Each track is self-contained. Start with the track that matches your interest:

- **Reproduce the failure-mode measurements & SGRV** → [`MATHAI/README.md`](MATHAI/README.md)
- **Use the calibrated jury-verification library** → [`dajv/README.md`](dajv/README.md)

```bash
# Track 2 (verifyensemble) installs as a package and runs on cached data, no API key:
cd dajv && pip install -e . && PYTHONPATH=. python3 scripts/run_aggregation_comparison.py
```

Both tracks fix `seed=42` everywhere and report Clopper–Pearson CIs for
proportions and percentile-bootstrap CIs (B=10,000) for means.

## Reproducibility & integrity

- Hypotheses and stopping criteria are pre-registered in-repo before data
  collection: [`MATHAI/PRE_COMMIT_n500.md`](MATHAI/PRE_COMMIT_n500.md),
  [`dajv/PRE_REGISTRATION_v2.md`](dajv/PRE_REGISTRATION_v2.md).
- Negative results are reported as results — e.g. the four pre-registered DAJV
  H7′ aggregators that all failed to beat default DAJV, and the AIME-2025
  collapse above.
- Novelty was checked against concurrent work before each pivot
  ([`dajv/NOVELTY_REPORT_2026-05-25.md`](dajv/NOVELTY_REPORT_2026-05-25.md)).

## License

`verifyensemble` (Track 2) is released under **BSD-3-Clause**
([`dajv/LICENSE`](dajv/LICENSE)). The remaining research code is unpublished and
provided for reproduction of the associated manuscripts; contact the author
before reuse.
