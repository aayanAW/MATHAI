# MATHAI — Correlated-Verification Collapse & Spec-Grounded Selective Prediction

Empirical study of **why LLM self-verification fails on math reasoning**, and a
high-precision selective-prediction signal that survives it.

This is the X-SGRV / SGRV / ExeVer research track. The library implementation of
the follow-on calibration work lives in [`../dajv/`](../dajv) (see the
[repository root README](../README.md) for how the two tracks relate).

- **Paper title:** _Cross-Family LLM-Extracted Symbolic Verification for
  Contamination-Robust Selective Prediction on LLM Math Reasoning_
- **Target venues:** NeurIPS 2026 main track (primary), ICML AI4Math Workshop
  (floor), Nature Machine Intelligence (stretch)
- **Status:** v8 mechanism + selective prediction complete; v9 X-SGRV pivot
  complete; v10 cross-family scale-up complete; paper compiled.
- **Full design docs:** [`PROPOSAL.md`](PROPOSAL.md) (technical proposal v8),
  [`ARCHITECTURE.md`](ARCHITECTURE.md) (architecture v10).

## The core finding

Same-model self-verification has a **structural false-positive floor**: when a
model writes both the solution _and_ the checks for that solution, errors in the
reasoning propagate into the verification assertions. We call this
**correlated-verification collapse**. The symptom — false-positive verification
rate (FPVR) tracks the solver's own error rate and _grows with problem
difficulty_.

The remedy is **Specification-Grounded Randomized Verification (SGRV)**: generate
property tests from the _problem specification_ rather than from the model's own
solution. Independence between the thing being checked and the check is what
breaks the correlation.

## Headline results (measured)

| Result                                                            | Number                               |
| ----------------------------------------------------------------- | ------------------------------------ |
| ExeVer (same-model) FPVR, MATH-500 / Qwen2.5-Math-7B              | 13.8% [10.2–18.0]                    |
| SGRV FPVR, same setting                                           | 0.0% [0.0–2.2]                       |
| 2×2 ablation — independence is the operative factor               | Fisher p < 10⁻⁸                      |
| SGRV selective prediction top tier (MATH-500, n=175)              | 33.7% cov @ **100%** acc [93.9, 100] |
| Cross-extractor consensus (Llama-3.3-70B ∩ DeepSeek-V3), MATH-175 | 73/73 = **1.000** @ 41.7% cov        |

**The 2×2 mechanism ablation** (the headline experiment) separates two candidate
mechanisms — _independence_ (spec-grounded vs solution-grounded) and
_randomization_ (deterministic vs randomized testing) — and shows independence,
not randomization, drives the FPVR reduction.

## Honest caveats

- **MATH-500 is contaminated** for Qwen2.5-Math-7B (≈54.6% verbatim
  memorization, Wu et al. 2025). On the contamination-clean **AIME 2025** set
  _every_ top-tier signal tested collapses (SGRV 0/30, ExeVer 0/30, SC 0/30) and
  baseline accuracy drops 76.6% → 10.0%. Use AIME 2025 + CleanMath as the
  realistic operating points, not MATH-500.
- SGRV only tests the **computationally checkable** subset of steps (~20–70%
  depending on domain). Proof structure, deductive leaps, and strategic choices
  remain untestable.
- This is an **empirical** paper — no theorems, no deployment trial,
  text-only.

## Repository layout

```
MATHAI/
├── src/
│   ├── exever/        # same-model baseline verification (the failure mode)
│   ├── pbt/           # SGRV framework: claim classifier, templates, pipeline, PRM data
│   ├── xsgrv/         # cross-family LLM symbolic-verifier extractor
│   ├── baselines/     # CoT, semantic entropy
│   ├── inference/     # model wrappers (Together / Modal)
│   ├── data/          # MATH / AIME / CleanMath loaders
│   └── eval/          # answer checking, FPVR, risk-coverage metrics
├── experiments/       # run_exp{1..53}_*.py — every experiment, numbered
├── analysis/          # risk-coverage, calibration, selective-prediction, audits
├── training/          # train_prm_modal.py — Qwen2.5-Math-7B LoRA PRM
├── results/           # per-experiment JSON outputs (source of truth for the paper)
├── figures/           # fig{1..26}_*.{pdf,png}
├── paper/             # LaTeX source + .bib
├── modal_app.py       # Modal inference app
└── modal_inference.py
```

## Reproduction

```bash
pip install -r requirements.txt          # SymPy, z3-solver, Together/Modal SDKs, stats
export TOGETHER_API_KEY=...              # solver + extractor generation
export HF_TOKEN=...                      # PRM weights, gated datasets

# experiments are self-contained and numbered; e.g. the 2x2 mechanism ablation:
python experiments/run_exp24_2x2_ablation.py

# selective-prediction head-to-head (SGRV vs self-consistency vs verbalized):
python experiments/run_exp25_selective_prediction.py

# cross-family consensus on the X-SGRV extractor caches:
python analysis/xsgrv_consensus.py
```

All experiments write JSON to `results/`. Statistics use Clopper–Pearson CIs for
proportions and percentile bootstrap (B=10,000) for means, fixed `seed=42`
throughout. Every numerical claim in the paper is verified against the
corresponding `results/*.json`.
