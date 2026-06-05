# ExeVer Results Summary

**Project:** Executable Step-Level Math Verification (ExeVer)
**Model:** Qwen/Qwen2.5-Math-7B-Instruct (primary solver + verifier)
**Date:** 2026-04-06
**Benchmarks:** MATH-500, GSM8K (500 samples each)

---

## 1. Experiment Status

| Exp | Name | N | Status | Date | Key Result |
|-----|------|---|--------|------|------------|
| 1 | Baseline Qwen-7B | 300 | Done | 2026-04-06 | 83.7% greedy pass@1 |
| 2 | Two-Pass Feasibility | 300 | Done | 2026-04-06 | 89.7% script validity, 26.2% step coverage |
| 3 | Cross-Model Verification | 300 | Done | 2026-04-06 | Cross-model echo 44.7% vs same-model 13.4% -- WORSE |
| 4 | Repair Baselines (300) | 300 | Done | 2026-04-06 | ExeVer 84.0% vs Maj@4 83.0% vs Best@4 87.7% |
| 5 | MATH-500 Full | 500 | Done | 2026-04-06 | ExeVer 83.4%, Greedy 83.2%, Best@4 88.8% |
| 6 | DeepSeek Solver | 500 | Done | 2026-04-06 | DeepSeek greedy 67.4%, ExeVer no improvement |
| 7 | SymCode Baseline | 500 | Done | 2026-04-06 | 63.6% -- code-only verification degrades heavily |
| 8 | Self-Correction | 500 | Done | 2026-04-06 | 74.8% -- self-correction HURTS (83.2% -> 74.8%) |
| 9 | Ablations | 300 | Done | 2026-04-06 | Verify-only = rederivation = multisample = 83.3% |
| 10 | Scaling | 300 | Done | 2026-04-06 | 1.5B and general-7B models: ExeVer matches greedy |
| 11 | LLM-as-Judge | 500 | Done | 2026-04-06 | Judge@4 = 83.0%, worse than Majority@4 (84.8%) |
| 12 | GSM8K Transfer | 500 | Done | 2026-04-06 | 90.8% coverage, 3.5% echo -- much better than MATH |

---

## 2. Master Accuracy Table (MATH-500)

| Method | Accuracy | Samples | Notes |
|--------|----------|---------|-------|
| Greedy CoT | 83.2% | 1 | Single-pass baseline |
| Sampled Pass@1 | 81.0% | 1 | Random sample from 4 candidates |
| **ExeVer** | **83.4%** | **1+V** | **Verify + repair pipeline** |
| Self-Correction | 74.8% | 1+C | Prompt-based "check your work" (Exp 8) |
| SymCode | 63.6% | 1 | Code-only symbolic verification (Exp 7) |
| LLM-as-Judge@4 | 83.0% | 4+J | Judge selects best of 4 (Exp 11) |
| Majority@4 | 84.8% | 4 | Majority vote over 4 samples |
| Best-of-4 (oracle) | 88.8% | 4 | Oracle picks correct answer if any |

**Key takeaway:** ExeVer (83.4%) marginally beats greedy (83.2%) with +0.2pp, but uses only 1 solution + verification. It underperforms Majority@4 (84.8%) which uses 4 samples. Best-of-4 (88.8%) shows 5.4pp of headroom remains.

---

## 3. By-Level Breakdown (MATH-500, Exp 5)

| Level | N | Greedy | ExeVer | Maj@4 | Best@4 | Self-Corr | SymCode | Judge@4 |
|-------|---|--------|--------|-------|--------|-----------|---------|---------|
| 1 | 100 | 96.0% | 97.0% | 96.0% | 97.0% | 93.0% | 80.0% | 96.0% |
| 2 | 100 | 91.0% | 91.0% | 90.0% | 93.0% | 86.0% | 74.0% | 87.0% |
| 3 | 100 | 86.0% | 86.0% | 89.0% | 93.0% | 76.0% | 65.0% | 86.0% |
| 4 | 100 | 77.0% | 77.0% | 83.0% | 87.0% | 67.0% | 48.0% | 82.0% |
| 5 | 100 | 66.0% | 66.0% | 66.0% | 74.0% | 52.0% | 51.0% | 64.0% |

**Observations:**
- ExeVer gains +1pp at Level 1, matches greedy exactly at Levels 2-5
- Self-correction degrades at every level, worst at Level 5 (-14pp)
- SymCode degrades catastrophically at Levels 4-5
- Majority@4 helps most at Levels 3-4; no help at Level 5

---

## 4. By-Subject Breakdown (MATH-500, Exp 5)

| Subject | N | Greedy | ExeVer | Maj@4 | Best@4 | SymCode | Judge@4 |
|---------|---|--------|--------|-------|--------|---------|---------|
| Prealgebra | 93 | 92.5% | 92.5% | 91.4% | 94.6% | 86.0% | 92.5% |
| Algebra | 131 | 91.6% | 92.4% | 95.4% | 96.2% | 68.7% | 93.1% |
| Count & Prob | 44 | 84.1% | 84.1% | 81.8% | 88.6% | 77.3% | 79.5% |
| Number Theory | 42 | 81.0% | 81.0% | 85.7% | 85.7% | 57.1% | 78.6% |
| Precalculus | 60 | 76.7% | 76.7% | 76.7% | 85.0% | 40.0% | 73.3% |
| Geometry | 46 | 76.1% | 76.1% | 80.4% | 84.8% | 52.2% | 73.9% |
| Inter. Algebra | 84 | 69.0% | 69.0% | 70.2% | 77.4% | 50.0% | 72.6% |

**Observations:**
- ExeVer gains +0.8pp on Algebra, matches greedy on all other subjects
- SymCode worst on Precalculus (40.0%) and Inter. Algebra (50.0%)
- Geometry is universally hard; Best@4 still only 84.8%

---

## 5. Verification Metrics (MATH-500, Exp 5)

### Pipeline Statistics

| Metric | Value |
|--------|-------|
| Valid scripts generated | 448 / 500 (89.6%) |
| Scripts executed successfully | 384 / 500 (76.8%) |
| All assertions passed | 320 / 500 (64.0%) |
| Assertion failures (triggered repair) | 64 / 500 (12.8%) |
| Runtime errors | 61 / 500 (12.2%) |
| Syntax errors | 52 / 500 (10.4%) |
| Timeouts | 3 / 500 (0.6%) |

### Verdict Distribution

| Verdict | Count | % |
|---------|-------|---|
| ALL_PASS | 320 | 64.0% |
| REPAIRED | 25 | 5.0% |
| REPAIRED_UNVERIFIED | 29 | 5.8% |
| FAIL_STEP_-1 (kept greedy) | 10 | 2.0% |
| SYNTAX_ERROR (fell back to greedy) | 52 | 10.4% |
| RUNTIME_ERROR (fell back to greedy) | 61 | 12.2% |
| TIMEOUT | 3 | 0.6% |

### Quality Metrics

| Metric | Value |
|--------|-------|
| **Verification coverage** | 66.0% (330/500 problems where verification produced a verdict) |
| **Effective coverage** (incl. REPAIRED_UNVERIFIED) | 76.8% (384/500) |
| **Echo chamber rate** | 13.8% (44/320 ALL_PASS cases simply restated the answer) |
| **Repair success rate** | 84.4% (54/64 assertion failures successfully repaired) |
| **Script validity rate** | 89.6% (448/500) |
| **Avg assertions per script** | 1.86 |
| **Nontrivial assertion rate** | 98.6% (823/835) |

---

## 6. GSM8K Comparison (Exp 12)

| Metric | GSM8K | MATH-500 | Delta |
|--------|-------|----------|-------|
| Greedy accuracy | 96.0% | 83.2% | +12.8pp |
| ExeVer accuracy | 96.0% | 83.4% | +12.6pp |
| Verification coverage | 90.8% | 66.0% | +24.8pp |
| Echo chamber rate | 3.5% | 13.8% | -10.3pp |
| Valid scripts | 96.8% | 89.6% | +7.2pp |
| Avg assertions/script | 2.76 | 1.86 | +0.90 |
| Repair attempted | 70 | 64 | -- |
| Repair success | 68/70 (97.1%) | 54/64 (84.4%) | +12.7pp |

**Key takeaway:** ExeVer works dramatically better on GSM8K -- nearly full coverage (90.8%), minimal echo chamber (3.5%), and near-perfect repair. This confirms the pipeline is sound but MATH-level problems exceed current code-generation capability.

---

## 7. Ablation Results (Exp 9, N=300)

| Ablation | Accuracy | vs. Full ExeVer |
|----------|----------|-----------------|
| **ExeVer Full** | **84.0%** | -- |
| A1: Verify-only (no repair) | 83.3% | -0.7pp |
| A3: Rederivation (resolve from script) | 83.3% | -0.7pp |
| A4: Interleaved (gen code with solution) | 79.3% | -4.7pp |
| A5: Multisample verify (pick verified) | 83.3% | -0.7pp |
| CoT greedy baseline | 83.7% | -0.3pp |
| Majority@4 | 83.0% | -1.0pp |

**Observations:**
- Repair contributes +0.7pp over verify-only
- Interleaved generation is significantly worse (-4.7pp) -- coupling code with reasoning degrades both
- Multisample verify (pick best verified solution from 4) equals verify-only, suggesting verification signal is too weak to discriminate
- All ablations cluster within 1pp of greedy, confirming the core finding that ExeVer's gains are marginal on MATH

---

## 8. Scaling Results (Exp 10, N=300)

| Model | Greedy | ExeVer | Maj@4 | Best@4 | ExeVer - Greedy |
|-------|--------|--------|-------|--------|-----------------|
| Qwen2.5-Math-1.5B | 73.0% | 73.0% | 78.3% | 84.3% | +0.0pp |
| Qwen2.5-7B-General | 70.3% | 71.0% | 77.3% | 85.7% | +0.7pp |
| Qwen2.5-Math-7B | 83.7% | 84.0% | 83.0% | 87.7% | +0.3pp |

**Observations:**
- ExeVer gains are near-zero across model scales
- Smaller/weaker models generate even fewer valid verification scripts
- The 1.5B model verified 223/300 problems; the general 7B verified only 162/300
- Best-of-4 oracle shows large headroom at all scales (11pp+ above greedy for smaller models)

---

## 9. Key Findings

### What works
- **Pipeline is mechanically sound:** 89.6% valid scripts, 84.4% repair success on MATH; 96.8% valid, 97.1% repair on GSM8K
- **Echo chamber is low on easy problems:** Only 3.5% on GSM8K, confirming the model generates genuinely novel verification code for simpler math
- **Nontrivial assertions:** 98.6% of assertions are nontrivial (not just restating the answer)
- **Repair is effective when triggered:** 84.4% success rate on MATH, 97.1% on GSM8K

### What does not work
- **Marginal accuracy gains:** ExeVer improves over greedy by only +0.2pp on MATH-500 (83.4% vs 83.2%)
- **Coverage bottleneck:** Only 66.0% of MATH problems get meaningful verification; 34% fall back to greedy
- **Echo chamber on hard problems:** 13.8% on MATH vs 3.5% on GSM8K -- the model struggles to verify what it cannot trivially compute
- **Cross-model verification is WORSE:** Exp 3 showed echo chamber increases from 13.4% to 44.7% with a cross-model verifier (DeepSeek-R1-Distill)
- **Self-correction actively harms:** Drops accuracy from 83.2% to 74.8% (-8.4pp)
- **SymCode (code-only) is catastrophic:** 63.6%, a -19.6pp degradation from greedy
- **No scaling benefit:** Gains remain near-zero across 1.5B, 7B-general, and 7B-math models
- **LLM-as-Judge underperforms majority vote:** 83.0% vs 84.8%

### Critical insight
The fundamental limitation is that **the same model that solves the problem also verifies it**. When the model gets a problem wrong, it typically lacks the mathematical understanding to verify the solution correctly. This creates an inherent ceiling where verification cannot catch errors the model is fundamentally incapable of detecting. The 13.8% echo chamber rate on MATH (vs 3.5% on GSM8K) directly quantifies this phenomenon.

---

## 10. Statistical Analysis (completed 2026-04-06)

### Confidence Calibration (DONE)
- ALL_PASS accuracy: **86.2%** (276/320)
- REPAIRED accuracy: **100.0%** (54/54)
- FAIL_STEP_-1 accuracy: **20.0%** (2/10)
- Fallback accuracy: **69.0%** (117/170)
- Gap: +17.2pp between ALL_PASS and fallback → verification IS informative as confidence signal

### Echo Chamber = Perfect Error Detector (KILLER FINDING)
- Non-echo ALL_PASS: **100.0% accuracy** (276/276)
- Echo ALL_PASS: **0.0% accuracy** (0/44)
- Echo rate by level: L1=1.4%, L2=7.1%, L3=12.2%, L4=17.1%, L5=30.8%
- **Echo detection is a PERFECT binary classifier** within verified solutions

### Bootstrap Confidence Intervals (DONE)
- Greedy: 83.2% [79.8%, 86.4%]
- ExeVer: 83.4% [80.2%, 86.6%]
- Paired diff: +0.20pp [0.00, +0.60], p(≤0) = 0.37 → **NOT significant**
- Implication: ExeVer's value is in verification/analysis, NOT accuracy improvement

### Coverage-Conditioned Accuracy (DONE)
- Greedy on verified subset: 88.0%
- ExeVer on verified subset: 88.2%
- Greedy on fallback subset: 69.0%
- Verification-eligible problems are inherently easier (+18.9pp)

---

## 11. Remaining Work

### Must-do for paper
- [x] **Confidence calibration analysis** ← DONE
- [x] **Error taxonomy** ← DONE (num theory L5 worst, inter. algebra/precalculus hardest)
- [x] **Coverage-conditioned accuracy** ← DONE
- [x] **Statistical significance tests** ← DONE (NOT significant)
- [ ] **Stronger solver experiment** -- Run ExeVer with a 70B+ model (blocked: Modal billing limit)

### Should-do
- [ ] **Process reward model comparison** -- Compare ExeVer against Math-Shepherd or OmegaPRM as process verifiers
- [ ] **Hybrid pipeline** -- Use ExeVer for high-coverage subjects (algebra, prealgebra) and majority vote for low-coverage ones (geometry, precalculus)
- [ ] **Multi-round verification** -- Allow 2-3 rounds of script generation for syntax/runtime errors before falling back
- [ ] **AIME/AMC evaluation** -- Test on competition-level problems to quantify ceiling
- [ ] **Cost analysis** -- Compare token cost of ExeVer vs Majority@4 vs Best-of-N with reward model

### Stretch goals
- [ ] Train a small verification-specialized model (distill from Qwen-Math-72B verifier scripts)
- [ ] Formal verification backend (Lean4/Isabelle) for algebra/number theory subset
- [ ] Human study: Does step-level verification output improve human debugging?
