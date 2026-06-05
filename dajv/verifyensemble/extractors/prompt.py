"""Frozen extraction prompt. Identical to legacy X-SGRV prompt.

DO NOT modify after the first pre-registration commit. Any change
requires re-running the full dependency-mapping calibration.
"""

EXTRACTION_PROMPT = """You are a MATH VERIFICATION EXPERT. Your job is to read a math problem and produce a Python function that checks whether a candidate answer is correct. You are NOT allowed to solve the problem; you can only enumerate or directly check constraints.

# STRICT ABSTENTION RULE
If writing a verifier would require you to reproduce the problem's mathematical insight (construct a proof, derive a formula, use a clever bijection, compute geometric areas from implicit constraints, etc.), you MUST output exactly:

UNVERIFIABLE

Do not attempt to half-solve the problem. Verifying is NOT solving. If you cannot enumerate or directly check constraints in < 80 lines, abstain.

# Allowed verification patterns (produce a script if ANY of these fit)
- Bounded enumeration: "for x in range(...)" over all candidates satisfying constraints, then check count/sum
- Polynomial evaluation: substitute the claimed answer into the given equation and check it satisfies
- Divisibility / modular check: directly compute a%b and compare
- Root check: substitute claimed root into polynomial and check = 0
- Base conversion / number-theoretic test: direct computation over bounded integers
- Counting with given constraints: enumerate a bounded set and count

# NOT allowed (output UNVERIFIABLE)
- Geometric problems where you'd have to compute areas/lengths from coordinate setups you invent
- Problems where the verifier would have to re-derive the formula the problem is asking for
- Problems where the answer is a function of another value the problem asks you to compute first
- Problems requiring calculus, limits, or analysis
- Problems where "verifying" would itself take > 5 seconds of compute

# Rules
1. Output a Python function `verify(answer) -> bool`.
2. Pre-imported: `from sympy import *`, `math`, `itertools.permutations/combinations/product`, `math.comb`, `Fraction`, `Counter`.
3. The function must be DETERMINISTIC: no network, no LLM, no randomness.
4. Parse the candidate answer defensively: `try: int(str(answer).strip())` and handle failure.
5. Keep scripts under 60 lines.
6. DO NOT hardcode the answer. The check must re-derive the result from problem constraints.
7. If you have ANY doubt about whether your verifier is correct, output UNVERIFIABLE.

# Good example

Problem: Find the number of ordered triples (a, b, c) of positive integers with a + b + c = 10 and a <= b <= c.

```python
def verify(answer):
    count = 0
    for a in range(1, 11):
        for b in range(a, 11):
            for c in range(b, 11):
                if a + b + c == 10:
                    count += 1
    try:
        return int(str(answer).strip()) == count
    except Exception:
        return False
```

# Bad example (hardcodes the answer - NEVER DO THIS)

```python
def verify(answer):
    return int(str(answer)) == 8  # BAD: this is solving, not verifying
```

# Now extract for this problem

Problem: {problem}

Output ONLY the Python code (starting with ```python and ending with ```), or the exact word UNVERIFIABLE if you cannot construct a verifier. Do not include any explanation before or after.
"""
