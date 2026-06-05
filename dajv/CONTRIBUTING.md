# Contributing to verifyensemble / DAJV

The library is anonymous during the ICML 2027 review cycle; we still
welcome contributions through anonymous channels. After de-anonymization
the repo will move to a public GitHub URL.

## Development setup

```bash
git clone <repo>
cd dajv
pip install -e ".[dev]"
```

## Run the test suite

```bash
PYTHONPATH=. python3 -m pytest tests/
```

A passing PR must have a green test run on the developer's machine
and no new linting regressions.

## Adding a new extractor

1. Add a registry entry to
   `verifyensemble/extractors/api_wrappers.py:EXTRACTOR_REGISTRY`
   with the model id and provider.
2. If the provider is new (not Together, OpenAI, Anthropic, Google),
   add a `<provider>_call` wrapper in the same file.
3. Update `EXTRACTOR_REGISTRY` and re-run dependency mapping to add
   the new model to the matrix.
4. **Do not** modify the frozen extractor prompt
   (`verifyensemble/extractors/prompt.py`). Any change invalidates
   the calibration. If the prompt must change, the version must be
   bumped in `PRE_REGISTRATION_v<N>.md` and the calibration set must
   be re-run.

## Adding a new aggregation rule

1. Add a module under `verifyensemble/aggregate/` exposing a single
   function `<name>_aggregate(votes, calibration_or_weights, ...) ->
   dict`. The return dict must include `P_correct`, `lower`, `upper`,
   `recommendation`, `n_working`, `n_accept` keys.
2. Add a row to `scripts/run_aggregation_comparison.py`'s method
   table so the new rule is compared head-to-head.
3. Add unit tests under `tests/`.

## Adding a new dependency estimator

Add a function with signature `<name>(accept_i, accept_j,
problem_correct) -> float` under `verifyensemble/dependency/` and
register it in `DependencyMatrix.from_accept`.

## Coding style

- Public functions have docstrings stating purpose, args, returns.
- Type hints on all public APIs.
- No silent exception swallowing inside the sandbox.
- Reproducibility: any RNG-using code accepts a `seed` argument.
- Caching: any expensive computation must be deterministic given
  its inputs so it can be checkpointed and resumed.

## Pre-registration commitments

This is a pre-registered research project. The hypotheses, calibration
splits, decision thresholds, frozen prompts, and adversarial probe
sets are committed in `PRE_REGISTRATION_v1.md`. Changes to any of
those values require a new `PRE_REGISTRATION_v<N>.md` with explicit
rationale.
