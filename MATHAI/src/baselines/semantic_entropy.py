"""Semantic Entropy baselines (Kuhn, Gal, Farquhar ICLR 2023).

Implements two clustering variants:

1. NLI route (faithful to Kuhn 2023): bidirectional entailment clustering
   via a DeBERTa-large-mnli model. Two answers cluster together iff the
   model judges both (premise=A, hypothesis=B) and (premise=B, hypothesis=A)
   as entailment.

2. Math-specific route: cluster by symbolic equivalence using the existing
   answer_check.answers_equivalent function. Faster, cleaner for math,
   but deviates from the published NLI method.

Usage:
    scorer = SemanticEntropyScorer.from_nli()          # slower, published method
    scorer = SemanticEntropyScorer.from_math()         # faster, math-native
    samples = [(raw_text, boxed_answer), ...]           # 10 sampled generations
    entropy = scorer.score(question, samples)           # higher = more uncertain
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

try:
    # Math-specific route: reuse existing equivalence checker
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.eval.answer_check import answers_equivalent
except Exception:
    answers_equivalent = None  # type: ignore


@dataclass
class Sample:
    """One solver sample: raw chain of thought + extracted final answer."""
    raw: str
    answer: str


def _discrete_entropy(cluster_sizes: Sequence[int]) -> float:
    """Shannon entropy over discrete cluster probabilities (nats)."""
    n = sum(cluster_sizes)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in cluster_sizes if c > 0)


class SemanticEntropyScorer:
    """Compute semantic entropy over a bag of sampled answers."""

    def __init__(self, equivalence_fn: Callable[[str, str, str], bool], name: str):
        """
        Args:
            equivalence_fn: function (question, a, b) -> bool. Returns True iff
                sample a and sample b belong to the same meaning cluster.
            name: "nli" or "math" (used for logging / record-keeping).
        """
        self.equivalence_fn = equivalence_fn
        self.name = name

    @classmethod
    def from_math(cls) -> "SemanticEntropyScorer":
        """Cluster by SymPy symbolic equivalence of boxed answers."""
        if answers_equivalent is None:
            raise RuntimeError("answer_check module unavailable")

        def _eq(_question: str, a: str, b: str) -> bool:
            try:
                result = answers_equivalent(a, b)
                if isinstance(result, tuple):
                    return bool(result[0])
                return bool(result)
            except Exception:
                return a.strip() == b.strip()

        return cls(_eq, name="math")

    @classmethod
    def from_nli(cls, model_name: str = "microsoft/deberta-large-mnli") -> "SemanticEntropyScorer":
        """Cluster by bidirectional NLI entailment (Kuhn 2023 method)."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # DeBERTa-MNLI labels: 0=contradiction, 1=neutral, 2=entailment
        label2id = model.config.label2id
        entail_idx = label2id.get("ENTAILMENT", 2)

        @torch.no_grad()
        def _one_direction(premise: str, hypothesis: str) -> bool:
            enc = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits[0]
            pred = int(logits.argmax().item())
            return pred == entail_idx

        def _eq(question: str, a: str, b: str) -> bool:
            # Kuhn 2023: bidirectional entailment. Build full premise/hypothesis
            # by prepending the question so the NLI model has context.
            pa = f"{question} {a}".strip()
            pb = f"{question} {b}".strip()
            return _one_direction(pa, pb) and _one_direction(pb, pa)

        return cls(_eq, name="nli")

    def cluster(self, question: str, samples: Sequence[Sample]) -> list[list[int]]:
        """Group sample indices into equivalence-class clusters.

        Greedy clustering: each sample joins the first existing cluster whose
        representative is equivalent, else starts a new cluster. Kuhn 2023
        uses the same approach (not a full transitive closure).
        """
        clusters: list[list[int]] = []
        reps: list[int] = []
        for i, s in enumerate(samples):
            placed = False
            for cid, rep in enumerate(reps):
                try:
                    if self.equivalence_fn(question, samples[rep].answer, s.answer):
                        clusters[cid].append(i)
                        placed = True
                        break
                except Exception:
                    continue
            if not placed:
                clusters.append([i])
                reps.append(i)
        return clusters

    def score(self, question: str, samples: Sequence[Sample]) -> dict:
        """Return entropy + cluster metadata for one problem."""
        if not samples:
            return {"entropy": 0.0, "n_clusters": 0, "cluster_sizes": [], "variant": self.name}
        clusters = self.cluster(question, samples)
        sizes = [len(c) for c in clusters]
        entropy = _discrete_entropy(sizes)
        return {
            "entropy": entropy,
            "n_clusters": len(clusters),
            "cluster_sizes": sizes,
            "variant": self.name,
            "n_samples": len(samples),
        }
