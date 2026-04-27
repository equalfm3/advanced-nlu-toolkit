"""Verb and predicate identification for semantic role labeling.

Detects predicates (verbs) in a sentence and optionally disambiguates
their PropBank frame sense.  Contextualized embeddings help distinguish
"run a company" (frame: run.01 — operate) from "run a marathon"
(frame: run.02 — move quickly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Predicate:
    """A detected predicate with optional sense disambiguation.

    Attributes:
        index: Token index of the predicate.
        lemma: Lemmatized form of the verb.
        text: Surface form.
        sense: PropBank frame sense (e.g., "run.01").
        confidence: Detection confidence score.
    """

    index: int
    lemma: str
    text: str
    sense: str = ""
    confidence: float = 0.0


# Simple POS-like heuristic: common verb suffixes and known verbs
_VERB_INDICATORS: set[str] = {
    "is", "was", "were", "are", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    "can", "shall", "must",
}

_VERB_SUFFIXES: tuple[str, ...] = ("ed", "ing", "es", "ize", "ify", "ate")


# Minimal PropBank sense inventory for demo
SENSE_INVENTORY: dict[str, list[str]] = {
    "run": ["run.01", "run.02"],
    "give": ["give.01", "give.02"],
    "take": ["take.01", "take.02", "take.03"],
    "make": ["make.01", "make.02"],
    "get": ["get.01", "get.02", "get.03"],
    "discover": ["discover.01"],
    "win": ["win.01"],
    "born": ["bear.02"],
}


class PredicateDetector:
    """Detect predicates in a tokenized sentence.

    Uses a combination of heuristic verb detection and a learned
    classifier over token embeddings.  In production, this would be
    replaced by a fine-tuned transformer head.

    Args:
        hidden_dim: Token embedding dimension.
        threshold: Minimum confidence to accept a predicate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        threshold: float = 0.3,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        rng = np.random.default_rng(42)
        self.W_detect = rng.standard_normal(
            (1, hidden_dim)
        ).astype(np.float32) * 0.1
        self.b_detect = np.zeros(1, dtype=np.float32)
        self.W_sense = rng.standard_normal(
            (4, hidden_dim)
        ).astype(np.float32) * 0.1
        self.b_sense = np.zeros(4, dtype=np.float32)

    def _is_verb_heuristic(self, token: str) -> bool:
        """Simple heuristic check for verb-like tokens."""
        lower = token.lower()
        if lower in _VERB_INDICATORS:
            return True
        return any(lower.endswith(s) for s in _VERB_SUFFIXES)

    def _simple_lemma(self, token: str) -> str:
        """Minimal lemmatization by stripping common suffixes."""
        lower = token.lower()
        for suffix in ("ed", "ing", "es", "s"):
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                return lower[: -len(suffix)]
        return lower

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)

    def detect(
        self,
        tokens: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[Predicate]:
        """Detect predicates in a sentence.

        Args:
            tokens: Sentence tokens.
            embeddings: Token embeddings (T, hidden_dim).

        Returns:
            List of detected Predicate objects.
        """
        n = len(tokens)
        if n == 0:
            return []

        if embeddings is None:
            rng = np.random.default_rng(0)
            embeddings = rng.standard_normal(
                (n, self.hidden_dim)
            ).astype(np.float32)

        predicates: list[Predicate] = []
        for i, token in enumerate(tokens):
            if not self._is_verb_heuristic(token):
                continue

            logit = float((self.W_detect @ embeddings[i] + self.b_detect).item())
            conf = self._sigmoid(logit)

            if conf >= self.threshold:
                lemma = self._simple_lemma(token)
                sense = self._disambiguate_sense(lemma, embeddings[i])
                predicates.append(
                    Predicate(
                        index=i,
                        lemma=lemma,
                        text=token,
                        sense=sense,
                        confidence=conf,
                    )
                )

        return predicates

    def _disambiguate_sense(
        self,
        lemma: str,
        embedding: np.ndarray,
    ) -> str:
        """Select the best PropBank sense for a predicate.

        Args:
            lemma: Lemmatized verb form.
            embedding: Contextualized token embedding.

        Returns:
            PropBank frame identifier (e.g., "run.01").
        """
        senses = SENSE_INVENTORY.get(lemma, [])
        if not senses:
            return f"{lemma}.01"
        if len(senses) == 1:
            return senses[0]

        logits = self.W_sense[: len(senses)] @ embedding + self.b_sense[: len(senses)]
        best = int(np.argmax(logits))
        return senses[best]


if __name__ == "__main__":
    tokens = "Marie Curie discovered radium and was awarded two Nobel Prizes .".split()
    detector = PredicateDetector(hidden_dim=32, threshold=0.0)

    predicates = detector.detect(tokens)
    print(f"Sentence: {' '.join(tokens)}")
    print(f"\nDetected {len(predicates)} predicate(s):")
    for p in predicates:
        print(f"  [{p.index}] '{p.text}' → {p.sense}  "
              f"(lemma='{p.lemma}', conf={p.confidence:.3f})")
