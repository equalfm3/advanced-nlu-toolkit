"""Named entity mention detection for entity linking.

Identifies entity mentions (named entities) in text that should be
linked to a knowledge base.  Uses a simple BIO tagger over token
embeddings with heuristic features (capitalization, known entity
patterns).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EntityMention:
    """A detected entity mention for linking.

    Attributes:
        start: Start token index (inclusive).
        end: End token index (exclusive).
        text: Surface text of the mention.
        mention_type: Coarse type hint (PERSON, ORG, LOC, MISC).
        confidence: Detection confidence.
    """

    start: int
    end: int
    text: str
    mention_type: str = "MISC"
    confidence: float = 0.0


# BIO labels for entity mention detection
_BIO_LABELS: list[str] = [
    "O",
    "B-ENTITY",
    "I-ENTITY",
]


class LinkingMentionDetector:
    """Detect named entity mentions for entity linking.

    Combines a learned BIO tagger with capitalization heuristics.
    In production, this would be a fine-tuned NER model.

    Args:
        hidden_dim: Token embedding dimension.
        use_heuristics: Whether to boost scores for capitalized tokens.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        use_heuristics: bool = True,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.use_heuristics = use_heuristics
        rng = np.random.default_rng(42)
        n_labels = len(_BIO_LABELS)
        self.W = rng.standard_normal(
            (n_labels, hidden_dim)
        ).astype(np.float32) * 0.1
        self.b = np.zeros(n_labels, dtype=np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def _is_entity_like(self, token: str, position: int) -> bool:
        """Heuristic: capitalized non-initial tokens are likely entities."""
        if not token or not token[0].isupper():
            return False
        if position == 0:
            return len(token) > 1 and token[1:].islower()
        return True

    def detect(
        self,
        tokens: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[EntityMention]:
        """Detect entity mentions in a tokenized sentence.

        Args:
            tokens: Sentence tokens.
            embeddings: Token embeddings (T, hidden_dim).

        Returns:
            List of EntityMention objects.
        """
        n = len(tokens)
        if n == 0:
            return []

        if embeddings is None:
            rng = np.random.default_rng(0)
            embeddings = rng.standard_normal(
                (n, self.hidden_dim)
            ).astype(np.float32)

        tags: list[tuple[str, float]] = []
        for i, token in enumerate(tokens):
            logits = self.W @ embeddings[i] + self.b

            if self.use_heuristics and self._is_entity_like(token, i):
                logits[1] += 1.0  # boost B-ENTITY
                logits[2] += 0.5  # boost I-ENTITY

            probs = self._softmax(logits)
            best_idx = int(np.argmax(probs))
            tag = _BIO_LABELS[best_idx]

            if tag == "I-ENTITY" and (not tags or tags[-1][0] == "O"):
                tag = "B-ENTITY"

            tags.append((tag, float(probs[best_idx])))

        return self._decode_mentions(tokens, tags)

    def _decode_mentions(
        self,
        tokens: list[str],
        tags: list[tuple[str, float]],
    ) -> list[EntityMention]:
        """Decode BIO tags into entity mentions.

        Args:
            tokens: Sentence tokens.
            tags: BIO tag sequence.

        Returns:
            List of EntityMention objects.
        """
        mentions: list[EntityMention] = []
        current_start: Optional[int] = None
        confidences: list[float] = []

        for i, (tag, conf) in enumerate(tags):
            if tag == "B-ENTITY":
                if current_start is not None:
                    text = " ".join(tokens[current_start:i])
                    mentions.append(
                        EntityMention(
                            start=current_start,
                            end=i,
                            text=text,
                            confidence=float(np.mean(confidences)),
                        )
                    )
                current_start = i
                confidences = [conf]
            elif tag == "I-ENTITY" and current_start is not None:
                confidences.append(conf)
            else:
                if current_start is not None:
                    text = " ".join(tokens[current_start:i])
                    mentions.append(
                        EntityMention(
                            start=current_start,
                            end=i,
                            text=text,
                            confidence=float(np.mean(confidences)),
                        )
                    )
                    current_start = None
                    confidences = []

        if current_start is not None:
            text = " ".join(tokens[current_start:])
            mentions.append(
                EntityMention(
                    start=current_start,
                    end=len(tokens),
                    text=text,
                    confidence=float(np.mean(confidences)),
                )
            )

        return mentions


if __name__ == "__main__":
    tokens = "Marie Curie discovered radium in Paris .".split()
    detector = LinkingMentionDetector(hidden_dim=32)

    mentions = detector.detect(tokens)
    print(f"Sentence: {' '.join(tokens)}")
    print(f"\nDetected {len(mentions)} mention(s):")
    for m in mentions:
        print(f"  '{m.text}' [{m.start}:{m.end}]  "
              f"type={m.mention_type}  conf={m.confidence:.3f}")
