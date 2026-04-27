"""BIO and span-based argument labeling for semantic role labeling.

Given a predicate position, labels each token with a BIO tag indicating
its role in the predicate-argument structure:

    P(y_t | x, p) = softmax(W · BiLSTM([x_t; v_p]))

Supports both BIO sequence tagging and span-based decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.srl.predicate_detector import Predicate


# Standard SRL argument labels
SRL_LABELS: list[str] = [
    "O",
    "B-ARG0", "I-ARG0",
    "B-ARG1", "I-ARG1",
    "B-ARG2", "I-ARG2",
    "B-ARGM-TMP", "I-ARGM-TMP",
    "B-ARGM-LOC", "I-ARGM-LOC",
    "B-ARGM-MNR", "I-ARGM-MNR",
    "B-V",
]


@dataclass
class ArgumentSpan:
    """A labeled argument span.

    Attributes:
        start: Start token index (inclusive).
        end: End token index (exclusive).
        label: Argument role label (e.g., "ARG0", "ARG1").
        text: Surface text of the argument.
        confidence: Average tag confidence over the span.
    """

    start: int
    end: int
    label: str
    text: str = ""
    confidence: float = 0.0


class ArgumentLabeler:
    """BIO sequence tagger for SRL argument labeling.

    For each predicate, produces a BIO tag sequence over the sentence
    tokens.  Tags are decoded into argument spans.

    Args:
        hidden_dim: Token embedding dimension.
        predicate_dim: Predicate indicator embedding dimension.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        predicate_dim: int = 16,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.predicate_dim = predicate_dim
        self.labels = SRL_LABELS
        rng = np.random.default_rng(42)
        input_dim = hidden_dim + predicate_dim
        n_labels = len(self.labels)
        self.W = rng.standard_normal(
            (n_labels, input_dim)
        ).astype(np.float32) * 0.1
        self.b = np.zeros(n_labels, dtype=np.float32)
        self.pred_emb_true = rng.standard_normal(
            predicate_dim
        ).astype(np.float32) * 0.1
        self.pred_emb_false = rng.standard_normal(
            predicate_dim
        ).astype(np.float32) * 0.1

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def label_tokens(
        self,
        tokens: list[str],
        predicate: Predicate,
        embeddings: Optional[np.ndarray] = None,
    ) -> list[tuple[str, float]]:
        """Produce BIO tags for each token given a predicate.

        Args:
            tokens: Sentence tokens.
            predicate: The target predicate.
            embeddings: Token embeddings (T, hidden_dim).

        Returns:
            List of (tag, confidence) pairs, one per token.
        """
        n = len(tokens)
        if embeddings is None:
            rng = np.random.default_rng(0)
            embeddings = rng.standard_normal(
                (n, self.hidden_dim)
            ).astype(np.float32)

        tags: list[tuple[str, float]] = []
        for t in range(n):
            v_p = self.pred_emb_true if t == predicate.index else self.pred_emb_false
            features = np.concatenate([embeddings[t], v_p])
            logits = self.W @ features + self.b
            probs = self._softmax(logits)
            best_idx = int(np.argmax(probs))
            tags.append((self.labels[best_idx], float(probs[best_idx])))

        return self._enforce_bio_constraints(tags)

    def _enforce_bio_constraints(
        self,
        tags: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Fix invalid BIO sequences (e.g., I-X without preceding B-X).

        Args:
            tags: Raw predicted tags.

        Returns:
            Corrected tag sequence.
        """
        corrected: list[tuple[str, float]] = []
        prev_label = "O"
        for tag, conf in tags:
            if tag.startswith("I-"):
                expected_b = "B-" + tag[2:]
                if prev_label != expected_b and prev_label != tag:
                    tag = "B-" + tag[2:]
            prev_label = tag
            corrected.append((tag, conf))
        return corrected

    def decode_spans(
        self,
        tokens: list[str],
        tags: list[tuple[str, float]],
    ) -> list[ArgumentSpan]:
        """Decode BIO tags into argument spans.

        Args:
            tokens: Sentence tokens.
            tags: BIO tag sequence with confidences.

        Returns:
            List of ArgumentSpan objects.
        """
        spans: list[ArgumentSpan] = []
        current_label: Optional[str] = None
        current_start = 0
        confidences: list[float] = []

        for i, (tag, conf) in enumerate(tags):
            if tag.startswith("B-"):
                if current_label is not None:
                    spans.append(
                        ArgumentSpan(
                            start=current_start,
                            end=i,
                            label=current_label,
                            text=" ".join(tokens[current_start:i]),
                            confidence=float(np.mean(confidences)),
                        )
                    )
                current_label = tag[2:]
                current_start = i
                confidences = [conf]
            elif tag.startswith("I-") and current_label == tag[2:]:
                confidences.append(conf)
            else:
                if current_label is not None:
                    spans.append(
                        ArgumentSpan(
                            start=current_start,
                            end=i,
                            label=current_label,
                            text=" ".join(tokens[current_start:i]),
                            confidence=float(np.mean(confidences)),
                        )
                    )
                    current_label = None
                    confidences = []

        if current_label is not None:
            spans.append(
                ArgumentSpan(
                    start=current_start,
                    end=len(tokens),
                    label=current_label,
                    text=" ".join(tokens[current_start:]),
                    confidence=float(np.mean(confidences)),
                )
            )

        return spans

    def label(
        self,
        tokens: list[str],
        predicate: Predicate,
        embeddings: Optional[np.ndarray] = None,
    ) -> list[ArgumentSpan]:
        """Full labeling pipeline: tag tokens then decode spans.

        Args:
            tokens: Sentence tokens.
            predicate: Target predicate.
            embeddings: Token embeddings.

        Returns:
            Decoded argument spans.
        """
        tags = self.label_tokens(tokens, predicate, embeddings)
        return self.decode_spans(tokens, tags)


if __name__ == "__main__":
    tokens = "Marie Curie discovered radium in her laboratory .".split()
    pred = Predicate(index=2, lemma="discover", text="discovered", sense="discover.01")

    labeler = ArgumentLabeler(hidden_dim=32, predicate_dim=8)
    tags = labeler.label_tokens(tokens, pred)

    print(f"Sentence: {' '.join(tokens)}")
    print(f"Predicate: '{pred.text}' at index {pred.index}\n")
    print("BIO tags:")
    for tok, (tag, conf) in zip(tokens, tags):
        print(f"  {tok:15s} → {tag:15s} ({conf:.3f})")

    spans = labeler.decode_spans(tokens, tags)
    print(f"\nDecoded {len(spans)} argument span(s):")
    for s in spans:
        print(f"  {s.label}: '{s.text}' [{s.start}:{s.end}]  "
              f"conf={s.confidence:.3f}")
