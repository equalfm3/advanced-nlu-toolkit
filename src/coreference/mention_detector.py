"""Span enumeration and mention scoring for coreference resolution.

Enumerates candidate spans up to a maximum width, scores each span as a
potential entity mention, and prunes to the top-λT candidates.  This is
the first stage of the end-to-end mention-ranking pipeline (Lee et al.,
2017).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Span:
    """A contiguous token span in a document.

    Attributes:
        start: Start token index (inclusive).
        end: End token index (exclusive).
        text: Surface text of the span.
        mention_score: Scalar score indicating mention likelihood.
    """

    start: int
    end: int
    text: str = ""
    mention_score: float = 0.0

    @property
    def width(self) -> int:
        """Number of tokens in the span."""
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        """Check whether two spans overlap."""
        return self.start < other.end and other.start < self.end


class MentionScorer:
    """Feed-forward mention scorer.

    Computes s_m(i) = W_m · [g_start; g_end; g_hat; phi(width)] + b_m
    where g_start/g_end are boundary embeddings, g_hat is an attention-
    weighted span representation, and phi encodes span width.

    Args:
        hidden_dim: Dimension of token embeddings.
        max_width: Maximum span width to consider.
        width_embedding_dim: Dimension of width feature embedding.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        max_width: int = 10,
        width_embedding_dim: int = 16,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.max_width = max_width
        self.width_embedding_dim = width_embedding_dim

        rng = np.random.default_rng(42)
        input_dim = 3 * hidden_dim + width_embedding_dim
        self.W = rng.standard_normal((1, input_dim)).astype(np.float32) * 0.01
        self.b = np.zeros(1, dtype=np.float32)
        self.width_emb = rng.standard_normal(
            (max_width + 1, width_embedding_dim)
        ).astype(np.float32) * 0.1

    def score_span(
        self,
        token_embeddings: np.ndarray,
        start: int,
        end: int,
    ) -> float:
        """Score a single span as a potential mention.

        Args:
            token_embeddings: Array of shape (T, hidden_dim).
            start: Start index (inclusive).
            end: End index (exclusive).

        Returns:
            Scalar mention score.
        """
        g_start = token_embeddings[start]
        g_end = token_embeddings[end - 1]
        g_hat = token_embeddings[start:end].mean(axis=0)
        width = min(end - start, self.max_width)
        phi = self.width_emb[width]
        features = np.concatenate([g_start, g_end, g_hat, phi])
        return float((self.W @ features + self.b).item())


class MentionDetector:
    """Enumerate candidate spans and prune to top mentions.

    Given a tokenized document, generates all spans up to *max_width*
    tokens, scores each with a MentionScorer, and retains the top
    λT candidates (where T is the document length).

    Args:
        max_width: Maximum span width.
        lam: Ratio of mentions to keep relative to document length.
        hidden_dim: Token embedding dimension.
    """

    def __init__(
        self,
        max_width: int = 10,
        lam: float = 0.4,
        hidden_dim: int = 64,
    ) -> None:
        self.max_width = max_width
        self.lam = lam
        self.scorer = MentionScorer(hidden_dim=hidden_dim, max_width=max_width)

    def enumerate_spans(self, tokens: list[str]) -> list[Span]:
        """Generate all candidate spans up to max_width.

        Args:
            tokens: Document tokens.

        Returns:
            List of Span objects (unscored).
        """
        spans: list[Span] = []
        n = len(tokens)
        for start in range(n):
            for end in range(start + 1, min(start + self.max_width + 1, n + 1)):
                text = " ".join(tokens[start:end])
                spans.append(Span(start=start, end=end, text=text))
        return spans

    def detect(
        self,
        tokens: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[Span]:
        """Detect and prune mentions from a tokenized document.

        Args:
            tokens: Document tokens.
            embeddings: Token embeddings of shape (T, hidden_dim).
                If None, random embeddings are generated for demo purposes.

        Returns:
            Top-λT scored spans sorted by descending mention score.
        """
        n = len(tokens)
        if n == 0:
            return []

        if embeddings is None:
            rng = np.random.default_rng(0)
            embeddings = rng.standard_normal(
                (n, self.scorer.hidden_dim)
            ).astype(np.float32)

        spans = self.enumerate_spans(tokens)
        for span in spans:
            span.mention_score = self.scorer.score_span(
                embeddings, span.start, span.end
            )

        spans.sort(key=lambda s: s.mention_score, reverse=True)
        keep = max(1, int(self.lam * n))
        return spans[:keep]


if __name__ == "__main__":
    doc = "Marie Curie discovered radium . She won two Nobel Prizes .".split()
    detector = MentionDetector(max_width=5, lam=0.5, hidden_dim=32)

    print(f"Document ({len(doc)} tokens): {' '.join(doc)}")
    total_spans = detector.enumerate_spans(doc)
    print(f"Candidate spans (width ≤ 5): {len(total_spans)}")

    mentions = detector.detect(doc, embeddings=None)
    print(f"Top mentions (λT = {max(1, int(0.5 * len(doc)))}):")
    for m in mentions:
        print(f"  [{m.start}:{m.end}] '{m.text}'  score={m.mention_score:.4f}")
