"""End-to-end coreference resolution with antecedent ranking.

Implements the mention-ranking approach from Lee et al. (2017).  For each
detected mention, the model scores all prior mentions as potential
antecedents and selects the highest-scoring one.  The combined score is:

    s(i, j) = s_m(i) + s_m(j) + s_a(i, j)

where s_m is the mention score and s_a is the pairwise antecedent score
computed from span representations and distance features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.coreference.mention_detector import MentionDetector, Span
from src.coreference.cluster_merger import ClusterMerger, Cluster


@dataclass
class AntecedentLink:
    """A scored link between a mention and its best antecedent.

    Attributes:
        mention_idx: Index of the current mention.
        antecedent_idx: Index of the best antecedent (-1 = new entity).
        score: Combined s(i, j) score.
    """

    mention_idx: int
    antecedent_idx: int
    score: float


class AntecedentScorer:
    """Pairwise antecedent scorer.

    Computes s_a(i, j) = W_a · [g_i; g_j; g_i ⊙ g_j; φ(i,j)] + b_a
    where g_i, g_j are span representations, ⊙ is element-wise product,
    and φ encodes distance features.

    Args:
        span_dim: Dimension of span representations.
        distance_bins: Number of distance buckets.
        distance_dim: Dimension of distance feature embedding.
    """

    def __init__(
        self,
        span_dim: int = 64,
        distance_bins: int = 10,
        distance_dim: int = 16,
    ) -> None:
        self.span_dim = span_dim
        self.distance_bins = distance_bins
        rng = np.random.default_rng(42)
        input_dim = 3 * span_dim + distance_dim
        self.W = rng.standard_normal((1, input_dim)).astype(np.float32) * 0.01
        self.b = np.zeros(1, dtype=np.float32)
        self.dist_emb = rng.standard_normal(
            (distance_bins, distance_dim)
        ).astype(np.float32) * 0.1

    def _bucket_distance(self, dist: int) -> int:
        """Map token distance to a bucket index."""
        if dist <= 0:
            return 0
        log_dist = min(int(np.log2(dist)), self.distance_bins - 1)
        return log_dist

    def score(
        self,
        g_i: np.ndarray,
        g_j: np.ndarray,
        distance: int,
    ) -> float:
        """Score a (mention_i, antecedent_j) pair.

        Args:
            g_i: Span representation for mention i.
            g_j: Span representation for antecedent j.
            distance: Token distance between spans.

        Returns:
            Scalar antecedent score s_a(i, j).
        """
        hadamard = g_i * g_j
        bucket = self._bucket_distance(distance)
        phi = self.dist_emb[bucket]
        features = np.concatenate([g_i, g_j, hadamard, phi])
        return float((self.W @ features + self.b).item())


class CoreferenceModel:
    """End-to-end coreference resolution pipeline.

    Combines mention detection, antecedent ranking, and cluster merging
    into a single forward pass.

    Args:
        max_width: Maximum mention span width.
        lam: Mention pruning ratio (keep top λT).
        hidden_dim: Token embedding dimension.
        new_entity_score: Threshold for starting a new entity cluster.
    """

    def __init__(
        self,
        max_width: int = 10,
        lam: float = 0.4,
        hidden_dim: int = 64,
        new_entity_score: float = 0.0,
    ) -> None:
        self.detector = MentionDetector(
            max_width=max_width, lam=lam, hidden_dim=hidden_dim
        )
        self.antecedent_scorer = AntecedentScorer(span_dim=hidden_dim)
        self.merger = ClusterMerger()
        self.hidden_dim = hidden_dim
        self.new_entity_score = new_entity_score

    def _span_representation(
        self,
        embeddings: np.ndarray,
        span: Span,
    ) -> np.ndarray:
        """Compute a span representation by averaging token embeddings.

        Args:
            embeddings: Token embeddings of shape (T, hidden_dim).
            span: The span to represent.

        Returns:
            Span vector of shape (hidden_dim,).
        """
        return embeddings[span.start : span.end].mean(axis=0)

    def resolve(
        self,
        tokens: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[Cluster]:
        """Run full coreference resolution on a tokenized document.

        Args:
            tokens: Document tokens.
            embeddings: Token embeddings (T, hidden_dim). Generated
                randomly if None (for demo purposes).

        Returns:
            List of coreference clusters.
        """
        n = len(tokens)
        if n == 0:
            return []

        if embeddings is None:
            rng = np.random.default_rng(0)
            embeddings = rng.standard_normal(
                (n, self.hidden_dim)
            ).astype(np.float32)

        mentions = self.detector.detect(tokens, embeddings)
        if not mentions:
            return []

        mentions.sort(key=lambda s: s.start)

        span_reps = [self._span_representation(embeddings, m) for m in mentions]

        antecedent_indices: list[int] = []
        for i, m_i in enumerate(mentions):
            best_idx = -1
            best_score = self.new_entity_score

            for j in range(i):
                m_j = mentions[j]
                dist = m_i.start - m_j.end
                sa = self.antecedent_scorer.score(span_reps[i], span_reps[j], dist)
                total = m_i.mention_score + m_j.mention_score + sa
                if total > best_score:
                    best_score = total
                    best_idx = j

            antecedent_indices.append(best_idx)

        clusters = self.merger.merge_from_links(mentions, antecedent_indices)
        return [c for c in clusters if c.size > 0]


if __name__ == "__main__":
    doc = "Marie Curie discovered radium . She won two Nobel Prizes .".split()
    model = CoreferenceModel(max_width=5, lam=0.5, hidden_dim=32)

    print(f"Document: {' '.join(doc)}\n")
    clusters = model.resolve(doc)

    print(f"Found {len(clusters)} cluster(s):")
    for c in clusters:
        texts = [f"'{m.text}' [{m.start}:{m.end}]" for m in c.mentions]
        print(f"  Cluster {c.cluster_id}: {', '.join(texts)}")
