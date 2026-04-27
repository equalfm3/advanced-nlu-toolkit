"""Agglomerative mention clustering for coreference resolution.

After the mention-ranking model assigns each mention its best antecedent,
this module merges those pairwise links into entity clusters using a
union-find structure, then optionally refines clusters with an
agglomerative pass based on average-link similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.coreference.mention_detector import Span


@dataclass
class Cluster:
    """A coreference cluster grouping coreferent mentions.

    Attributes:
        cluster_id: Unique cluster identifier.
        mentions: Ordered list of mentions in the cluster.
    """

    cluster_id: int
    mentions: list[Span] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Number of mentions in the cluster."""
        return len(self.mentions)

    def representative(self) -> Span:
        """Return the first (usually most informative) mention."""
        return self.mentions[0]


class UnionFind:
    """Disjoint-set data structure with path compression and union by rank.

    Used to efficiently merge mention pairs into clusters.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Merge sets containing x and y."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


class ClusterMerger:
    """Merge pairwise antecedent links into entity clusters.

    Stage 1: Union-find over (mention, antecedent) pairs.
    Stage 2 (optional): Agglomerative refinement — merge the two
    closest clusters if their average-link similarity exceeds a
    threshold.

    Args:
        merge_threshold: Minimum average-link similarity to merge
            two clusters during the agglomerative pass.
    """

    def __init__(self, merge_threshold: float = 0.0) -> None:
        self.merge_threshold = merge_threshold

    def merge_from_links(
        self,
        mentions: list[Span],
        antecedent_indices: list[int],
    ) -> list[Cluster]:
        """Build clusters from antecedent assignments.

        Args:
            mentions: Ordered list of detected mentions.
            antecedent_indices: For each mention i, the index of its
                best antecedent (or -1 / i for new-cluster).

        Returns:
            List of Cluster objects.
        """
        n = len(mentions)
        uf = UnionFind(n)

        for i, ant in enumerate(antecedent_indices):
            if 0 <= ant < n and ant != i:
                uf.union(i, ant)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = uf.find(i)
            groups.setdefault(root, []).append(i)

        clusters: list[Cluster] = []
        for cid, (_, member_ids) in enumerate(sorted(groups.items())):
            cluster_mentions = [mentions[j] for j in member_ids]
            clusters.append(Cluster(cluster_id=cid, mentions=cluster_mentions))
        return clusters

    def agglomerative_refine(
        self,
        clusters: list[Cluster],
        embeddings: dict[int, np.ndarray],
    ) -> list[Cluster]:
        """Optionally merge clusters using average-link similarity.

        Args:
            clusters: Initial clusters from union-find.
            embeddings: Mapping from mention start index to its
                embedding vector.

        Returns:
            Refined list of clusters.
        """
        if len(clusters) <= 1:
            return clusters

        def cluster_embedding(c: Cluster) -> np.ndarray:
            vecs = [embeddings[m.start] for m in c.mentions if m.start in embeddings]
            if not vecs:
                return np.zeros(64, dtype=np.float32)
            return np.mean(vecs, axis=0)

        active = list(clusters)
        changed = True
        while changed and len(active) > 1:
            changed = False
            best_sim = -float("inf")
            best_pair = (-1, -1)
            reps = [cluster_embedding(c) for c in active]

            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    norm_i = np.linalg.norm(reps[i])
                    norm_j = np.linalg.norm(reps[j])
                    if norm_i < 1e-8 or norm_j < 1e-8:
                        continue
                    sim = float(np.dot(reps[i], reps[j]) / (norm_i * norm_j))
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)

            if best_sim > self.merge_threshold and best_pair[0] >= 0:
                i, j = best_pair
                merged = Cluster(
                    cluster_id=active[i].cluster_id,
                    mentions=active[i].mentions + active[j].mentions,
                )
                active = [c for k, c in enumerate(active) if k not in (i, j)]
                active.append(merged)
                changed = True

        for idx, c in enumerate(active):
            c.cluster_id = idx
        return active


if __name__ == "__main__":
    mentions = [
        Span(0, 2, "Marie Curie"),
        Span(3, 4, "radium"),
        Span(6, 7, "She"),
        Span(9, 12, "two Nobel Prizes"),
    ]
    antecedents = [-1, -1, 0, -1]

    merger = ClusterMerger()
    clusters = merger.merge_from_links(mentions, antecedents)

    print("Coreference clusters:")
    for c in clusters:
        names = [f"'{m.text}'" for m in c.mentions]
        print(f"  Cluster {c.cluster_id}: {', '.join(names)}")
