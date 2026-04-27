"""Alias table + TF-IDF candidate retrieval for entity linking.

Given an entity mention, generates a ranked list of candidate KB
entities using two strategies:

1. Alias table lookup — exact and fuzzy string matching against known
   entity aliases.
2. TF-IDF re-ranking — scores candidates by TF-IDF similarity between
   the mention context and entity descriptions.

Returns ~30 candidates per mention for the disambiguator to re-rank.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.linking.kb_index import KBIndex, KBEntity
from src.linking.mention_detector import EntityMention


@dataclass
class Candidate:
    """A candidate entity for linking.

    Attributes:
        entity: The KB entity.
        alias_score: Score from alias table matching.
        tfidf_score: Score from TF-IDF context similarity.
        combined_score: Weighted combination of scores.
    """

    entity: KBEntity
    alias_score: float = 0.0
    tfidf_score: float = 0.0
    combined_score: float = 0.0


class TFIDFScorer:
    """TF-IDF scorer for mention context vs. entity descriptions.

    Builds a simple TF-IDF model over entity descriptions and scores
    mention contexts against them.
    """

    def __init__(self) -> None:
        self._doc_freq: Counter[str] = Counter()
        self._n_docs: int = 0
        self._entity_tfs: dict[str, Counter[str]] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def index_entity(self, entity_id: str, description: str) -> None:
        """Add an entity description to the TF-IDF index.

        Args:
            entity_id: Entity identifier.
            description: Entity description text.
        """
        tokens = self._tokenize(description)
        tf = Counter(tokens)
        self._entity_tfs[entity_id] = tf
        for term in set(tokens):
            self._doc_freq[term] += 1
        self._n_docs += 1

    def score(self, context: str, entity_id: str) -> float:
        """Score a mention context against an entity description.

        Args:
            context: Mention context text.
            entity_id: Entity to score against.

        Returns:
            TF-IDF cosine similarity score.
        """
        if entity_id not in self._entity_tfs:
            return 0.0

        context_tokens = self._tokenize(context)
        context_tf = Counter(context_tokens)
        entity_tf = self._entity_tfs[entity_id]

        def tfidf_vec(tf: Counter[str]) -> dict[str, float]:
            vec: dict[str, float] = {}
            for term, count in tf.items():
                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + self._n_docs / (1 + df))
                vec[term] = count * idf
            return vec

        ctx_vec = tfidf_vec(context_tf)
        ent_vec = tfidf_vec(entity_tf)

        common = set(ctx_vec) & set(ent_vec)
        if not common:
            return 0.0

        dot = sum(ctx_vec[t] * ent_vec[t] for t in common)
        norm_ctx = math.sqrt(sum(v ** 2 for v in ctx_vec.values()))
        norm_ent = math.sqrt(sum(v ** 2 for v in ent_vec.values()))
        if norm_ctx < 1e-8 or norm_ent < 1e-8:
            return 0.0
        return dot / (norm_ctx * norm_ent)


class CandidateGenerator:
    """Generate candidate entities for a mention.

    Combines alias table lookup with TF-IDF re-ranking.

    Args:
        kb: Knowledge base index.
        max_candidates: Maximum candidates to return per mention.
        alias_weight: Weight for alias match score.
        tfidf_weight: Weight for TF-IDF score.
    """

    def __init__(
        self,
        kb: KBIndex,
        max_candidates: int = 30,
        alias_weight: float = 0.6,
        tfidf_weight: float = 0.4,
    ) -> None:
        self.kb = kb
        self.max_candidates = max_candidates
        self.alias_weight = alias_weight
        self.tfidf_weight = tfidf_weight
        self.tfidf = TFIDFScorer()
        self._build_tfidf_index()

    def _build_tfidf_index(self) -> None:
        """Index all KB entity descriptions for TF-IDF scoring."""
        for eid, entity in self.kb._entities.items():
            desc = f"{entity.name} {entity.description}"
            self.tfidf.index_entity(eid, desc)

    def _alias_match_score(self, mention_text: str, entity: KBEntity) -> float:
        """Score how well a mention matches an entity's aliases.

        Args:
            mention_text: Surface text of the mention.
            entity: Candidate entity.

        Returns:
            Alias match score in [0, 1].
        """
        mention_lower = mention_text.lower()
        if mention_lower == entity.name.lower():
            return 1.0
        for alias in entity.aliases:
            if mention_lower == alias.lower():
                return 0.9
        if mention_lower in entity.name.lower():
            return 0.5
        for alias in entity.aliases:
            if mention_lower in alias.lower():
                return 0.4
        return 0.0

    def generate(
        self,
        mention: EntityMention,
        context: str = "",
    ) -> list[Candidate]:
        """Generate ranked candidates for a mention.

        Args:
            mention: The entity mention to link.
            context: Surrounding sentence context.

        Returns:
            Ranked list of Candidate objects.
        """
        alias_matches = self.kb.lookup_alias(mention.text)

        seen_ids: set[str] = set()
        candidates: list[Candidate] = []

        for entity in alias_matches:
            alias_score = self._alias_match_score(mention.text, entity)
            tfidf_score = self.tfidf.score(context, entity.entity_id) if context else 0.0
            combined = self.alias_weight * alias_score + self.tfidf_weight * tfidf_score
            candidates.append(
                Candidate(
                    entity=entity,
                    alias_score=alias_score,
                    tfidf_score=tfidf_score,
                    combined_score=combined,
                )
            )
            seen_ids.add(entity.entity_id)

        if len(candidates) < self.max_candidates and context:
            rng = np.random.default_rng(0)
            query_emb = rng.standard_normal(self.kb.embedding_dim).astype(np.float32)
            nn_results = self.kb.nearest_neighbors(query_emb, top_k=self.max_candidates)
            for entity, sim in nn_results:
                if entity.entity_id not in seen_ids:
                    tfidf_score = self.tfidf.score(context, entity.entity_id)
                    candidates.append(
                        Candidate(
                            entity=entity,
                            alias_score=0.0,
                            tfidf_score=tfidf_score,
                            combined_score=self.tfidf_weight * tfidf_score,
                        )
                    )
                    seen_ids.add(entity.entity_id)

        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        return candidates[: self.max_candidates]


if __name__ == "__main__":
    from src.linking.kb_index import build_sample_kb

    kb = build_sample_kb(embedding_dim=32)
    generator = CandidateGenerator(kb, max_candidates=5)

    mention = EntityMention(start=0, end=1, text="Washington")
    context = "Washington is the capital of the United States"

    candidates = generator.generate(mention, context)
    print(f"Mention: '{mention.text}'")
    print(f"Context: '{context}'\n")
    print(f"Top {len(candidates)} candidates:")
    for c in candidates:
        print(f"  {c.entity.entity_id}: {c.entity.name:25s}  "
              f"alias={c.alias_score:.2f}  tfidf={c.tfidf_score:.3f}  "
              f"combined={c.combined_score:.3f}")
