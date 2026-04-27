"""Bi-encoder entity disambiguation for entity linking.

Scores each candidate entity against the mention context using a
weighted combination of three signals:

    score(m, e) = α · sim(h_m, h_e) + β · P_prior(e|m) + γ · coherence(e, E_doc)

where h_m is the contextual mention embedding, h_e is the entity
embedding, P_prior is the prior probability from alias statistics,
and coherence measures compatibility with other entities already
linked in the document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.linking.kb_index import KBIndex, KBEntity, build_sample_kb
from src.linking.mention_detector import EntityMention, LinkingMentionDetector
from src.linking.candidate_generator import CandidateGenerator, Candidate


@dataclass
class LinkedEntity:
    """A mention linked to a KB entity.

    Attributes:
        mention: The original mention.
        entity: The linked KB entity (None if NIL).
        score: Disambiguation score.
        prior: Prior probability component.
        similarity: Embedding similarity component.
        coherence: Document coherence component.
    """

    mention: EntityMention
    entity: Optional[KBEntity] = None
    score: float = 0.0
    prior: float = 0.0
    similarity: float = 0.0
    coherence: float = 0.0

    @property
    def is_nil(self) -> bool:
        """Whether the mention could not be linked."""
        return self.entity is None


class EntityDisambiguator:
    """Disambiguate entity mentions against KB candidates.

    Combines embedding similarity, prior probability, and document-level
    coherence to select the best entity for each mention.

    Args:
        kb: Knowledge base index.
        alpha: Weight for embedding similarity.
        beta: Weight for prior probability.
        gamma: Weight for document coherence.
        nil_threshold: Minimum score to accept a link (below = NIL).
        hidden_dim: Mention embedding dimension.
    """

    def __init__(
        self,
        kb: KBIndex,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        nil_threshold: float = 0.1,
        hidden_dim: int = 64,
    ) -> None:
        self.kb = kb
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nil_threshold = nil_threshold
        self.hidden_dim = hidden_dim
        self.candidate_gen = CandidateGenerator(kb, max_candidates=30)

    def _mention_embedding(
        self,
        tokens: list[str],
        mention: EntityMention,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute a contextual mention embedding.

        Args:
            tokens: Document tokens.
            mention: The mention to embed.
            embeddings: Token embeddings.

        Returns:
            Mention embedding vector.
        """
        if embeddings is not None:
            return embeddings[mention.start : mention.end].mean(axis=0)
        rng = np.random.default_rng(hash(mention.text) % (2**31))
        return rng.standard_normal(self.hidden_dim).astype(np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _prior_probability(self, mention_text: str, entity: KBEntity) -> float:
        """Estimate P_prior(e|m) from alias statistics and popularity.

        Args:
            mention_text: Surface text of the mention.
            entity: Candidate entity.

        Returns:
            Prior probability estimate.
        """
        name_match = 1.0 if mention_text.lower() == entity.name.lower() else 0.0
        alias_match = 0.0
        for alias in entity.aliases:
            if mention_text.lower() == alias.lower():
                alias_match = 0.8
                break
        base = max(name_match, alias_match)
        return 0.5 * base + 0.5 * entity.popularity

    def _coherence_score(
        self,
        entity: KBEntity,
        linked_entities: list[KBEntity],
    ) -> float:
        """Score entity coherence with already-linked entities.

        Args:
            entity: Candidate entity.
            linked_entities: Entities already linked in the document.

        Returns:
            Coherence score.
        """
        if not linked_entities or entity.embedding is None:
            return 0.0

        similarities: list[float] = []
        for other in linked_entities:
            if other.embedding is not None:
                sim = self._cosine_similarity(entity.embedding, other.embedding)
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def disambiguate_mention(
        self,
        tokens: list[str],
        mention: EntityMention,
        context: str = "",
        linked_entities: Optional[list[KBEntity]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> LinkedEntity:
        """Disambiguate a single mention.

        Args:
            tokens: Document tokens.
            mention: The mention to disambiguate.
            context: Surrounding context text.
            linked_entities: Previously linked entities for coherence.
            embeddings: Token embeddings.

        Returns:
            LinkedEntity with the best match or NIL.
        """
        linked = linked_entities or []
        candidates = self.candidate_gen.generate(mention, context)

        if not candidates:
            return LinkedEntity(mention=mention)

        h_m = self._mention_embedding(tokens, mention, embeddings)
        best_entity: Optional[KBEntity] = None
        best_score = -float("inf")
        best_prior = 0.0
        best_sim = 0.0
        best_coh = 0.0

        for cand in candidates:
            entity = cand.entity
            sim = 0.0
            if entity.embedding is not None:
                sim = self._cosine_similarity(h_m, entity.embedding)

            prior = self._prior_probability(mention.text, entity)
            coh = self._coherence_score(entity, linked)

            score = self.alpha * sim + self.beta * prior + self.gamma * coh
            if score > best_score:
                best_score = score
                best_entity = entity
                best_prior = prior
                best_sim = sim
                best_coh = coh

        if best_score < self.nil_threshold:
            return LinkedEntity(mention=mention, score=best_score)

        return LinkedEntity(
            mention=mention,
            entity=best_entity,
            score=best_score,
            prior=best_prior,
            similarity=best_sim,
            coherence=best_coh,
        )

    def disambiguate_document(
        self,
        tokens: list[str],
        mentions: list[EntityMention],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[LinkedEntity]:
        """Disambiguate all mentions in a document.

        Processes mentions left-to-right, using previously linked
        entities for coherence scoring.

        Args:
            tokens: Document tokens.
            mentions: Detected entity mentions.
            embeddings: Token embeddings.

        Returns:
            List of LinkedEntity results.
        """
        context = " ".join(tokens)
        linked_entities: list[KBEntity] = []
        results: list[LinkedEntity] = []

        for mention in mentions:
            result = self.disambiguate_mention(
                tokens, mention, context, linked_entities, embeddings
            )
            results.append(result)
            if result.entity is not None:
                linked_entities.append(result.entity)

        return results


if __name__ == "__main__":
    kb = build_sample_kb(embedding_dim=32)
    disambiguator = EntityDisambiguator(kb, hidden_dim=32)

    tokens = "Marie Curie discovered radium in Washington .".split()
    mentions = [
        EntityMention(0, 2, "Marie Curie"),
        EntityMention(3, 4, "radium"),
        EntityMention(5, 6, "Washington"),
    ]

    results = disambiguator.disambiguate_document(tokens, mentions)
    print(f"Document: {' '.join(tokens)}\n")
    print("Entity linking results:")
    for r in results:
        if r.is_nil:
            print(f"  '{r.mention.text}' → NIL (score={r.score:.3f})")
        else:
            assert r.entity is not None
            print(f"  '{r.mention.text}' → {r.entity.entity_id}: {r.entity.name}  "
                  f"(score={r.score:.3f}, sim={r.similarity:.3f}, "
                  f"prior={r.prior:.3f}, coh={r.coherence:.3f})")
