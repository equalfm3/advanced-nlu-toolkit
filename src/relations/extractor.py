"""Span-pair relation classifier.

Given a sentence with two marked entity spans, classifies the relation
type (or "no relation").  The classifier uses concatenated span
representations with element-wise product:

    P(r | e1, e2, c) = softmax(W_r · [h_e1; h_e2; h_e1 ⊙ h_e2] + b_r)

This captures both individual entity semantics and their pairwise
interaction without an explicit attention mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.relations.typing import EntityType, EntityTyper, TypedEntity, is_type_compatible


@dataclass
class RelationTriple:
    """An extracted relation triple.

    Attributes:
        subject: Subject entity text.
        relation: Predicted relation label.
        obj: Object entity text.
        confidence: Classification confidence.
        subject_span: (start, end) token indices.
        object_span: (start, end) token indices.
    """

    subject: str
    relation: str
    obj: str
    confidence: float = 0.0
    subject_span: tuple[int, int] = (0, 0)
    object_span: tuple[int, int] = (0, 0)


# Default relation schema
DEFAULT_RELATIONS: list[str] = [
    "NA",
    "bornIn",
    "headquarteredIn",
    "foundedBy",
    "dateOfBirth",
    "memberOf",
    "locatedIn",
    "creatorOf",
]


class RelationClassifier:
    """Feed-forward relation classifier over span pairs.

    Args:
        hidden_dim: Dimension of span embeddings.
        relation_types: List of relation labels (first should be "NA").
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        relation_types: Optional[list[str]] = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.relation_types = relation_types or DEFAULT_RELATIONS
        rng = np.random.default_rng(42)
        input_dim = 3 * hidden_dim
        n_classes = len(self.relation_types)
        self.W = rng.standard_normal(
            (n_classes, input_dim)
        ).astype(np.float32) * 0.1
        self.b = np.zeros(n_classes, dtype=np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def classify(
        self,
        h_subj: np.ndarray,
        h_obj: np.ndarray,
    ) -> tuple[str, float, dict[str, float]]:
        """Classify the relation between two entity spans.

        Args:
            h_subj: Subject span embedding (hidden_dim,).
            h_obj: Object span embedding (hidden_dim,).

        Returns:
            Tuple of (predicted_relation, confidence, all_scores).
        """
        hadamard = h_subj * h_obj
        features = np.concatenate([h_subj, h_obj, hadamard])
        logits = self.W @ features + self.b
        probs = self._softmax(logits)
        scores = {r: float(probs[i]) for i, r in enumerate(self.relation_types)}
        best_idx = int(np.argmax(probs))
        return self.relation_types[best_idx], float(probs[best_idx]), scores


class RelationExtractor:
    """Extract relation triples from a tokenized sentence.

    Combines entity typing with span-pair classification.  Entity pairs
    are first filtered by type compatibility, then classified.

    Args:
        hidden_dim: Token embedding dimension.
        relation_types: Relation label vocabulary.
        na_threshold: Minimum non-NA confidence to emit a triple.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        relation_types: Optional[list[str]] = None,
        na_threshold: float = 0.0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.classifier = RelationClassifier(hidden_dim, relation_types)
        self.typer = EntityTyper(hidden_dim)
        self.na_threshold = na_threshold

    def _span_embedding(
        self,
        embeddings: np.ndarray,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Average-pool token embeddings for a span."""
        return embeddings[start:end].mean(axis=0)

    def extract(
        self,
        tokens: list[str],
        entity_spans: list[tuple[int, int]],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[RelationTriple]:
        """Extract relations between all entity pairs in a sentence.

        Args:
            tokens: Sentence tokens.
            entity_spans: List of (start, end) entity spans.
            embeddings: Token embeddings (T, hidden_dim).

        Returns:
            List of RelationTriple objects for non-NA predictions.
        """
        n = len(tokens)
        if embeddings is None:
            rng = np.random.default_rng(0)
            embeddings = rng.standard_normal(
                (n, self.hidden_dim)
            ).astype(np.float32)

        typed_entities = self.typer.type_spans(tokens, entity_spans, embeddings)

        triples: list[RelationTriple] = []
        for i, subj in enumerate(typed_entities):
            for j, obj in enumerate(typed_entities):
                if i == j:
                    continue

                h_subj = self._span_embedding(embeddings, subj.start, subj.end)
                h_obj = self._span_embedding(embeddings, obj.start, obj.end)
                rel, conf, _ = self.classifier.classify(h_subj, h_obj)

                if rel != "NA" and conf > self.na_threshold:
                    triples.append(
                        RelationTriple(
                            subject=subj.text,
                            relation=rel,
                            obj=obj.text,
                            confidence=conf,
                            subject_span=(subj.start, subj.end),
                            object_span=(obj.start, obj.end),
                        )
                    )

        triples.sort(key=lambda t: t.confidence, reverse=True)
        return triples


if __name__ == "__main__":
    tokens = "Albert Einstein was born in Ulm in 1879 .".split()
    entity_spans = [(0, 2), (5, 6), (7, 8)]

    extractor = RelationExtractor(hidden_dim=32)
    triples = extractor.extract(tokens, entity_spans)

    print(f"Sentence: {' '.join(tokens)}")
    print(f"Entities: {[' '.join(tokens[s:e]) for s, e in entity_spans]}")
    print(f"\nExtracted {len(triples)} relation(s):")
    for t in triples:
        print(f"  ({t.subject}, {t.relation}, {t.obj})  "
              f"conf={t.confidence:.3f}")
