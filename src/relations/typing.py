"""Fine-grained entity typing for relation filtering.

Assigns fine-grained type labels (PERSON, ORG, LOC, DATE, etc.) to
entity spans.  Used by the relation extractor to filter incompatible
entity pairs before classification — e.g., (PERSON, bornIn, LOC) is
valid but (DATE, bornIn, LOC) is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class EntityType(Enum):
    """Fine-grained entity type labels."""

    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    NUMBER = "NUMBER"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    WORK_OF_ART = "WORK_OF_ART"
    OTHER = "OTHER"


@dataclass
class TypedEntity:
    """An entity span with a predicted type.

    Attributes:
        start: Start token index (inclusive).
        end: End token index (exclusive).
        text: Surface text.
        entity_type: Predicted fine-grained type.
        type_scores: Score distribution over all types.
    """

    start: int
    end: int
    text: str
    entity_type: EntityType = EntityType.OTHER
    type_scores: dict[EntityType, float] = field(default_factory=dict)


# Type compatibility matrix: which (subject_type, object_type) pairs are
# valid for at least one relation.  Used to prune impossible pairs.
TYPE_COMPATIBILITY: dict[str, set[tuple[EntityType, EntityType]]] = {
    "bornIn": {(EntityType.PERSON, EntityType.LOCATION)},
    "headquarteredIn": {(EntityType.ORGANIZATION, EntityType.LOCATION)},
    "foundedBy": {(EntityType.ORGANIZATION, EntityType.PERSON)},
    "dateOfBirth": {(EntityType.PERSON, EntityType.DATE)},
    "memberOf": {
        (EntityType.PERSON, EntityType.ORGANIZATION),
        (EntityType.ORGANIZATION, EntityType.ORGANIZATION),
    },
    "locatedIn": {
        (EntityType.LOCATION, EntityType.LOCATION),
        (EntityType.ORGANIZATION, EntityType.LOCATION),
    },
    "creatorOf": {
        (EntityType.PERSON, EntityType.WORK_OF_ART),
        (EntityType.PERSON, EntityType.PRODUCT),
        (EntityType.ORGANIZATION, EntityType.PRODUCT),
    },
}


class EntityTyper:
    """Classify entity spans into fine-grained types.

    Uses a simple feed-forward network over span embeddings.  In a
    production system this would be a fine-tuned transformer head.

    Args:
        hidden_dim: Dimension of span embeddings.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        self.hidden_dim = hidden_dim
        self.types = list(EntityType)
        rng = np.random.default_rng(42)
        self.W = rng.standard_normal(
            (len(self.types), hidden_dim)
        ).astype(np.float32) * 0.1
        self.b = np.zeros(len(self.types), dtype=np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def type_span(
        self,
        span_embedding: np.ndarray,
        text: str = "",
        start: int = 0,
        end: int = 0,
    ) -> TypedEntity:
        """Predict the entity type for a single span.

        Args:
            span_embedding: Vector of shape (hidden_dim,).
            text: Surface text of the span.
            start: Start token index.
            end: End token index.

        Returns:
            TypedEntity with predicted type and score distribution.
        """
        logits = self.W @ span_embedding + self.b
        probs = self._softmax(logits)
        scores = {t: float(probs[i]) for i, t in enumerate(self.types)}
        best_type = self.types[int(np.argmax(probs))]
        return TypedEntity(
            start=start,
            end=end,
            text=text,
            entity_type=best_type,
            type_scores=scores,
        )

    def type_spans(
        self,
        tokens: list[str],
        spans: list[tuple[int, int]],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[TypedEntity]:
        """Type multiple entity spans.

        Args:
            tokens: Document tokens.
            spans: List of (start, end) index pairs.
            embeddings: Token embeddings (T, hidden_dim).

        Returns:
            List of TypedEntity objects.
        """
        n = len(tokens)
        if embeddings is None:
            rng = np.random.default_rng(0)
            embeddings = rng.standard_normal(
                (n, self.hidden_dim)
            ).astype(np.float32)

        results: list[TypedEntity] = []
        for start, end in spans:
            span_emb = embeddings[start:end].mean(axis=0)
            text = " ".join(tokens[start:end])
            results.append(self.type_span(span_emb, text, start, end))
        return results


def is_type_compatible(
    subj_type: EntityType,
    obj_type: EntityType,
    relation: Optional[str] = None,
) -> bool:
    """Check if a (subject_type, object_type) pair is compatible.

    Args:
        subj_type: Type of the subject entity.
        obj_type: Type of the object entity.
        relation: If given, check compatibility for this specific
            relation.  If None, check against all known relations.

    Returns:
        True if the pair is compatible with at least one relation.
    """
    pair = (subj_type, obj_type)
    if relation is not None:
        valid = TYPE_COMPATIBILITY.get(relation, set())
        return pair in valid

    for valid_pairs in TYPE_COMPATIBILITY.values():
        if pair in valid_pairs:
            return True
    return False


if __name__ == "__main__":
    tokens = "Albert Einstein was born in Ulm in 1879 .".split()
    spans = [(0, 2), (5, 6), (7, 8)]

    typer = EntityTyper(hidden_dim=32)
    typed = typer.type_spans(tokens, spans, embeddings=None)

    print("Entity typing results:")
    for te in typed:
        print(f"  '{te.text}' → {te.entity_type.value}  "
              f"(top score: {max(te.type_scores.values()):.3f})")

    print("\nType compatibility checks:")
    for rel, pairs in TYPE_COMPATIBILITY.items():
        for s, o in pairs:
            ok = is_type_compatible(s, o, rel)
            print(f"  ({s.value}, {rel}, {o.value}) → {ok}")
