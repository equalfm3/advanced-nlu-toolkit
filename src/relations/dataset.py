"""Relation extraction dataset with negative sampling.

Builds training examples from annotated sentences.  Most entity pairs in
a sentence have no relation (the "NA" class), so the dataset uses
type-constrained negative sampling to avoid overwhelming the model with
trivial negatives — only pairing entities whose types are compatible
with at least one relation schema.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from src.relations.typing import EntityType, TypedEntity, is_type_compatible


@dataclass
class RelationInstance:
    """A single relation extraction training example.

    Attributes:
        subject: Subject entity.
        obj: Object entity.
        relation: Gold relation label ("NA" for no relation).
        sentence_tokens: Tokenized sentence.
    """

    subject: TypedEntity
    obj: TypedEntity
    relation: str
    sentence_tokens: list[str] = field(default_factory=list)


@dataclass
class AnnotatedSentence:
    """A sentence with entity and relation annotations.

    Attributes:
        tokens: Tokenized sentence.
        entities: List of typed entities in the sentence.
        relations: List of (subj_idx, obj_idx, relation_label) triples
            indexing into the entities list.
    """

    tokens: list[str]
    entities: list[TypedEntity] = field(default_factory=list)
    relations: list[tuple[int, int, str]] = field(default_factory=list)


class RelationDataset:
    """Build relation extraction examples with negative sampling.

    For each annotated sentence, generates positive examples from gold
    relations and negative examples from type-compatible entity pairs
    that have no annotated relation.

    Args:
        neg_ratio: Maximum ratio of negative to positive examples.
        seed: Random seed for reproducibility.
    """

    def __init__(self, neg_ratio: float = 3.0, seed: int = 42) -> None:
        self.neg_ratio = neg_ratio
        self.rng = random.Random(seed)
        self.instances: list[RelationInstance] = []

    def build_from_sentence(
        self,
        sentence: AnnotatedSentence,
    ) -> list[RelationInstance]:
        """Generate positive and negative instances from one sentence.

        Args:
            sentence: Annotated sentence with entities and relations.

        Returns:
            List of RelationInstance objects.
        """
        gold_pairs: set[tuple[int, int]] = set()
        positives: list[RelationInstance] = []

        for subj_idx, obj_idx, rel in sentence.relations:
            gold_pairs.add((subj_idx, obj_idx))
            positives.append(
                RelationInstance(
                    subject=sentence.entities[subj_idx],
                    obj=sentence.entities[obj_idx],
                    relation=rel,
                    sentence_tokens=sentence.tokens,
                )
            )

        neg_candidates: list[RelationInstance] = []
        n_ents = len(sentence.entities)
        for i in range(n_ents):
            for j in range(n_ents):
                if i == j or (i, j) in gold_pairs:
                    continue
                subj = sentence.entities[i]
                obj = sentence.entities[j]
                if is_type_compatible(subj.entity_type, obj.entity_type):
                    neg_candidates.append(
                        RelationInstance(
                            subject=subj,
                            obj=obj,
                            relation="NA",
                            sentence_tokens=sentence.tokens,
                        )
                    )

        max_neg = max(1, int(len(positives) * self.neg_ratio))
        if len(neg_candidates) > max_neg:
            self.rng.shuffle(neg_candidates)
            neg_candidates = neg_candidates[:max_neg]

        instances = positives + neg_candidates
        self.instances.extend(instances)
        return instances

    def build_from_sentences(
        self,
        sentences: list[AnnotatedSentence],
    ) -> list[RelationInstance]:
        """Build dataset from multiple annotated sentences.

        Args:
            sentences: List of annotated sentences.

        Returns:
            All generated instances.
        """
        all_instances: list[RelationInstance] = []
        for sent in sentences:
            all_instances.extend(self.build_from_sentence(sent))
        return all_instances

    def statistics(self) -> dict[str, int]:
        """Return dataset statistics.

        Returns:
            Dict with counts of positive, negative, and total instances.
        """
        pos = sum(1 for inst in self.instances if inst.relation != "NA")
        neg = len(self.instances) - pos
        return {"positive": pos, "negative": neg, "total": len(self.instances)}


if __name__ == "__main__":
    sent = AnnotatedSentence(
        tokens="Albert Einstein was born in Ulm .".split(),
        entities=[
            TypedEntity(0, 2, "Albert Einstein", EntityType.PERSON),
            TypedEntity(5, 6, "Ulm", EntityType.LOCATION),
        ],
        relations=[(0, 1, "bornIn")],
    )

    dataset = RelationDataset(neg_ratio=2.0)
    instances = dataset.build_from_sentence(sent)

    print(f"Generated {len(instances)} instances from 1 sentence:")
    for inst in instances:
        print(f"  ({inst.subject.text}, {inst.relation}, {inst.obj.text})")

    stats = dataset.statistics()
    print(f"\nDataset stats: {stats}")
