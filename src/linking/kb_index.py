"""Knowledge base entity index.

Stores entity records with embeddings, descriptions, and alias mappings.
Supports lookup by entity ID, alias string, and nearest-neighbor search
over entity embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class KBEntity:
    """A knowledge base entity record.

    Attributes:
        entity_id: Unique identifier (e.g., Wikidata QID).
        name: Canonical entity name.
        description: Short entity description.
        aliases: Alternative surface forms.
        entity_type: Coarse type label.
        embedding: Dense entity embedding vector.
        popularity: Prior popularity score (e.g., from page views).
    """

    entity_id: str
    name: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    entity_type: str = ""
    embedding: Optional[np.ndarray] = None
    popularity: float = 0.0


class KBIndex:
    """In-memory knowledge base index.

    Provides fast lookup by entity ID and alias string, plus
    approximate nearest-neighbor search over entity embeddings.

    Args:
        embedding_dim: Dimension of entity embeddings.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        self.embedding_dim = embedding_dim
        self._entities: dict[str, KBEntity] = {}
        self._alias_map: dict[str, list[str]] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._id_list: list[str] = []
        self._dirty = True

    @property
    def size(self) -> int:
        """Number of entities in the index."""
        return len(self._entities)

    def add_entity(self, entity: KBEntity) -> None:
        """Add or update an entity in the index.

        Args:
            entity: Entity record to add.
        """
        self._entities[entity.entity_id] = entity
        all_names = [entity.name.lower()] + [a.lower() for a in entity.aliases]
        for alias in all_names:
            self._alias_map.setdefault(alias, [])
            if entity.entity_id not in self._alias_map[alias]:
                self._alias_map[alias].append(entity.entity_id)
        self._dirty = True

    def get_entity(self, entity_id: str) -> Optional[KBEntity]:
        """Retrieve an entity by ID.

        Args:
            entity_id: Entity identifier.

        Returns:
            KBEntity or None if not found.
        """
        return self._entities.get(entity_id)

    def lookup_alias(self, alias: str) -> list[KBEntity]:
        """Find entities matching an alias string.

        Args:
            alias: Surface form to look up (case-insensitive).

        Returns:
            List of matching entities.
        """
        ids = self._alias_map.get(alias.lower(), [])
        return [self._entities[eid] for eid in ids if eid in self._entities]

    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild the dense embedding matrix for nearest-neighbor search."""
        self._id_list = []
        vecs: list[np.ndarray] = []
        for eid, entity in self._entities.items():
            if entity.embedding is not None:
                self._id_list.append(eid)
                vecs.append(entity.embedding)
        if vecs:
            self._embeddings = np.stack(vecs)
        else:
            self._embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
        self._dirty = False

    def nearest_neighbors(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[KBEntity, float]]:
        """Find nearest entities by cosine similarity.

        Args:
            query_embedding: Query vector of shape (embedding_dim,).
            top_k: Number of results.

        Returns:
            List of (entity, similarity) pairs sorted by descending
            similarity.
        """
        if self._dirty:
            self._rebuild_embedding_matrix()

        if self._embeddings is None or len(self._id_list) == 0:
            return []

        query_norm = np.linalg.norm(query_embedding)
        if query_norm < 1e-8:
            return []

        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = self._embeddings / norms
        query_normalized = query_embedding / query_norm

        similarities = normalized @ query_normalized
        k = min(top_k, len(self._id_list))
        top_indices = np.argsort(similarities)[::-1][:k]

        results: list[tuple[KBEntity, float]] = []
        for idx in top_indices:
            eid = self._id_list[idx]
            entity = self._entities[eid]
            results.append((entity, float(similarities[idx])))
        return results


def build_sample_kb(embedding_dim: int = 64) -> KBIndex:
    """Build a small sample knowledge base for demos.

    Args:
        embedding_dim: Dimension of entity embeddings.

    Returns:
        Populated KBIndex.
    """
    rng = np.random.default_rng(42)
    kb = KBIndex(embedding_dim=embedding_dim)

    entities = [
        KBEntity("Q7186", "Marie Curie",
                 "Polish-French physicist and chemist",
                 ["Maria Sklodowska", "Madame Curie"],
                 "PERSON", popularity=0.95),
        KBEntity("Q1107", "Radium",
                 "Chemical element with symbol Ra",
                 ["Ra", "radium"], "CHEMICAL", popularity=0.6),
        KBEntity("Q61", "Washington, D.C.",
                 "Capital of the United States",
                 ["Washington", "DC", "District of Columbia"],
                 "LOCATION", popularity=0.9),
        KBEntity("Q23", "George Washington",
                 "First president of the United States",
                 ["Washington", "President Washington"],
                 "PERSON", popularity=0.85),
        KBEntity("Q1223", "Washington (state)",
                 "State in the Pacific Northwest",
                 ["Washington", "Washington State", "WA"],
                 "LOCATION", popularity=0.7),
    ]

    for entity in entities:
        entity.embedding = rng.standard_normal(embedding_dim).astype(np.float32)
        kb.add_entity(entity)

    return kb


if __name__ == "__main__":
    kb = build_sample_kb(embedding_dim=32)
    print(f"Knowledge base: {kb.size} entities\n")

    matches = kb.lookup_alias("Washington")
    print(f"Alias lookup 'Washington' → {len(matches)} match(es):")
    for e in matches:
        print(f"  {e.entity_id}: {e.name} ({e.entity_type})")

    print()
    entity = kb.get_entity("Q7186")
    if entity and entity.embedding is not None:
        neighbors = kb.nearest_neighbors(entity.embedding, top_k=3)
        print(f"Nearest neighbors to '{entity.name}':")
        for e, sim in neighbors:
            print(f"  {e.entity_id}: {e.name}  sim={sim:.4f}")
