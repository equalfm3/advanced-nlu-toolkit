"""Predicate-argument frame construction for SRL.

Combines predicate detection and argument labeling into complete
predicate-argument frames.  Each frame captures "who did what to whom"
for a single predicate in the sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.srl.predicate_detector import Predicate, PredicateDetector
from src.srl.labeler import ArgumentLabeler, ArgumentSpan


@dataclass
class SemanticFrame:
    """A predicate-argument frame.

    Attributes:
        predicate: The predicate (verb) anchoring this frame.
        arguments: Labeled argument spans.
        sentence_tokens: Original sentence tokens.
    """

    predicate: Predicate
    arguments: list[ArgumentSpan] = field(default_factory=list)
    sentence_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert frame to a serializable dictionary."""
        return {
            "predicate": {
                "text": self.predicate.text,
                "index": self.predicate.index,
                "sense": self.predicate.sense,
            },
            "arguments": [
                {
                    "role": arg.label,
                    "text": arg.text,
                    "span": [arg.start, arg.end],
                }
                for arg in self.arguments
            ],
        }

    def summary(self) -> str:
        """One-line summary of the frame."""
        parts: list[str] = []
        for arg in sorted(self.arguments, key=lambda a: a.start):
            parts.append(f"{arg.label}='{arg.text}'")
        args_str = ", ".join(parts) if parts else "(no arguments)"
        return f"{self.predicate.text}({self.predicate.sense}): {args_str}"


class FrameBuilder:
    """Build predicate-argument frames for a sentence.

    Orchestrates predicate detection and argument labeling to produce
    complete semantic frames.

    Args:
        hidden_dim: Token embedding dimension.
        predicate_threshold: Minimum confidence for predicate detection.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        predicate_threshold: float = 0.3,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.detector = PredicateDetector(
            hidden_dim=hidden_dim, threshold=predicate_threshold
        )
        self.labeler = ArgumentLabeler(hidden_dim=hidden_dim)

    def build_frames(
        self,
        tokens: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[SemanticFrame]:
        """Build all predicate-argument frames for a sentence.

        Args:
            tokens: Sentence tokens.
            embeddings: Token embeddings (T, hidden_dim).

        Returns:
            List of SemanticFrame objects, one per detected predicate.
        """
        n = len(tokens)
        if n == 0:
            return []

        if embeddings is None:
            rng = np.random.default_rng(0)
            embeddings = rng.standard_normal(
                (n, self.hidden_dim)
            ).astype(np.float32)

        predicates = self.detector.detect(tokens, embeddings)
        frames: list[SemanticFrame] = []

        for pred in predicates:
            arguments = self.labeler.label(tokens, pred, embeddings)
            frame = SemanticFrame(
                predicate=pred,
                arguments=arguments,
                sentence_tokens=tokens,
            )
            frames.append(frame)

        return frames

    def build_frames_for_predicates(
        self,
        tokens: list[str],
        predicates: list[Predicate],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[SemanticFrame]:
        """Build frames for pre-identified predicates.

        Args:
            tokens: Sentence tokens.
            predicates: Pre-detected predicates.
            embeddings: Token embeddings.

        Returns:
            List of SemanticFrame objects.
        """
        n = len(tokens)
        if embeddings is None:
            rng = np.random.default_rng(0)
            embeddings = rng.standard_normal(
                (n, self.hidden_dim)
            ).astype(np.float32)

        frames: list[SemanticFrame] = []
        for pred in predicates:
            arguments = self.labeler.label(tokens, pred, embeddings)
            frames.append(
                SemanticFrame(
                    predicate=pred,
                    arguments=arguments,
                    sentence_tokens=tokens,
                )
            )
        return frames


if __name__ == "__main__":
    tokens = "The scientist discovered a new element in her laboratory .".split()
    builder = FrameBuilder(hidden_dim=32, predicate_threshold=0.0)

    frames = builder.build_frames(tokens)
    print(f"Sentence: {' '.join(tokens)}")
    print(f"\nFound {len(frames)} frame(s):\n")

    for i, frame in enumerate(frames):
        print(f"Frame {i + 1}: {frame.summary()}")
        print(f"  Predicate: '{frame.predicate.text}' "
              f"[{frame.predicate.index}] ({frame.predicate.sense})")
        for arg in frame.arguments:
            print(f"  {arg.label:12s}: '{arg.text}' [{arg.start}:{arg.end}]")
        print()

    print("Serialized (first frame):")
    if frames:
        import json
        print(json.dumps(frames[0].to_dict(), indent=2))
