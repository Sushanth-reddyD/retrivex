"""Core package for RetriVex."""

from .models import (
    Chunk,
    ChunkMetadata,
    OrderingStrategy,
    RetrievalConfig,
    SeedHit,
    Span,
)
from .retriever import SpanComposer

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "OrderingStrategy",
    "RetrievalConfig",
    "SeedHit",
    "Span",
    "SpanComposer",
]
