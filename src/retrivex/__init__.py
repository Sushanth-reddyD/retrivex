"""
RetriVex: Lightweight neighbor-expansion retriever for long-context LLM applications.

Addresses the U-shaped positional bias in LLMs by intelligently expanding retrieved
chunks with their neighbors, scoring spans, and using edge-balanced ordering.
"""

__version__ = "0.1.0"

from .core.models import (
    Chunk,
    ChunkMetadata,
    OrderingStrategy,
    RetrievalConfig,
    SeedHit,
    Span,
)
from .core.retriever import SpanComposer

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "OrderingStrategy",
    "RetrievalConfig",
    "SeedHit",
    "Span",
    "SpanComposer",
]
