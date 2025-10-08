"""Core models and types for RetriVex."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderingStrategy(str, Enum):
    """Strategy for ordering spans in the final context."""

    FRONT_FIRST = "front_first"
    BACK_FIRST = "back_first"
    EDGE_BALANCED = "edge_balanced"  # Default: mitigates U-shaped bias
    SCORE_DESC = "score_desc"


@dataclass
class ChunkMetadata:
    """
    Metadata for a single chunk.

    Following §2.2: minimal metadata per chunk for neighbor lookups.
    """

    doc_id: str
    chunk_id: int  # Monotonic within doc
    char_start: int
    char_end: int
    parent_id: Optional[str] = None
    heading_path: Optional[List[str]] = None
    page: Optional[int] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chunk metadata."""
        if self.chunk_id < 0:
            raise ValueError(f"chunk_id must be non-negative, got {self.chunk_id}")
        if self.char_start < 0 or self.char_end < 0:
            raise ValueError("char_start and char_end must be non-negative")
        if self.char_start > self.char_end:
            raise ValueError(f"char_start ({self.char_start}) > char_end ({self.char_end})")


@dataclass
class Chunk:
    """A single text chunk with embeddings and metadata."""

    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Validate chunk."""
        if not self.text:
            raise ValueError("Chunk text cannot be empty")


@dataclass
class SeedHit:
    """A base kNN hit from vector search."""

    chunk: Chunk
    similarity_score: float  # Cosine similarity or equivalent

    def __post_init__(self) -> None:
        """Validate seed hit."""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(f"similarity_score must be in [0, 1], got {self.similarity_score}")


@dataclass
class Span:
    """
    A contiguous span of chunks with provenance.

    Following §4: group contiguous chunk_ids into spans per doc.
    """

    doc_id: str
    chunk_ids: List[int]  # Contiguous range
    chunks: List[Chunk]
    score: float
    char_start: int
    char_end: int
    token_count: int
    # Score components for transparency
    sim_score: float = 0.0
    adjacency_score: float = 0.0
    continuity_score: float = 0.0
    parent_bonus: float = 0.0

    def __post_init__(self) -> None:
        """Validate span."""
        if not self.chunks:
            raise ValueError("Span must contain at least one chunk")
        if len(self.chunk_ids) != len(self.chunks):
            raise ValueError("chunk_ids and chunks must have same length")
        # Verify contiguity
        sorted_ids = sorted(self.chunk_ids)
        if sorted_ids != list(range(sorted_ids[0], sorted_ids[-1] + 1)):
            raise ValueError(f"chunk_ids must be contiguous, got {self.chunk_ids}")

    @property
    def text(self) -> str:
        """Concatenate chunk texts."""
        return " ".join(chunk.text for chunk in self.chunks)

    @property
    def span_range(self) -> tuple[int, int]:
        """Return (start_chunk_id, end_chunk_id) inclusive."""
        return (min(self.chunk_ids), max(self.chunk_ids))


@dataclass
class RetrievalConfig:
    """
    Configuration for the retrieval procedure (§4, §5).

    Default knobs from §4:
    - L = 2 (neighbor window)
    - k = 6-8 (base hits)
    - β = 0.7 (distance decay)
    - weights: sim:1.0, adj:0.6, cont:0.2, parent:0.1
    """

    # Core parameters
    window: int = 2  # L: ±L neighbors around each seed
    k: int = 6  # Number of base kNN hits
    token_budget: int = 2000  # Max tokens in final context

    # Scoring weights
    similarity_weight: float = 1.0  # w0
    adjacency_weight: float = 0.6  # w1
    continuity_weight: float = 0.2  # w2
    parent_weight: float = 0.1  # w3

    # Scoring parameters
    distance_decay_beta: float = 0.7  # β: decay with distance
    min_continuity: float = 0.5  # Minimum continuity to cross boundaries

    # Constraints
    max_spans: Optional[int] = None  # Limit number of spans
    max_parent_chars: int = 2000  # Trim large parents
    stop_on_heading_change: bool = True  # Stop expansion at heading boundaries

    # Ordering
    ordering_strategy: OrderingStrategy = OrderingStrategy.EDGE_BALANCED

    # De-duplication policy
    dedupe_overlaps: bool = True  # Favor longer, higher-score spans
    prefer_fewer_spans: bool = True  # Tie-breaker: fewer, longer spans

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.window < 0:
            raise ValueError(f"window must be non-negative, got {self.window}")
        if self.k < 1:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.token_budget < 1:
            raise ValueError(f"token_budget must be positive, got {self.token_budget}")
        if self.distance_decay_beta < 0:
            raise ValueError(
                f"distance_decay_beta must be non-negative, got {self.distance_decay_beta}"
            )
        if not 0.0 <= self.min_continuity <= 1.0:
            raise ValueError(f"min_continuity must be in [0, 1], got {self.min_continuity}")
