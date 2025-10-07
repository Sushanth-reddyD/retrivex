"""Unit tests for core models."""

import pytest

from retrivex.core.models import (
    Chunk,
    ChunkMetadata,
    OrderingStrategy,
    RetrievalConfig,
    SeedHit,
    Span,
)


class TestChunkMetadata:
    """Tests for ChunkMetadata."""

    def test_valid_metadata(self) -> None:
        """Test creating valid metadata."""
        metadata = ChunkMetadata(
            doc_id="doc1",
            chunk_id=0,
            char_start=0,
            char_end=100,
        )
        assert metadata.doc_id == "doc1"
        assert metadata.chunk_id == 0
        assert metadata.char_start == 0
        assert metadata.char_end == 100

    def test_negative_chunk_id(self) -> None:
        """Test that negative chunk_id raises error."""
        with pytest.raises(ValueError, match="chunk_id must be non-negative"):
            ChunkMetadata(
                doc_id="doc1",
                chunk_id=-1,
                char_start=0,
                char_end=100,
            )

    def test_invalid_char_range(self) -> None:
        """Test that invalid char range raises error."""
        with pytest.raises(ValueError, match="char_start.*>.*char_end"):
            ChunkMetadata(
                doc_id="doc1",
                chunk_id=0,
                char_start=100,
                char_end=50,
            )


class TestChunk:
    """Tests for Chunk."""

    def test_valid_chunk(self) -> None:
        """Test creating valid chunk."""
        metadata = ChunkMetadata(
            doc_id="doc1",
            chunk_id=0,
            char_start=0,
            char_end=100,
        )
        chunk = Chunk(text="Sample text", metadata=metadata)
        assert chunk.text == "Sample text"
        assert chunk.metadata.doc_id == "doc1"

    def test_empty_text(self) -> None:
        """Test that empty text raises error."""
        metadata = ChunkMetadata(
            doc_id="doc1",
            chunk_id=0,
            char_start=0,
            char_end=100,
        )
        with pytest.raises(ValueError, match="Chunk text cannot be empty"):
            Chunk(text="", metadata=metadata)


class TestSeedHit:
    """Tests for SeedHit."""

    def test_valid_seed_hit(self) -> None:
        """Test creating valid seed hit."""
        metadata = ChunkMetadata(
            doc_id="doc1",
            chunk_id=0,
            char_start=0,
            char_end=100,
        )
        chunk = Chunk(text="Sample text", metadata=metadata)
        hit = SeedHit(chunk=chunk, similarity_score=0.95)
        assert hit.similarity_score == 0.95

    def test_invalid_similarity_score(self) -> None:
        """Test that invalid similarity score raises error."""
        metadata = ChunkMetadata(
            doc_id="doc1",
            chunk_id=0,
            char_start=0,
            char_end=100,
        )
        chunk = Chunk(text="Sample text", metadata=metadata)
        with pytest.raises(ValueError, match="similarity_score must be in"):
            SeedHit(chunk=chunk, similarity_score=1.5)


class TestSpan:
    """Tests for Span."""

    def test_valid_span(self) -> None:
        """Test creating valid span."""
        chunks = [
            Chunk(
                text=f"Chunk {i}",
                metadata=ChunkMetadata(
                    doc_id="doc1",
                    chunk_id=i,
                    char_start=i * 100,
                    char_end=(i + 1) * 100,
                ),
            )
            for i in range(3)
        ]

        span = Span(
            doc_id="doc1",
            chunk_ids=[0, 1, 2],
            chunks=chunks,
            score=0.9,
            char_start=0,
            char_end=300,
            token_count=75,
        )

        assert span.doc_id == "doc1"
        assert span.chunk_ids == [0, 1, 2]
        assert len(span.chunks) == 3
        assert span.text == "Chunk 0 Chunk 1 Chunk 2"
        assert span.span_range == (0, 2)

    def test_non_contiguous_chunk_ids(self) -> None:
        """Test that non-contiguous chunk_ids raise error."""
        chunks = [
            Chunk(
                text=f"Chunk {i}",
                metadata=ChunkMetadata(
                    doc_id="doc1",
                    chunk_id=i,
                    char_start=i * 100,
                    char_end=(i + 1) * 100,
                ),
            )
            for i in [0, 2]  # Gap at 1
        ]

        with pytest.raises(ValueError, match="chunk_ids must be contiguous"):
            Span(
                doc_id="doc1",
                chunk_ids=[0, 2],
                chunks=chunks,
                score=0.9,
                char_start=0,
                char_end=300,
                token_count=75,
            )

    def test_empty_span(self) -> None:
        """Test that empty span raises error."""
        with pytest.raises(ValueError, match="Span must contain at least one chunk"):
            Span(
                doc_id="doc1",
                chunk_ids=[],
                chunks=[],
                score=0.9,
                char_start=0,
                char_end=100,
                token_count=25,
            )


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = RetrievalConfig()
        assert config.window == 2
        assert config.k == 6
        assert config.token_budget == 2000
        assert config.similarity_weight == 1.0
        assert config.adjacency_weight == 0.6
        assert config.continuity_weight == 0.2
        assert config.parent_weight == 0.1
        assert config.distance_decay_beta == 0.7
        assert config.ordering_strategy == OrderingStrategy.EDGE_BALANCED

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RetrievalConfig(
            window=3,
            k=10,
            token_budget=4000,
            ordering_strategy=OrderingStrategy.SCORE_DESC,
        )
        assert config.window == 3
        assert config.k == 10
        assert config.token_budget == 4000
        assert config.ordering_strategy == OrderingStrategy.SCORE_DESC

    def test_invalid_window(self) -> None:
        """Test that negative window raises error."""
        with pytest.raises(ValueError, match="window must be non-negative"):
            RetrievalConfig(window=-1)

    def test_invalid_k(self) -> None:
        """Test that non-positive k raises error."""
        with pytest.raises(ValueError, match="k must be positive"):
            RetrievalConfig(k=0)

    def test_invalid_token_budget(self) -> None:
        """Test that non-positive token_budget raises error."""
        with pytest.raises(ValueError, match="token_budget must be positive"):
            RetrievalConfig(token_budget=0)
