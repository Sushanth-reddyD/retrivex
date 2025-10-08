"""Unit tests for span composer and retriever."""

import pytest

from retrivex.core.models import Chunk, ChunkMetadata, RetrievalConfig, SeedHit, OrderingStrategy
from retrivex.core.retriever import SpanComposer


class TestSpanComposer:
    """Tests for SpanComposer."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        chunks = {}
        for i in range(10):
            metadata = ChunkMetadata(
                doc_id="doc1",
                chunk_id=i,
                char_start=i * 100,
                char_end=(i + 1) * 100,
            )
            chunk = Chunk(
                text=f"This is chunk {i} with some text content.",
                metadata=metadata,
                embedding=[0.1 * i] * 10,  # Simple embedding
            )
            chunks[("doc1", i)] = chunk
        return chunks

    @pytest.fixture
    def composer(self):
        """Create a span composer with default config."""
        return SpanComposer(RetrievalConfig())

    def test_neighbor_expansion_basic(self, composer, sample_chunks):
        """Test basic neighbor expansion."""
        # Create seed hit at chunk 5
        seed = SeedHit(chunk=sample_chunks[("doc1", 5)], similarity_score=0.9)

        expanded = composer._expand_neighbors([seed], sample_chunks)

        # With window=2, should get chunks 3, 4, 5, 6, 7
        assert "doc1" in expanded
        expected_ids = {3, 4, 5, 6, 7}
        assert expanded["doc1"] == expected_ids

    def test_neighbor_expansion_boundary(self, composer, sample_chunks):
        """Test neighbor expansion at document boundaries."""
        # Seed at chunk 0 (start boundary)
        seed = SeedHit(chunk=sample_chunks[("doc1", 0)], similarity_score=0.9)

        expanded = composer._expand_neighbors([seed], sample_chunks)

        # With window=2, should get chunks 0, 1, 2 (no negative IDs)
        assert expanded["doc1"] == {0, 1, 2}

    def test_span_composition(self, composer, sample_chunks):
        """Test span composition from expanded chunks."""
        # Expanded chunks with a gap
        expanded = {"doc1": {0, 1, 2, 5, 6, 7}}

        seed = SeedHit(chunk=sample_chunks[("doc1", 1)], similarity_score=0.9)
        spans = composer._compose_spans(expanded, [seed], sample_chunks)

        # Should create two spans: [0,1,2] and [5,6,7]
        assert len(spans) == 2

        span1, span2 = sorted(spans, key=lambda s: s.chunk_ids[0])
        assert span1.chunk_ids == [0, 1, 2]
        assert span2.chunk_ids == [5, 6, 7]

    def test_span_scoring(self, composer, sample_chunks):
        """Test span scoring."""
        # Create a simple span
        span = composer._create_span("doc1", [5, 6, 7], sample_chunks)

        # Create seed hit
        seed = SeedHit(chunk=sample_chunks[("doc1", 5)], similarity_score=0.95)

        # Score the span
        scored_span = composer._score_span(span, [seed])

        # Check that score components are computed
        assert scored_span.sim_score == 0.95
        assert scored_span.adjacency_score > 0
        assert scored_span.score > 0

    def test_budget_selection(self, composer, sample_chunks):
        """Test span selection within budget."""
        # Create spans with different scores and token counts
        span1 = composer._create_span("doc1", [0, 1], sample_chunks)
        span1.score = 0.9
        span1.token_count = 50

        span2 = composer._create_span("doc1", [3, 4], sample_chunks)
        span2.score = 0.8
        span2.token_count = 50

        span3 = composer._create_span("doc1", [6, 7], sample_chunks)
        span3.score = 0.7
        span3.token_count = 50

        # Set budget to allow only 2 spans
        composer.config.token_budget = 100

        selected = composer._select_within_budget([span1, span2, span3])

        # Should select top 2 by score
        assert len(selected) == 2
        assert span1 in selected
        assert span2 in selected

    def test_edge_balanced_ordering(self, composer, sample_chunks):
        """Test edge-balanced ordering."""
        # Create multiple spans
        spans = []
        for i, chunk_ids in enumerate([[0, 1], [3, 4], [6, 7], [9]]):
            span = composer._create_span("doc1", chunk_ids, sample_chunks)
            span.score = 1.0 - i * 0.1  # Decreasing scores
            spans.append(span)

        ordered = composer._edge_balanced_ordering(spans)

        # Should alternate: highest to front, second to back, etc.
        # Result: [span0, span2, span3, span1]
        assert len(ordered) == 4
        assert ordered[0] == spans[0]  # Highest score to front
        assert ordered[-1] == spans[1]  # Second highest to back

    def test_deduplication(self, composer, sample_chunks):
        """Test span deduplication."""
        # Create overlapping spans
        span1 = composer._create_span("doc1", [0, 1, 2], sample_chunks)
        span1.score = 0.9

        span2 = composer._create_span("doc1", [2, 3, 4], sample_chunks)
        span2.score = 0.8

        span3 = composer._create_span("doc1", [6, 7], sample_chunks)
        span3.score = 0.7

        deduplicated = composer._deduplicate_spans([span1, span2, span3])

        # Should keep span1 and span3 (no overlap), drop span2 (overlaps with span1)
        assert len(deduplicated) == 2
        assert span1 in deduplicated
        assert span3 in deduplicated

    def test_full_retrieval_pipeline(self, composer, sample_chunks):
        """Test full retrieval pipeline end-to-end."""
        # Create seed hits
        seeds = [
            SeedHit(chunk=sample_chunks[("doc1", 2)], similarity_score=0.95),
            SeedHit(chunk=sample_chunks[("doc1", 7)], similarity_score=0.85),
        ]

        # Run retrieval
        result = composer.retrieve(seeds, sample_chunks)

        # Should return ordered spans
        assert len(result) > 0
        assert all(isinstance(span, type(result[0])) for span in result)

        # Verify all spans are within budget
        total_tokens = sum(span.token_count for span in result)
        assert total_tokens <= composer.config.token_budget

    def test_ordering_strategies(self, sample_chunks):
        """Test different ordering strategies."""
        spans = []
        for i, chunk_ids in enumerate([[0, 1], [3, 4], [6, 7]]):
            composer = SpanComposer(RetrievalConfig())
            span = composer._create_span("doc1", chunk_ids, sample_chunks)
            span.score = 1.0 - i * 0.1
            spans.append(span)

        # Test SCORE_DESC
        config = RetrievalConfig(ordering_strategy=OrderingStrategy.SCORE_DESC)
        composer = SpanComposer(config)
        ordered = composer._order_spans(spans.copy())
        assert ordered[0].score > ordered[1].score > ordered[2].score

        # Test FRONT_FIRST
        config = RetrievalConfig(ordering_strategy=OrderingStrategy.FRONT_FIRST)
        composer = SpanComposer(config)
        ordered = composer._order_spans(spans.copy())
        assert ordered == spans

        # Test BACK_FIRST
        config = RetrievalConfig(ordering_strategy=OrderingStrategy.BACK_FIRST)
        composer = SpanComposer(config)
        ordered = composer._order_spans(spans.copy())
        assert ordered == list(reversed(spans))
