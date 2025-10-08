"""Unit tests for chunking utilities."""

from retrivex.utils.chunking import SentenceChunker, SimpleChunker


class TestSimpleChunker:
    """Tests for SimpleChunker."""

    def test_basic_chunking(self) -> None:
        """Test basic chunking."""
        chunker = SimpleChunker(chunk_size=50, overlap=10)
        text = "A" * 1000  # Simple long text

        chunks = chunker.chunk_text(text, doc_id="doc1")

        assert len(chunks) > 0
        # Check monotonic chunk_ids
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_id == i
            assert chunk.metadata.doc_id == "doc1"

    def test_chunk_metadata(self) -> None:
        """Test that chunks have correct metadata."""
        chunker = SimpleChunker(chunk_size=50, overlap=10)
        text = "This is a test document with some content."

        chunks = chunker.chunk_text(
            text,
            doc_id="doc1",
            parent_id="parent1",
            heading_path=["Section 1", "Subsection A"],
        )

        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.metadata.doc_id == "doc1"
        assert chunk.metadata.parent_id == "parent1"
        assert chunk.metadata.heading_path == ["Section 1", "Subsection A"]
        assert chunk.metadata.char_start >= 0
        assert chunk.metadata.char_end > chunk.metadata.char_start

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        chunker = SimpleChunker()
        chunks = chunker.chunk_text("", doc_id="doc1")
        assert chunks == []

    def test_overlap(self) -> None:
        """Test that overlap works correctly."""
        chunker = SimpleChunker(chunk_size=20, overlap=5)
        text = "A" * 200

        chunks = chunker.chunk_text(text, doc_id="doc1")

        # Check overlap between consecutive chunks
        if len(chunks) > 1:
            # Second chunk should start before first chunk ends
            assert chunks[1].metadata.char_start < chunks[0].metadata.char_end


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_sentence_chunking(self) -> None:
        """Test sentence-based chunking."""
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=1)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk_text(text, doc_id="doc1")

        assert len(chunks) > 0
        # Check monotonic chunk_ids
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_id == i

    def test_sentence_splitting(self) -> None:
        """Test that sentences are split correctly."""
        chunker = SentenceChunker(sentences_per_chunk=1, overlap_sentences=0)
        text = "First! Second? Third."

        sentences = chunker._split_sentences(text)

        assert len(sentences) == 3
        assert "First!" in sentences[0]

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        chunker = SentenceChunker()
        chunks = chunker.chunk_text("", doc_id="doc1")
        assert chunks == []

    def test_sentence_overlap(self) -> None:
        """Test that sentence overlap works."""
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=1)
        text = "A. B. C. D. E."

        chunks = chunker.chunk_text(text, doc_id="doc1")

        # With 2 sentences per chunk and 1 overlap, we should get multiple chunks
        assert len(chunks) > 1
