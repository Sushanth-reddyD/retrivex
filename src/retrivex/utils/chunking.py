"""Chunking utilities following §2 requirements."""

import re
from typing import List, Optional

from ..core.models import Chunk, ChunkMetadata


class SimpleChunker:
    """
    Simple fixed-size chunker with overlap.

    Following §2.1: Fixed-size with stable ordering, preserving document order
    for neighbor lookups.
    """

    def __init__(
        self,
        chunk_size: int = 200,  # tokens (roughly)
        overlap: int = 50,
        chars_per_token: int = 4,  # Rough estimate
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            chars_per_token: Character-to-token ratio (rough)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chars_per_token = chars_per_token
        self.chunk_chars = chunk_size * chars_per_token
        self.overlap_chars = overlap * chars_per_token

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        parent_id: Optional[str] = None,
        heading_path: Optional[List[str]] = None,
    ) -> List[Chunk]:
        """
        Chunk text into fixed-size chunks with overlap.

        Args:
            text: Text to chunk
            doc_id: Document identifier
            parent_id: Optional parent document/section ID
            heading_path: Optional heading hierarchy

        Returns:
            List of Chunk objects with monotonic chunk_ids
        """
        if not text:
            return []

        chunks = []
        chunk_id = 0
        start = 0

        while start < len(text):
            end = min(start + self.chunk_chars, len(text))

            # Try to break at sentence or word boundary
            if end < len(text):
                end = self._find_boundary(text, end)

            chunk_text = text[start:end].strip()

            if chunk_text:
                metadata = ChunkMetadata(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    char_start=start,
                    char_end=end,
                    parent_id=parent_id,
                    heading_path=heading_path,
                )

                chunks.append(Chunk(text=chunk_text, metadata=metadata))
                chunk_id += 1

            # Move to next chunk with overlap
            start = end - self.overlap_chars
            if start <= 0 or end >= len(text):
                break

        return chunks

    def _find_boundary(self, text: str, pos: int) -> int:
        """
        Find a good boundary near pos (sentence or word boundary).

        Looks backward up to 100 chars for '.', '!', '?', or whitespace.
        """
        search_start = max(0, pos - 100)

        # Try to find sentence boundary
        for delim in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            idx = text.rfind(delim, search_start, pos)
            if idx != -1:
                return idx + len(delim)

        # Fall back to word boundary
        idx = text.rfind(" ", search_start, pos)
        if idx != -1:
            return idx + 1

        # No good boundary found
        return pos


class SentenceChunker:
    """
    Sentence-based chunker with windowing.

    Following §2.3: sentence window nodes with prev/next relations.
    """

    def __init__(
        self,
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 1,
    ):
        """
        Initialize sentence chunker.

        Args:
            sentences_per_chunk: Number of sentences per chunk
            overlap_sentences: Number of overlapping sentences
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        parent_id: Optional[str] = None,
        heading_path: Optional[List[str]] = None,
    ) -> List[Chunk]:
        """
        Chunk text by sentences.

        Args:
            text: Text to chunk
            doc_id: Document identifier
            parent_id: Optional parent document/section ID
            heading_path: Optional heading hierarchy

        Returns:
            List of Chunk objects with monotonic chunk_ids
        """
        if not text:
            return []

        # Simple sentence splitting (can be enhanced with NLTK/spaCy)
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        chunk_id = 0
        i = 0

        # Precompute sentence spans to get accurate character offsets
        sentence_spans = []
        cursor = 0
        for sentence in sentences:
            idx = text.find(sentence, cursor)
            if idx == -1:
                idx = cursor
            start_idx = idx
            end_idx = start_idx + len(sentence)
            sentence_spans.append((sentence, start_idx, end_idx))
            cursor = end_idx

        chunks = []
        chunk_id = 0
        i = 0

        while i < len(sentences):
            # Take sentences_per_chunk sentences
            end = min(i + self.sentences_per_chunk, len(sentences))
            window = sentence_spans[i:end]
            chunk_start = window[0][1]
            chunk_end = window[-1][2]
            chunk_text = text[chunk_start:chunk_end]

            if chunk_text:
                metadata = ChunkMetadata(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    char_start=chunk_start,
                    char_end=chunk_end,
                    parent_id=parent_id,
                    heading_path=heading_path,
                )

                chunks.append(Chunk(text=chunk_text, metadata=metadata))
                chunk_id += 1

            # Move forward with overlap
            i += self.sentences_per_chunk - self.overlap_sentences
            if i <= 0 or end >= len(sentences):
                break
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter.

        Can be enhanced with nltk.sent_tokenize or spaCy.
        """
        # Simple rule-based splitting
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Filter empty
        return [s.strip() for s in sentences if s.strip()]
