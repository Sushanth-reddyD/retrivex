"""Core retrieval logic implementing neighbor expansion and span composition."""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .models import Chunk, OrderingStrategy, RetrievalConfig, SeedHit, Span


class SpanComposer:
    """
    Composes and scores spans from seed hits following §4.

    Steps:
    1. Neighbor sweep: pull chunks [chunk_id-L, ..., chunk_id+L]
    2. Parent fetch (optional)
    3. Span compose: group contiguous chunk_ids into spans per doc
    4. Score each span: sim + adj + cont + parent_bonus
    5. Budget & merge: greedy selection until token budget
    6. Order spans: edge-balanced ordering (§4)
    """

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def retrieve(
        self,
        seed_hits: List[SeedHit],
        chunk_store: Dict[Tuple[str, int], Chunk],
        parent_store: Optional[Dict[str, str]] = None,
    ) -> List[Span]:
        """
        Main retrieval procedure following §4.

        Args:
            seed_hits: Base kNN hits with similarity scores
            chunk_store: Mapping from (doc_id, chunk_id) to Chunk
            parent_store: Optional mapping from parent_id to parent text

        Returns:
            Ordered list of spans to include in context
        """
        # Step 1: Neighbor sweep
        expanded_chunks = self._expand_neighbors(seed_hits, chunk_store)

        # Step 2 & 3: Group into spans per doc
        spans = self._compose_spans(expanded_chunks, seed_hits, chunk_store)

        # Step 4: Score spans
        scored_spans = [self._score_span(span, seed_hits) for span in spans]

        # Step 5: Budget & merge
        selected_spans = self._select_within_budget(scored_spans)

        # Step 6: Order for prompt
        ordered_spans = self._order_spans(selected_spans)

        return ordered_spans

    def _expand_neighbors(
        self,
        seed_hits: List[SeedHit],
        chunk_store: Dict[Tuple[str, int], Chunk],
    ) -> Dict[str, Set[int]]:
        """
        Expand each seed hit to include ±L neighbors (§4 step 2).

        Returns:
            Mapping from doc_id to set of chunk_ids to include
        """
        expanded: Dict[str, Set[int]] = defaultdict(set)

        for hit in seed_hits:
            doc_id = hit.chunk.metadata.doc_id
            chunk_id = hit.chunk.metadata.chunk_id

            # Always include the seed
            expanded[doc_id].add(chunk_id)

            # Expand neighbors
            for delta in range(1, self.config.window + 1):
                # Check backward
                neighbor_id = chunk_id - delta
                if self._can_expand_to(hit.chunk, chunk_id, neighbor_id, chunk_store):
                    expanded[doc_id].add(neighbor_id)

                # Check forward
                neighbor_id = chunk_id + delta
                if self._can_expand_to(hit.chunk, chunk_id, neighbor_id, chunk_store):
                    expanded[doc_id].add(neighbor_id)

        return expanded

    def _can_expand_to(
        self,
        seed_chunk: Chunk,
        seed_id: int,
        neighbor_id: int,
        chunk_store: Dict[Tuple[str, int], Chunk],
    ) -> bool:
        """Check if we can expand to a neighbor chunk."""
        doc_id = seed_chunk.metadata.doc_id
        key = (doc_id, neighbor_id)

        if key not in chunk_store:
            return False

        neighbor = chunk_store[key]

        # Check heading boundary if configured
        if self.config.stop_on_heading_change:
            seed_heading = seed_chunk.metadata.heading_path
            neighbor_heading = neighbor.metadata.heading_path

            if seed_heading != neighbor_heading:
                # Different headings - check continuity
                continuity = self._compute_continuity(seed_chunk, neighbor)
                if continuity < self.config.min_continuity:
                    return False

        return True

    def _compose_spans(
        self,
        expanded_chunks: Dict[str, Set[int]],
        seed_hits: List[SeedHit],
        chunk_store: Dict[Tuple[str, int], Chunk],
    ) -> List[Span]:
        """
        Group contiguous chunk_ids into spans per doc (§4 step 3).

        Returns:
            List of Span objects
        """
        spans = []

        for doc_id, chunk_ids in expanded_chunks.items():
            # Sort and find contiguous ranges
            sorted_ids = sorted(chunk_ids)

            # Group into contiguous spans
            current_span_ids = [sorted_ids[0]]

            for chunk_id in sorted_ids[1:]:
                if chunk_id == current_span_ids[-1] + 1:
                    # Contiguous
                    current_span_ids.append(chunk_id)
                else:
                    # Gap - finalize current span and start new one
                    spans.append(self._create_span(doc_id, current_span_ids, chunk_store))
                    current_span_ids = [chunk_id]

            # Finalize last span
            if current_span_ids:
                spans.append(self._create_span(doc_id, current_span_ids, chunk_store))

        return spans

    def _create_span(
        self,
        doc_id: str,
        chunk_ids: List[int],
        chunk_store: Dict[Tuple[str, int], Chunk],
    ) -> Span:
        """Create a Span object from chunk IDs."""
        chunks = [chunk_store[(doc_id, cid)] for cid in chunk_ids]

        # Compute character range
        char_start = min(c.metadata.char_start for c in chunks)
        char_end = max(c.metadata.char_end for c in chunks)

        # Estimate token count (rough: 4 chars per token)
        token_count = sum(len(c.text) for c in chunks) // 4

        return Span(
            doc_id=doc_id,
            chunk_ids=chunk_ids,
            chunks=chunks,
            score=0.0,  # Will be computed later
            char_start=char_start,
            char_end=char_end,
            token_count=token_count,
        )

    def _score_span(self, span: Span, seed_hits: List[SeedHit]) -> Span:
        """
        Score a span following §4 step 4.

        score = w0·sim + w1·adj + w2·cont + w3·parent_bonus
        """
        # Similarity score: max (or mean) of seed scores inside span
        span_chunk_ids = set(span.chunk_ids)
        seed_scores = [
            hit.similarity_score
            for hit in seed_hits
            if hit.chunk.metadata.doc_id == span.doc_id
            and hit.chunk.metadata.chunk_id in span_chunk_ids
        ]
        sim_score = max(seed_scores) if seed_scores else 0.0

        # Adjacency score: sum of exp(-β * |Δchunks|) for neighbors
        adj_score = self._compute_adjacency_score(span, seed_hits)

        # Continuity score: mean cosine of adjacent pairs
        cont_score = self._compute_continuity_score(span)

        # Parent bonus (placeholder - would check if parent included)
        parent_bonus = 0.0

        # Weighted sum
        total_score = (
            self.config.similarity_weight * sim_score
            + self.config.adjacency_weight * adj_score
            + self.config.continuity_weight * cont_score
            + self.config.parent_weight * parent_bonus
        )

        # Update span with scores
        span.score = total_score
        span.sim_score = sim_score
        span.adjacency_score = adj_score
        span.continuity_score = cont_score
        span.parent_bonus = parent_bonus

        return span

    def _compute_adjacency_score(self, span: Span, seed_hits: List[SeedHit]) -> float:
        """
        Compute adjacency score: sum over neighbors of exp(-β * |Δchunks|).

        This measures how close the span chunks are to seed hits.
        """
        score = 0.0

        # Find seed hits in same doc
        doc_seeds = [hit for hit in seed_hits if hit.chunk.metadata.doc_id == span.doc_id]

        for chunk_id in span.chunk_ids:
            for seed_hit in doc_seeds:
                seed_id = seed_hit.chunk.metadata.chunk_id
                distance = abs(chunk_id - seed_id)
                # Apply exponential decay
                score += math.exp(-self.config.distance_decay_beta * distance)

        return score

    def _compute_continuity_score(self, span: Span) -> float:
        """
        Compute continuity score: mean cosine similarity of adjacent pairs.

        Higher score means chunks flow together semantically.
        """
        if len(span.chunks) < 2:
            return 1.0  # Single chunk is perfectly continuous

        similarities = []
        for i in range(len(span.chunks) - 1):
            sim = self._compute_continuity(span.chunks[i], span.chunks[i + 1])
            similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _compute_continuity(self, chunk1: Chunk, chunk2: Chunk) -> float:
        """
        Compute semantic continuity between two chunks.

        Returns cosine similarity if embeddings available, else 0.5 (neutral).
        """
        if chunk1.embedding is None or chunk2.embedding is None:
            return 0.5  # Neutral default

        # Cosine similarity
        emb1 = np.array(chunk1.embedding)
        emb2 = np.array(chunk2.embedding)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def _select_within_budget(self, spans: List[Span]) -> List[Span]:
        """
        Greedy selection of highest-scoring spans within token budget (§4 step 5).

        Handles de-duplication and span limits.
        """
        # Sort by score descending
        sorted_spans = sorted(spans, key=lambda s: s.score, reverse=True)

        # De-duplicate overlapping spans if configured
        if self.config.dedupe_overlaps:
            sorted_spans = self._deduplicate_spans(sorted_spans)

        # Greedy selection
        selected: List[Span] = []
        total_tokens = 0

        for span in sorted_spans:
            if self.config.max_spans and len(selected) >= self.config.max_spans:
                break

            if total_tokens + span.token_count <= self.config.token_budget:
                selected.append(span)
                total_tokens += span.token_count
            elif not selected:
                # Always include at least one span (trimmed if needed)
                selected.append(span)
                break
        return selected

    def _deduplicate_spans(self, spans: List[Span]) -> List[Span]:
        """
        Remove overlapping spans, favoring longer and higher-scoring ones.
        """
        if not spans:
            return []

        # Track occupied chunk ranges per doc
        occupied: Dict[str, Set[int]] = defaultdict(set)
        deduplicated = []

        for span in spans:
            span_chunk_ids = set(span.chunk_ids)
            doc_occupied = occupied[span.doc_id]

            # Check for overlap
            overlap = span_chunk_ids & doc_occupied

            if not overlap:
                # No overlap - include it
                deduplicated.append(span)
                occupied[span.doc_id].update(span_chunk_ids)
            elif len(overlap) < len(span_chunk_ids):
                # Partial overlap - could include non-overlapping portion
                # For simplicity, skip (could be enhanced later)
                pass

        return deduplicated

    def _order_spans(self, spans: List[Span]) -> List[Span]:
        """
        Order spans for prompt to mitigate U-shaped bias (§4 step 6).

        Strategies:
        - EDGE_BALANCED: alternate top/bottom (default)
        - FRONT_FIRST: all at beginning
        - BACK_FIRST: all at end
        - SCORE_DESC: highest score first
        """
        if not spans:
            return []

        strategy = self.config.ordering_strategy

        if strategy == OrderingStrategy.SCORE_DESC:
            return sorted(spans, key=lambda s: s.score, reverse=True)
        elif strategy == OrderingStrategy.FRONT_FIRST:
            return spans  # Already in order
        elif strategy == OrderingStrategy.BACK_FIRST:
            return list(reversed(spans))
        else:  # edge_balanced (default)
            return self._edge_balanced_ordering(spans)

    def _edge_balanced_ordering(self, spans: List[Span]) -> List[Span]:
        """
        Edge-balanced ordering: alternate between top and bottom.

        This counters U-shaped bias by placing content at edges where
        LLMs attend better.
        """
        if len(spans) <= 2:
            return spans

        # Sort by score
        sorted_spans = sorted(spans, key=lambda s: s.score, reverse=True)

        # Alternate: first to front, second to back, etc.
        front = []
        back = []

        for i, span in enumerate(sorted_spans):
            if i % 2 == 0:
                front.append(span)
            else:
                back.append(span)

        # Combine: front spans + reversed back spans
        return front + list(reversed(back))
