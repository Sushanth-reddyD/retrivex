"""
Base classes and utilities for evaluation framework.

This module provides the foundation for evaluating RetriVex against benchmark datasets
like LongBench, RULER, and NIAH (Needle-in-a-Haystack).
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.models import Chunk, RetrievalConfig, SeedHit, Span
from ..core.retriever import SpanComposer


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    
    # Dataset parameters
    max_samples: Optional[int] = None  # Limit samples for faster eval
    random_seed: int = 42
    
    # Retrieval parameters
    retrieval_configs: List[RetrievalConfig] = field(default_factory=lambda: [
        RetrievalConfig(),  # Default config
    ])
    
    # Comparison baselines
    include_baselines: bool = True
    baseline_methods: List[str] = field(default_factory=lambda: [
        "vanilla_knn",  # Pure vector search
        "simple_expansion",  # Basic neighbor expansion
    ])
    
    # Output configuration
    save_results: bool = True
    results_dir: str = "eval_results"
    verbose: bool = True


@dataclass
class EvaluationSample:
    """A single evaluation sample."""
    
    # Sample identification
    sample_id: str
    dataset: str
    task_type: str
    
    # Input data
    query: str
    context: str  # Full document/context
    chunks: List[Chunk]  # Pre-chunked context
    
    # Ground truth
    answer: str
    answer_span: Optional[Tuple[int, int]] = None  # Character span in context
    answer_chunk_ids: Optional[List[int]] = None  # Chunk IDs containing answer
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Results from a single retrieval run."""
    
    # Method information
    method_name: str
    spans: List[Span]
    total_tokens: int
    retrieval_time_ms: float
    seed_hits: List[SeedHit]
    
    # Optional fields with defaults
    config: Optional[RetrievalConfig] = None
    expanded_chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    
    # Retrieval metrics
    recall_at_k_span: float = 0.0  # % queries where span contains answer
    mrr: float = 0.0  # Mean reciprocal rank
    rescued_by_neighbor_pct: float = 0.0  # % rescued by non-seed chunks
    
    # End-to-end metrics
    exact_match: float = 0.0  # Exact string match
    f1_score: float = 0.0  # Token-level F1
    
    # Position sensitivity
    position_bias_score: float = 0.0  # U-shaped bias metric
    front_performance: float = 0.0  # Performance on front answers
    middle_performance: float = 0.0  # Performance on middle answers
    back_performance: float = 0.0  # Performance on back answers
    
    # Efficiency metrics
    avg_retrieval_time_ms: float = 0.0
    tokens_per_query: float = 0.0
    
    # Count metrics
    total_samples: int = 0
    successful_retrievals: int = 0


class BaseEvaluator(ABC):
    """Base class for benchmark evaluators."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []
        
    @abstractmethod
    def load_dataset(self) -> List[EvaluationSample]:
        """Load and prepare the evaluation dataset."""
        pass
    
    @abstractmethod
    def create_chunks(self, sample: EvaluationSample) -> List[Chunk]:
        """Create chunks from sample context."""
        pass
    
    @abstractmethod
    def simulate_vector_search(
        self, 
        query: str, 
        chunks: List[Chunk], 
        k: int = 6
    ) -> List[SeedHit]:
        """Simulate vector search to get seed hits."""
        pass
    
    def evaluate_retrieval_method(
        self,
        method_name: str,
        sample: EvaluationSample,
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalResult:
        """Evaluate a single retrieval method on one sample."""
        
        start_time = time.time()
        
        if method_name == "vanilla_knn":
            result = self._evaluate_vanilla_knn(sample, config)
        elif method_name == "simple_expansion":
            result = self._evaluate_simple_expansion(sample, config)
        elif method_name == "retrivex":
            result = self._evaluate_retrivex(sample, config or RetrievalConfig())
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Add timing
        result.retrieval_time_ms = (time.time() - start_time) * 1000
        result.method_name = method_name
        result.config = config
        
        return result
    
    def _evaluate_vanilla_knn(
        self, 
        sample: EvaluationSample, 
        config: Optional[RetrievalConfig]
    ) -> RetrievalResult:
        """Baseline: vanilla k-NN without expansion."""
        
        # Get seed hits
        k = config.k if config else 6
        seed_hits = self.simulate_vector_search(sample.query, sample.chunks, k=k)
        
        # Convert seed hits to spans (no expansion)
        spans = []
        total_tokens = 0
        budget = config.token_budget if config else 2000
        
        for hit in seed_hits:
            if total_tokens >= budget:
                break
            
            # Create single-chunk span
            span = Span(
                doc_id=hit.chunk.metadata.doc_id,
                chunks=[hit.chunk],
                chunk_ids=[hit.chunk.metadata.chunk_id],
                score=hit.similarity_score,
                sim_score=hit.similarity_score,
                adjacency_score=0.0,
                continuity_score=0.0,
                char_start=hit.chunk.metadata.char_start,
                char_end=hit.chunk.metadata.char_end,
                token_count=len(hit.chunk.text.split())
            )
            spans.append(span)
            total_tokens += span.token_count
        
        return RetrievalResult(
            method_name="vanilla_knn",
            spans=spans,
            total_tokens=total_tokens,
            retrieval_time_ms=0.0,  # Will be set by caller
            seed_hits=seed_hits
        )
    
    def _evaluate_simple_expansion(
        self, 
        sample: EvaluationSample, 
        config: Optional[RetrievalConfig]
    ) -> RetrievalResult:
        """Baseline: simple neighbor expansion without sophisticated scoring."""
        
        # Get seed hits
        k = config.k if config else 6
        window = config.window if config else 2
        seed_hits = self.simulate_vector_search(sample.query, sample.chunks, k=k)
        
        # Create chunk store
        chunk_store = {
            (chunk.metadata.doc_id, chunk.metadata.chunk_id): chunk
            for chunk in sample.chunks
        }
        
        # Simple expansion: just add neighbors
        expanded_chunk_ids = set()
        for hit in seed_hits:
            chunk_id = hit.chunk.metadata.chunk_id
            doc_id = hit.chunk.metadata.doc_id
            
            # Add seed and neighbors
            for offset in range(-window, window + 1):
                neighbor_id = chunk_id + offset
                neighbor_key = (doc_id, neighbor_id)
                if neighbor_key in chunk_store:
                    expanded_chunk_ids.add(neighbor_id)
        
        # Get chunks in order
        expanded_chunks = []
        for chunk_id in sorted(expanded_chunk_ids):
            neighbor_key = (sample.chunks[0].metadata.doc_id, chunk_id)
            if neighbor_key in chunk_store:
                expanded_chunks.append(chunk_store[neighbor_key])
        
        # Group consecutive chunks into spans
        spans = []
        current_span_chunks = []
        total_tokens = 0
        budget = config.token_budget if config else 2000
        
        i = 0
        while i < len(expanded_chunks) and total_tokens < budget:
            chunk = expanded_chunks[i]
            tokens = len(chunk.text.split())
            
            if total_tokens + tokens > budget:
                break
            
            # Check if this chunk is contiguous with the current span
            if (not current_span_chunks or 
                chunk.metadata.chunk_id == current_span_chunks[-1].metadata.chunk_id + 1):
                # Contiguous - add to current span
                current_span_chunks.append(chunk)
                total_tokens += tokens
            else:
                # Non-contiguous - finalize current span and start new one
                if current_span_chunks:
                    span = self._create_simple_span(current_span_chunks)
                    spans.append(span)
                
                # Start new span
                current_span_chunks = [chunk]
                total_tokens += tokens
            
            i += 1
        
        # Finalize last span
        if current_span_chunks:
            span = self._create_simple_span(current_span_chunks)
            spans.append(span)
        
        return RetrievalResult(
            method_name="simple_expansion",
            spans=spans,
            total_tokens=total_tokens,
            retrieval_time_ms=0.0,  # Will be set by caller
            seed_hits=seed_hits,
            expanded_chunks=list(expanded_chunks)
        )
    
    def _create_simple_span(self, chunks: List[Chunk]) -> Span:
        """Create a span from chunks with simple scoring."""
        return Span(
            doc_id=chunks[0].metadata.doc_id,
            chunks=chunks,
            chunk_ids=[c.metadata.chunk_id for c in chunks],
            score=1.0,  # Simple scoring
            sim_score=1.0,
            adjacency_score=0.0,
            continuity_score=0.0,
            char_start=min(c.metadata.char_start for c in chunks),
            char_end=max(c.metadata.char_end for c in chunks),
            token_count=sum(len(c.text.split()) for c in chunks)
        )
    
    def _evaluate_retrivex(
        self, 
        sample: EvaluationSample, 
        config: RetrievalConfig
    ) -> RetrievalResult:
        """Evaluate RetriVex method."""
        
        # Get seed hits
        seed_hits = self.simulate_vector_search(sample.query, sample.chunks, k=config.k)
        
        # Create chunk store
        chunk_store = {
            (chunk.metadata.doc_id, chunk.metadata.chunk_id): chunk
            for chunk in sample.chunks
        }
        
        # Use RetriVex
        composer = SpanComposer(config)
        spans = composer.retrieve(seed_hits=seed_hits, chunk_store=chunk_store)
        
        total_tokens = sum(span.token_count for span in spans)
        
        return RetrievalResult(
            method_name="retrivex",
            spans=spans,
            total_tokens=total_tokens,
            retrieval_time_ms=0.0,  # Will be set by caller
            seed_hits=seed_hits
        )
    
    def compute_metrics(
        self, 
        results: List[RetrievalResult], 
        samples: List[EvaluationSample]
    ) -> EvaluationMetrics:
        """Compute comprehensive evaluation metrics."""
        
        if len(results) != len(samples):
            raise ValueError("Results and samples must have same length")
        
        metrics = EvaluationMetrics()
        metrics.total_samples = len(samples)
        metrics.successful_retrievals = len([r for r in results if r.spans])
        
        if metrics.successful_retrievals == 0:
            return metrics
        
        # Collect per-sample metrics
        recall_scores = []
        mrr_scores = []
        rescued_counts = []
        em_scores = []
        f1_scores = []
        retrieval_times = []
        token_counts = []
        position_performances = {"front": [], "middle": [], "back": []}
        
        for result, sample in zip(results, samples):
            if not result.spans:
                continue
            
            # Recall@k_span
            recall = self._compute_recall_at_k_span(result, sample)
            recall_scores.append(recall)
            
            # MRR
            mrr = self._compute_mrr(result, sample)
            mrr_scores.append(mrr)
            
            # Rescued by neighbor
            rescued = self._compute_rescued_by_neighbor(result, sample)
            rescued_counts.append(rescued)
            
            # EM and F1
            em, f1 = self._compute_em_f1(result, sample)
            em_scores.append(em)
            f1_scores.append(f1)
            
            # Timing and tokens
            retrieval_times.append(result.retrieval_time_ms)
            token_counts.append(result.total_tokens)
            
            # Position sensitivity
            position = self._get_answer_position(sample)
            if position in position_performances:
                position_performances[position].append(f1)
        
        # Aggregate metrics
        metrics.recall_at_k_span = np.mean(recall_scores) if recall_scores else 0.0
        metrics.mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        metrics.rescued_by_neighbor_pct = np.mean(rescued_counts) * 100 if rescued_counts else 0.0
        metrics.exact_match = np.mean(em_scores) if em_scores else 0.0
        metrics.f1_score = np.mean(f1_scores) if f1_scores else 0.0
        metrics.avg_retrieval_time_ms = np.mean(retrieval_times) if retrieval_times else 0.0
        metrics.tokens_per_query = np.mean(token_counts) if token_counts else 0.0
        
        # Position sensitivity
        metrics.front_performance = np.mean(position_performances["front"]) if position_performances["front"] else 0.0
        metrics.middle_performance = np.mean(position_performances["middle"]) if position_performances["middle"] else 0.0
        metrics.back_performance = np.mean(position_performances["back"]) if position_performances["back"] else 0.0
        
        # U-shaped bias score (lower is better)
        front_back_avg = (metrics.front_performance + metrics.back_performance) / 2
        if front_back_avg > 0:
            metrics.position_bias_score = 1.0 - (metrics.middle_performance / front_back_avg)
        
        return metrics
    
    def _compute_recall_at_k_span(self, result: RetrievalResult, sample: EvaluationSample) -> float:
        """Compute whether any span contains the answer."""
        if not sample.answer_chunk_ids:
            return 0.0
        
        # Check if any retrieved span overlaps with answer chunks
        for span in result.spans:
            span_chunk_ids = set(span.chunk_ids)
            answer_chunk_ids = set(sample.answer_chunk_ids)
            if span_chunk_ids & answer_chunk_ids:  # Intersection
                return 1.0
        
        return 0.0
    
    def _compute_mrr(self, result: RetrievalResult, sample: EvaluationSample) -> float:
        """Compute mean reciprocal rank."""
        if not sample.answer_chunk_ids:
            return 0.0
        
        answer_chunk_ids = set(sample.answer_chunk_ids)
        
        for rank, span in enumerate(result.spans, 1):
            span_chunk_ids = set(span.chunk_ids)
            if span_chunk_ids & answer_chunk_ids:
                return 1.0 / rank
        
        return 0.0
    
    def _compute_rescued_by_neighbor(self, result: RetrievalResult, sample: EvaluationSample) -> float:
        """Compute if neighbors (not seeds) contained the answer."""
        if not sample.answer_chunk_ids:
            return 0.0
        
        # Get seed chunk IDs
        seed_chunk_ids = {hit.chunk.metadata.chunk_id for hit in result.seed_hits}
        answer_chunk_ids = set(sample.answer_chunk_ids)
        
        # Check if answer is in seeds
        if seed_chunk_ids & answer_chunk_ids:
            return 0.0  # Answer was in seeds, not rescued
        
        # Check if answer is in any retrieved span
        for span in result.spans:
            span_chunk_ids = set(span.chunk_ids)
            if span_chunk_ids & answer_chunk_ids:
                return 1.0  # Answer was rescued by expansion
        
        return 0.0
    
    def _compute_em_f1(self, result: RetrievalResult, sample: EvaluationSample) -> Tuple[float, float]:
        """Compute exact match and F1 score."""
        retrieved_text = " ".join(span.text for span in result.spans)
        
        # Simple exact match (case-insensitive)
        em = 1.0 if sample.answer.lower() in retrieved_text.lower() else 0.0
        
        # Token-level F1
        answer_tokens = set(sample.answer.lower().split())
        retrieved_tokens = set(retrieved_text.lower().split())
        
        if not answer_tokens:
            return em, 1.0 if not retrieved_tokens else 0.0
        
        intersection = answer_tokens & retrieved_tokens
        precision = len(intersection) / len(retrieved_tokens) if retrieved_tokens else 0.0
        recall = len(intersection) / len(answer_tokens)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return em, f1
    
    def _get_answer_position(self, sample: EvaluationSample) -> str:
        """Determine if answer is at front, middle, or back of context."""
        if not sample.answer_chunk_ids:
            return "unknown"
        
        total_chunks = len(sample.chunks)
        if total_chunks == 0:
            return "unknown"
        
        # Use first answer chunk for position
        answer_chunk_id = min(sample.answer_chunk_ids)
        position_ratio = answer_chunk_id / total_chunks
        
        if position_ratio < 0.33:
            return "front"
        elif position_ratio > 0.67:
            return "back"
        else:
            return "middle"
    
    def run_evaluation(self) -> Dict[str, EvaluationMetrics]:
        """Run complete evaluation across all methods and configurations."""
        
        # Load dataset
        if self.config.verbose:
            print(f"Loading {self.__class__.__name__} dataset...")
        
        samples = self.load_dataset()
        
        if self.config.max_samples:
            samples = samples[:self.config.max_samples]
        
        if self.config.verbose:
            print(f"Evaluating on {len(samples)} samples")
        
        # Prepare all methods to evaluate
        methods = []
        
        if self.config.include_baselines:
            methods.extend(self.config.baseline_methods)
        
        # Add RetriVex with different configs
        for config in self.config.retrieval_configs:
            methods.append(("retrivex", config))
        
        # Run evaluation for each method
        all_results = {}
        
        for method in methods:
            if isinstance(method, tuple):
                method_name, config = method
                method_key = f"{method_name}_{id(config)}"
            else:
                method_name = method
                config = None
                method_key = method_name
            
            if self.config.verbose:
                print(f"\nEvaluating {method_key}...")
            
            # Evaluate all samples
            method_results = []
            for i, sample in enumerate(samples):
                if self.config.verbose and (i + 1) % 10 == 0:
                    print(f"  Sample {i + 1}/{len(samples)}")
                
                result = self.evaluate_retrieval_method(method_name, sample, config)
                method_results.append(result)
            
            # Compute metrics
            metrics = self.compute_metrics(method_results, samples)
            all_results[method_key] = metrics
            
            if self.config.verbose:
                print(f"  Recall@k_span: {metrics.recall_at_k_span:.3f}")
                print(f"  F1 Score: {metrics.f1_score:.3f}")
                print(f"  Avg Time: {metrics.avg_retrieval_time_ms:.1f}ms")
        
        return all_results
