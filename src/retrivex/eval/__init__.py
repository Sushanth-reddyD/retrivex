"""
Evaluation utilities for RetriVex.

This module provides comprehensive evaluation capabilities including:
- LongBench evaluation for multi-task long-context QA
- RULER evaluation for synthetic position sensitivity testing
- NIAH (Needle-in-a-Haystack) evaluation for quick position bias detection
- Comprehensive benchmarking and reporting
"""

from .base import BaseEvaluator, EvaluationConfig, EvaluationSample, EvaluationMetrics, RetrievalResult
from .benchmark import ComprehensiveBenchmark, run_benchmark
from .longbench import LongBenchEvaluator
from .metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_map,
    compute_exact_match,
    compute_f1_score,
    compute_batch_f1,
    compute_position_bias_metrics,
    compute_rescued_by_neighbor_rate,
    compute_token_efficiency_metrics
)
from .niah import NIAHEvaluator
from .ruler import RULEREvaluator

__all__ = [
    # Base classes
    "BaseEvaluator",
    "EvaluationConfig", 
    "EvaluationSample",
    "EvaluationMetrics",
    "RetrievalResult",
    
    # Evaluators
    "LongBenchEvaluator",
    "RULEREvaluator", 
    "NIAHEvaluator",
    
    # Benchmarking
    "ComprehensiveBenchmark",
    "run_benchmark",
    
    # Metrics
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_map",
    "compute_exact_match",
    "compute_f1_score",
    "compute_batch_f1",
    "compute_position_bias_metrics",
    "compute_rescued_by_neighbor_rate",
    "compute_token_efficiency_metrics"
]
