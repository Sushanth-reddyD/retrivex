"""
Evaluation metrics computation utilities.

This module provides utilities for computing various evaluation metrics
used in retrieval and QA evaluation.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def compute_recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Compute Recall@k metric.
    
    Args:
        retrieved_ids: List of retrieved item IDs in rank order
        relevant_ids: Set of relevant item IDs
        k: Cut-off rank
    
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / len(relevant_ids)


def compute_precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Compute Precision@k metric.
    
    Args:
        retrieved_ids: List of retrieved item IDs in rank order
        relevant_ids: Set of relevant item IDs
        k: Cut-off rank
    
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / k


def compute_mrr(retrieved_lists: List[List[int]], relevant_sets: List[Set[int]]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_lists: List of retrieved item lists for each query
        relevant_sets: List of relevant item sets for each query
    
    Returns:
        MRR score (0.0 to 1.0)
    """
    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError("retrieved_lists and relevant_sets must have same length")
    
    reciprocal_ranks = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        if not relevant:
            continue
        
        for rank, item_id in enumerate(retrieved, 1):
            if item_id in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_ndcg_at_k(
    retrieved_ids: List[int], 
    relevance_scores: Dict[int, float], 
    k: int
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k).
    
    Args:
        retrieved_ids: List of retrieved item IDs in rank order
        relevance_scores: Dict mapping item IDs to relevance scores
        k: Cut-off rank
    
    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    # Compute DCG@k
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        rel = relevance_scores.get(item_id, 0.0)
        dcg += rel / np.log2(i + 2)  # i+2 because rank starts from 1
    
    # Compute IDCG@k (ideal DCG)
    sorted_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_scores[:k]):
        idcg += rel / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_map(retrieved_lists: List[List[int]], relevant_sets: List[Set[int]]) -> float:
    """
    Compute Mean Average Precision (MAP).
    
    Args:
        retrieved_lists: List of retrieved item lists for each query
        relevant_sets: List of relevant item sets for each query
    
    Returns:
        MAP score (0.0 to 1.0)
    """
    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError("retrieved_lists and relevant_sets must have same length")
    
    average_precisions = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        if not relevant:
            continue
        
        precisions = []
        relevant_found = 0
        
        for i, item_id in enumerate(retrieved):
            if item_id in relevant:
                relevant_found += 1
                precision = relevant_found / (i + 1)
                precisions.append(precision)
        
        if precisions:
            average_precisions.append(np.mean(precisions))
        else:
            average_precisions.append(0.0)
    
    return np.mean(average_precisions) if average_precisions else 0.0


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute Exact Match (EM) accuracy.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
    
    Returns:
        EM accuracy (0.0 to 1.0)
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    
    exact_matches = [
        pred.strip().lower() == ref.strip().lower()
        for pred, ref in zip(predictions, references)
    ]
    
    return np.mean(exact_matches)


def compute_f1_score(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score between prediction and reference.
    
    Args:
        prediction: Predicted text
        reference: Reference text
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    from collections import Counter

    pred_counter = Counter(prediction.lower().split())
    ref_counter = Counter(reference.lower().split())

    if not ref_counter:
        return 1.0 if not pred_counter else 0.0

    if not pred_counter:
        return 0.0

    intersection = (pred_counter & ref_counter)
    common = sum(intersection.values())
    precision = common / sum(pred_counter.values())
    recall = common / sum(ref_counter.values())

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_batch_f1(predictions: List[str], references: List[str]) -> float:
    """
    Compute average F1 score over a batch of predictions.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Average F1 score (0.0 to 1.0)
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    
    f1_scores = [
        compute_f1_score(pred, ref) 
        for pred, ref in zip(predictions, references)
    ]
    
    return np.mean(f1_scores)


def compute_position_bias_metrics(
    performances: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Compute position bias metrics from position-specific performances.
    
    Args:
        performances: Dict with keys 'front', 'middle', 'back' and performance lists
    
    Returns:
        Dict with bias metrics
    """
    metrics = {}
    
    # Average performance by position
    for position in ['front', 'middle', 'back']:
        if position in performances and performances[position]:
            metrics[f"{position}_performance"] = np.mean(performances[position])
        else:
            metrics[f"{position}_performance"] = 0.0
    
    # U-shaped bias score (lower is better)
    front_back_avg = (
        metrics.get('front_performance', 0.0) + 
        metrics.get('back_performance', 0.0)
    ) / 2
    
    if front_back_avg > 0:
        middle_perf = metrics.get('middle_performance', 0.0)
        metrics['position_bias_score'] = 1.0 - (middle_perf / front_back_avg)
    else:
        metrics['position_bias_score'] = 0.0
    
    # Variance across positions (lower is better for consistent performance)
    position_perfs = [
        metrics.get('front_performance', 0.0),
        metrics.get('middle_performance', 0.0),
        metrics.get('back_performance', 0.0)
    ]
    metrics['position_variance'] = np.var(position_perfs)
    
    return metrics


def compute_rescued_by_neighbor_rate(
    seed_chunk_ids: List[Set[int]],
    retrieved_chunk_ids: List[Set[int]],
    answer_chunk_ids: List[Set[int]]
) -> float:
    """
    Compute the rate at which neighbors (not seeds) rescued the answer.
    
    Args:
        seed_chunk_ids: List of seed chunk ID sets for each query
        retrieved_chunk_ids: List of all retrieved chunk ID sets for each query
        answer_chunk_ids: List of answer chunk ID sets for each query
    
    Returns:
        Rescued rate (0.0 to 1.0)
    """
    if len(seed_chunk_ids) != len(retrieved_chunk_ids) != len(answer_chunk_ids):
        raise ValueError("All input lists must have same length")
    
    rescued_count = 0
    total_count = 0
    
    for seeds, retrieved, answers in zip(seed_chunk_ids, retrieved_chunk_ids, answer_chunk_ids):
        if not answers:
            continue
        
        total_count += 1
        
        # Check if answer is in seeds
        if seeds & answers:
            continue  # Answer was in seeds, not rescued
        
        # Check if answer is in retrieved chunks
        if retrieved & answers:
            rescued_count += 1
    
    return rescued_count / total_count if total_count > 0 else 0.0


def compute_token_efficiency_metrics(
    retrieved_texts: List[str],
    answer_texts: List[str],
    f1_scores: List[float]
) -> Dict[str, float]:
    """
    Compute token efficiency metrics.
    
    Args:
        retrieved_texts: List of retrieved text strings
        answer_texts: List of answer text strings
        f1_scores: List of F1 scores for each sample
    
    Returns:
        Dict with efficiency metrics
    """
    if not (len(retrieved_texts) == len(answer_texts) == len(f1_scores)):
        raise ValueError("All input lists must have same length")
    
    token_counts = [len(text.split()) for text in retrieved_texts]
    answer_token_counts = [len(text.split()) for text in answer_texts]
    
    # Basic efficiency metrics
    metrics = {
        'avg_retrieved_tokens': np.mean(token_counts),
        'avg_answer_tokens': np.mean(answer_token_counts),
        'avg_f1_score': np.mean(f1_scores)
    }
    
    # Token efficiency: F1 per token
    token_efficiencies = [
        f1 / max(tokens, 1) for f1, tokens in zip(f1_scores, token_counts)
    ]
    metrics['token_efficiency'] = np.mean(token_efficiencies)
    
    # Answer coverage ratio
    coverage_ratios = [
        min(1.0, ans_tokens / max(ret_tokens, 1))
        for ans_tokens, ret_tokens in zip(answer_token_counts, token_counts)
    ]
    metrics['answer_coverage_ratio'] = np.mean(coverage_ratios)
    
    return metrics
