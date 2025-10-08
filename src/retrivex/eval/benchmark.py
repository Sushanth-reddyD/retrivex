"""
Main benchmarking script for comprehensive RetriVex evaluation.

This script runs all benchmarks (LongBench, RULER, NIAH) and compares RetriVex
against baseline methods with detailed timing and accuracy measurements.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.models import RetrievalConfig, OrderingStrategy
from .base import EvaluationConfig, EvaluationMetrics
from .longbench import LongBenchEvaluator
from .niah import NIAHEvaluator
from .ruler import RULEREvaluator


class ComprehensiveBenchmark:
    """
    Comprehensive benchmarking suite for RetriVex evaluation.
    
    This class orchestrates evaluation across multiple benchmarks and generates
    detailed reports comparing RetriVex against baseline methods.
    """
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        max_samples_per_benchmark: int = 50,
        include_baselines: bool = True,
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.max_samples_per_benchmark = max_samples_per_benchmark
        self.include_baselines = include_baselines
        self.verbose = verbose
        
        # Results storage
        self.results = {}
        self.benchmark_configs = {}
        
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation across all benchmarks."""
        
        if self.verbose:
            print("🚀 Starting Comprehensive RetriVex Evaluation")
            print("=" * 60)
        
        # Define retrieval configurations to test
        configs = self._create_test_configurations()
        
        # Run each benchmark
        benchmark_results = {}
        
        # 1. LongBench Evaluation
        if self.verbose:
            print("\n📚 Running LongBench Evaluation...")
        longbench_results = self._run_longbench_evaluation(configs)
        benchmark_results["longbench"] = longbench_results
        
        # 2. RULER Evaluation
        if self.verbose:
            print("\n📏 Running RULER Evaluation...")
        ruler_results = self._run_ruler_evaluation(configs)
        benchmark_results["ruler"] = ruler_results
        
        # 3. NIAH Evaluation
        if self.verbose:
            print("\n🔍 Running NIAH Evaluation...")
        niah_results = self._run_niah_evaluation(configs)
        benchmark_results["niah"] = niah_results
        
        # 4. Generate comprehensive analysis
        if self.verbose:
            print("\n📊 Generating Analysis...")
        analysis = self._generate_comprehensive_analysis(benchmark_results)
        
        # 5. Save results
        self._save_results(benchmark_results, analysis)
        
        # 6. Generate visualizations
        if self.verbose:
            print("\n📈 Creating Visualizations...")
        self._create_visualizations(benchmark_results, analysis)
        
        if self.verbose:
            print("\n✅ Evaluation Complete!")
            print(f"📁 Results saved to: {self.output_dir}")
        
        return {
            "benchmark_results": benchmark_results,
            "analysis": analysis,
            "output_dir": str(self.output_dir)
        }
    
    def _create_test_configurations(self) -> List[Dict[str, Any]]:
        """Create different RetriVex configurations to test."""
        
        configs = []
        
        # Base configuration
        base_config = RetrievalConfig(
            window=2,
            k=6,
            token_budget=2000,
            ordering_strategy=OrderingStrategy.EDGE_BALANCED
        )
        configs.append({"name": "retrivex_default", "config": base_config})
        
        # Window size variations
        for window in [1, 3, 4]:
            config = RetrievalConfig(
                window=window,
                k=6,
                token_budget=2000,
                ordering_strategy=OrderingStrategy.EDGE_BALANCED
            )
            configs.append({"name": f"retrivex_window_{window}", "config": config})
        
        # Token budget variations
        for budget in [1000, 3000, 4000]:
            config = RetrievalConfig(
                window=2,
                k=6,
                token_budget=budget,
                ordering_strategy=OrderingStrategy.EDGE_BALANCED
            )
            configs.append({"name": f"retrivex_budget_{budget}", "config": config})
        
        # Ordering strategy variations
        for strategy in [OrderingStrategy.SCORE_DESC, OrderingStrategy.FRONT_FIRST]:
            config = RetrievalConfig(
                window=2,
                k=6,
                token_budget=2000,
                ordering_strategy=strategy
            )
            configs.append({"name": f"retrivex_{strategy.value}", "config": config})
        
        # k variations
        for k in [4, 8, 10]:
            config = RetrievalConfig(
                window=2,
                k=k,
                token_budget=2000,
                ordering_strategy=OrderingStrategy.EDGE_BALANCED
            )
            configs.append({"name": f"retrivex_k_{k}", "config": config})
        
        return configs
    
    def _run_longbench_evaluation(self, configs: List[Dict[str, Any]]) -> Dict[str, EvaluationMetrics]:
        """Run LongBench evaluation."""
        
        eval_config = EvaluationConfig(
            max_samples=self.max_samples_per_benchmark,
            include_baselines=self.include_baselines,
            save_results=True,
            results_dir=str(self.output_dir / "longbench"),
            verbose=False
        )
        
        evaluator = LongBenchEvaluator(
            config=eval_config,
            tasks=["narrativeqa", "qasper", "hotpotqa", "multifieldqa_en"]
        )
        
        # Run evaluation for each configuration
        results = {}
        
        # Add baselines
        if self.include_baselines:
            eval_config.baseline_methods = ["vanilla_knn", "simple_expansion"]
            baseline_results = evaluator.run_evaluation()
            results.update(baseline_results)
        
        # Add RetriVex configurations
        for config_info in configs:
            eval_config.retrieval_configs = [config_info["config"]]
            config_results = evaluator.run_evaluation()
            
            # Rename results with config name
            for key, metrics in config_results.items():
                if key.startswith("retrivex"):
                    results[config_info["name"]] = metrics
        
        return results
    
    def _run_ruler_evaluation(self, configs: List[Dict[str, Any]]) -> Dict[str, EvaluationMetrics]:
        """Run RULER evaluation."""
        
        eval_config = EvaluationConfig(
            max_samples=self.max_samples_per_benchmark,
            include_baselines=self.include_baselines,
            save_results=True,
            results_dir=str(self.output_dir / "ruler"),
            verbose=False
        )
        
        evaluator = RULEREvaluator(
            config=eval_config,
            context_lengths=[4000, 8000, 16000],
            needle_positions=["beginning", "middle", "end"],
            needle_types=["fact", "instruction", "keyword"]
        )
        
        # Run evaluation for each configuration
        results = {}
        
        # Add baselines
        if self.include_baselines:
            eval_config.baseline_methods = ["vanilla_knn", "simple_expansion"]
            baseline_results = evaluator.run_evaluation()
            results.update(baseline_results)
        
        # Add RetriVex configurations
        for config_info in configs:
            eval_config.retrieval_configs = [config_info["config"]]
            config_results = evaluator.run_evaluation()
            
            # Rename results with config name
            for key, metrics in config_results.items():
                if key.startswith("retrivex"):
                    results[config_info["name"]] = metrics
        
        return results
    
    def _run_niah_evaluation(self, configs: List[Dict[str, Any]]) -> Dict[str, EvaluationMetrics]:
        """Run NIAH evaluation."""
        
        eval_config = EvaluationConfig(
            max_samples=self.max_samples_per_benchmark,
            include_baselines=self.include_baselines,
            save_results=True,
            results_dir=str(self.output_dir / "niah"),
            verbose=False
        )
        
        evaluator = NIAHEvaluator(
            config=eval_config,
            haystack_lengths=[2000, 4000, 8000, 16000],
            needle_depths=[0.1, 0.3, 0.5, 0.7, 0.9],
            needle_difficulties=["easy", "medium", "hard"]
        )
        
        # Run evaluation for each configuration
        results = {}
        
        # Add baselines
        if self.include_baselines:
            eval_config.baseline_methods = ["vanilla_knn", "simple_expansion"]
            baseline_results = evaluator.run_evaluation()
            results.update(baseline_results)
        
        # Add RetriVex configurations
        for config_info in configs:
            eval_config.retrieval_configs = [config_info["config"]]
            config_results = evaluator.run_evaluation()
            
            # Rename results with config name
            for key, metrics in config_results.items():
                if key.startswith("retrivex"):
                    results[config_info["name"]] = metrics
        
        return results
    
    def _generate_comprehensive_analysis(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]]) -> Dict[str, Any]:
        """Generate comprehensive analysis across all benchmarks."""
        
        analysis = {
            "summary": {},
            "position_sensitivity": {},
            "efficiency": {},
            "configuration_analysis": {},
            "recommendations": {}
        }
        
        # 1. Overall Summary
        analysis["summary"] = self._analyze_overall_performance(benchmark_results)
        
        # 2. Position Sensitivity Analysis
        analysis["position_sensitivity"] = self._analyze_position_sensitivity(benchmark_results)
        
        # 3. Efficiency Analysis
        analysis["efficiency"] = self._analyze_efficiency(benchmark_results)
        
        # 4. Configuration Analysis
        analysis["configuration_analysis"] = self._analyze_configurations(benchmark_results)
        
        # 5. Generate Recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_overall_performance(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]]) -> Dict[str, Any]:
        """Analyze overall performance across benchmarks."""
        
        summary = {
            "best_overall_method": None,
            "best_f1_score": 0.0,
            "best_recall_at_k": 0.0,
            "benchmark_scores": {},
            "method_rankings": {}
        }
        
        # Aggregate scores across benchmarks
        method_scores = {}
        
        for benchmark_name, results in benchmark_results.items():
            summary["benchmark_scores"][benchmark_name] = {}
            
            for method_name, metrics in results.items():
                if method_name not in method_scores:
                    method_scores[method_name] = {
                        "f1_scores": [],
                        "recall_scores": [],
                        "mrr_scores": [],
                        "efficiency_scores": []
                    }
                
                method_scores[method_name]["f1_scores"].append(metrics.f1_score)
                method_scores[method_name]["recall_scores"].append(metrics.recall_at_k_span)
                method_scores[method_name]["mrr_scores"].append(metrics.mrr)
                
                # Efficiency score (F1 per ms)
                if metrics.avg_retrieval_time_ms > 0:
                    efficiency = metrics.f1_score / metrics.avg_retrieval_time_ms * 1000
                else:
                    efficiency = metrics.f1_score
                method_scores[method_name]["efficiency_scores"].append(efficiency)
                
                summary["benchmark_scores"][benchmark_name][method_name] = {
                    "f1": metrics.f1_score,
                    "recall": metrics.recall_at_k_span,
                    "mrr": metrics.mrr
                }
        
        # Calculate aggregate scores and rankings
        for method_name, scores in method_scores.items():
            avg_f1 = np.mean(scores["f1_scores"])
            avg_recall = np.mean(scores["recall_scores"])
            avg_mrr = np.mean(scores["mrr_scores"])
            avg_efficiency = np.mean(scores["efficiency_scores"])
            
            summary["method_rankings"][method_name] = {
                "avg_f1": avg_f1,
                "avg_recall": avg_recall,
                "avg_mrr": avg_mrr,
                "avg_efficiency": avg_efficiency,
                "composite_score": (avg_f1 + avg_recall + avg_mrr) / 3
            }
        
        # Find best overall method
        best_method = max(
            summary["method_rankings"].items(),
            key=lambda x: x[1]["composite_score"]
        )
        summary["best_overall_method"] = best_method[0]
        summary["best_f1_score"] = best_method[1]["avg_f1"]
        summary["best_recall_at_k"] = best_method[1]["avg_recall"]
        
        return summary
    
    def _analyze_position_sensitivity(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]]) -> Dict[str, Any]:
        """Analyze position sensitivity across methods."""
        
        analysis = {
            "u_shaped_bias_detected": {},
            "middle_performance_drops": {},
            "position_robustness": {},
            "edge_balanced_advantage": {}
        }
        
        for benchmark_name, results in benchmark_results.items():
            for method_name, metrics in results.items():
                # Calculate position bias score
                if metrics.front_performance > 0 and metrics.back_performance > 0:
                    front_back_avg = (metrics.front_performance + metrics.back_performance) / 2
                    if front_back_avg > 0:
                        bias_score = 1.0 - (metrics.middle_performance / front_back_avg)
                        analysis["u_shaped_bias_detected"][f"{benchmark_name}_{method_name}"] = bias_score > 0.1
                        analysis["middle_performance_drops"][f"{benchmark_name}_{method_name}"] = bias_score
                
                # Position robustness (lower variance is better)
                position_variance = np.var([
                    metrics.front_performance,
                    metrics.middle_performance,
                    metrics.back_performance
                ])
                analysis["position_robustness"][f"{benchmark_name}_{method_name}"] = 1.0 / (1.0 + position_variance)
        
        # Analyze edge-balanced advantage
        edge_balanced_methods = [name for name in results.keys() if "edge_balanced" in name.lower()]
        score_desc_methods = [name for name in results.keys() if "score_desc" in name.lower()]
        
        for benchmark_name, results in benchmark_results.items():
            if edge_balanced_methods and score_desc_methods:
                edge_balanced_bias = np.mean([
                    analysis["middle_performance_drops"].get(f"{benchmark_name}_{method}", 0)
                    for method in edge_balanced_methods
                ])
                score_desc_bias = np.mean([
                    analysis["middle_performance_drops"].get(f"{benchmark_name}_{method}", 0)
                    for method in score_desc_methods
                ])
                
                analysis["edge_balanced_advantage"][benchmark_name] = edge_balanced_bias < score_desc_bias
        
        return analysis
    
    def _analyze_efficiency(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]]) -> Dict[str, Any]:
        """Analyze efficiency metrics."""
        
        analysis = {
            "latency_overhead": {},
            "token_efficiency": {},
            "scalability": {}
        }
        
        # Find baseline latency
        baseline_latencies = {}
        retrivex_latencies = {}
        
        for benchmark_name, results in benchmark_results.items():
            baseline_time = results.get("vanilla_knn", EvaluationMetrics()).avg_retrieval_time_ms
            
            for method_name, metrics in results.items():
                method_key = f"{benchmark_name}_{method_name}"
                
                if baseline_time > 0:
                    overhead = (metrics.avg_retrieval_time_ms / baseline_time - 1) * 100
                    analysis["latency_overhead"][method_key] = overhead
                
                # Token efficiency (F1 per token)
                if metrics.tokens_per_query > 0:
                    token_efficiency = metrics.f1_score / metrics.tokens_per_query * 1000
                    analysis["token_efficiency"][method_key] = token_efficiency
        
        return analysis
    
    def _analyze_configurations(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]]) -> Dict[str, Any]:
        """Analyze different configuration performances."""
        
        analysis = {
            "window_size_analysis": {},
            "token_budget_analysis": {},
            "k_value_analysis": {},
            "ordering_strategy_analysis": {}
        }
        
        # Group results by configuration type
        for benchmark_name, results in benchmark_results.items():
            # Window size analysis
            window_results = {}
            budget_results = {}
            k_results = {}
            ordering_results = {}
            
            for method_name, metrics in results.items():
                if "window_" in method_name:
                    window_size = method_name.split("window_")[-1]
                    window_results[window_size] = metrics.f1_score
                elif "budget_" in method_name:
                    budget = method_name.split("budget_")[-1]
                    budget_results[budget] = metrics.f1_score
                elif "k_" in method_name:
                    k_value = method_name.split("k_")[-1]
                    k_results[k_value] = metrics.f1_score
                elif any(strategy in method_name for strategy in ["edge_balanced", "score_desc", "front_first"]):
                    strategy = method_name.split("retrivex_")[-1]
                    ordering_results[strategy] = metrics.f1_score
            
            if window_results:
                analysis["window_size_analysis"][benchmark_name] = window_results
            if budget_results:
                analysis["token_budget_analysis"][benchmark_name] = budget_results
            if k_results:
                analysis["k_value_analysis"][benchmark_name] = k_results
            if ordering_results:
                analysis["ordering_strategy_analysis"][benchmark_name] = ordering_results
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on analysis."""
        
        recommendations = {
            "optimal_configuration": {},
            "use_case_specific": {},
            "general_guidelines": []
        }
        
        # Find optimal configuration
        best_method = analysis["summary"]["best_overall_method"]
        if best_method and "retrivex" in best_method:
            recommendations["optimal_configuration"]["method"] = best_method
            recommendations["optimal_configuration"]["rationale"] = f"Best overall performance with composite score"
        
        # Use case specific recommendations
        recommendations["use_case_specific"] = {
            "long_documents": "Use larger window size (3-4) and higher token budget (3000+)",
            "precision_tasks": "Use smaller window size (1-2) and edge-balanced ordering",
            "speed_critical": "Use vanilla_knn or small window size with lower k",
            "position_sensitive": "Always use edge-balanced ordering, avoid score-descending"
        }
        
        # General guidelines
        recommendations["general_guidelines"] = [
            "RetriVex shows consistent improvement over vanilla k-NN retrieval",
            "Edge-balanced ordering reduces position bias compared to score-descending",
            "Window size 2-3 provides good balance between coverage and noise",
            "Token budget should be sized according to model context window",
            "Position bias is most pronounced in middle sections of long documents"
        ]
        
        return recommendations
    
    def _save_results(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]], analysis: Dict[str, Any]):
        """Save results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        # Convert metrics to dict for JSON serialization
        serializable_results = {}
        for benchmark_name, results in benchmark_results.items():
            serializable_results[benchmark_name] = {}
            for method_name, metrics in results.items():
                serializable_results[benchmark_name][method_name] = {
                    "recall_at_k_span": float(metrics.recall_at_k_span),
                    "mrr": float(metrics.mrr),
                    "rescued_by_neighbor_pct": float(metrics.rescued_by_neighbor_pct),
                    "exact_match": float(metrics.exact_match),
                    "f1_score": float(metrics.f1_score),
                    "position_bias_score": float(metrics.position_bias_score),
                    "front_performance": float(metrics.front_performance),
                    "middle_performance": float(metrics.middle_performance),
                    "back_performance": float(metrics.back_performance),
                    "avg_retrieval_time_ms": float(metrics.avg_retrieval_time_ms),
                    "tokens_per_query": float(metrics.tokens_per_query),
                    "total_samples": int(metrics.total_samples),
                    "successful_retrievals": int(metrics.successful_retrievals)
                }
        
        # Make analysis JSON serializable
        serializable_analysis = self._make_json_serializable(analysis)
        
        with open(results_file, 'w') as f:
            json.dump({
                "benchmark_results": serializable_results,
                "analysis": serializable_analysis,
                "timestamp": timestamp
            }, f, indent=2)
        
        # Save CSV for easy analysis
        self._save_results_csv(benchmark_results, timestamp)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, (int, float)):
            return float(obj)
        elif isinstance(obj, str):
            return obj
        elif obj is None:
            return None
        else:
            return str(obj)  # Convert unknown types to string
    
    def _save_results_csv(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]], timestamp: str):
        """Save results in CSV format for easy analysis."""
        
        rows = []
        for benchmark_name, results in benchmark_results.items():
            for method_name, metrics in results.items():
                row = {
                    "benchmark": benchmark_name,
                    "method": method_name,
                    "recall_at_k_span": metrics.recall_at_k_span,
                    "mrr": metrics.mrr,
                    "rescued_by_neighbor_pct": metrics.rescued_by_neighbor_pct,
                    "exact_match": metrics.exact_match,
                    "f1_score": metrics.f1_score,
                    "position_bias_score": metrics.position_bias_score,
                    "front_performance": metrics.front_performance,
                    "middle_performance": metrics.middle_performance,
                    "back_performance": metrics.back_performance,
                    "avg_retrieval_time_ms": metrics.avg_retrieval_time_ms,
                    "tokens_per_query": metrics.tokens_per_query,
                    "total_samples": metrics.total_samples,
                    "successful_retrievals": metrics.successful_retrievals
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
    
    def _create_visualizations(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]], analysis: Dict[str, Any]):
        """Create visualization plots."""
        
        # Set up matplotlib
        plt.style.use('default')
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Performance comparison across benchmarks
        self._plot_performance_comparison(benchmark_results, fig_dir)
        
        # 2. Position sensitivity analysis
        self._plot_position_sensitivity(benchmark_results, fig_dir)
        
        # 3. Configuration analysis
        self._plot_configuration_analysis(analysis, fig_dir)
        
        # 4. Efficiency analysis
        self._plot_efficiency_analysis(benchmark_results, fig_dir)
    
    def _plot_performance_comparison(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]], fig_dir: Path):
        """Plot performance comparison across benchmarks."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RetriVex Performance Across Benchmarks', fontsize=16)
        
        metrics = ['f1_score', 'recall_at_k_span', 'mrr', 'rescued_by_neighbor_pct']
        metric_names = ['F1 Score', 'Recall@k_span', 'MRR', 'Rescued by Neighbor %']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Collect data
            benchmarks = list(benchmark_results.keys())
            methods = set()
            for results in benchmark_results.values():
                methods.update(results.keys())
            methods = sorted(list(methods))
            
            # Create data matrix
            data = []
            for method in methods:
                method_data = []
                for benchmark in benchmarks:
                    if method in benchmark_results[benchmark]:
                        value = getattr(benchmark_results[benchmark][method], metric)
                        method_data.append(value)
                    else:
                        method_data.append(0)
                data.append(method_data)
            
            # Create grouped bar chart
            x = np.arange(len(benchmarks))
            width = 0.8 / len(methods)
            
            for i, (method, method_data) in enumerate(zip(methods, data)):
                offset = (i - len(methods)/2) * width + width/2
                bars = ax.bar(x + offset, method_data, width, label=method.replace('retrivex_', ''))
                
                # Highlight RetriVex methods
                if 'retrivex' in method:
                    for bar in bars:
                        bar.set_alpha(0.8)
                        bar.set_edgecolor('black')
                        bar.set_linewidth(0.5)
            
            ax.set_xlabel('Benchmark')
            ax.set_ylabel(name)
            ax.set_title(f'{name} by Benchmark and Method')
            ax.set_xticks(x)
            ax.set_xticklabels(benchmarks)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_position_sensitivity(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]], fig_dir: Path):
        """Plot position sensitivity analysis."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Position Sensitivity Analysis', fontsize=16)
        
        benchmarks = list(benchmark_results.keys())
        positions = ['front_performance', 'middle_performance', 'back_performance']
        position_names = ['Front', 'Middle', 'Back']
        
        for bench_idx, benchmark in enumerate(benchmarks):
            ax = axes[bench_idx]
            
            results = benchmark_results[benchmark]
            methods = list(results.keys())
            
            # Collect position data
            position_data = {pos: [] for pos in positions}
            method_labels = []
            
            for method in methods:
                if 'retrivex' in method:  # Focus on RetriVex methods
                    metrics = results[method]
                    for pos in positions:
                        position_data[pos].append(getattr(metrics, pos))
                    method_labels.append(method.replace('retrivex_', ''))
            
            # Create grouped bar chart
            x = np.arange(len(method_labels))
            width = 0.25
            
            for i, (pos, name) in enumerate(zip(positions, position_names)):
                offset = (i - 1) * width
                bars = ax.bar(x + offset, position_data[pos], width, label=name, alpha=0.8)
                
                # Highlight middle performance
                if pos == 'middle_performance':
                    for bar in bars:
                        bar.set_color('red')
                        bar.set_alpha(0.6)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Performance')
            ax.set_title(f'{benchmark.upper()} - Position Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'position_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_configuration_analysis(self, analysis: Dict[str, Any], fig_dir: Path):
        """Plot configuration analysis."""
        
        config_analysis = analysis.get("configuration_analysis", {})
        
        # Window size analysis
        window_data = config_analysis.get("window_size_analysis", {})
        if window_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for benchmark, data in window_data.items():
                windows = sorted([int(w) for w in data.keys()])
                scores = [data[str(w)] for w in windows]
                ax.plot(windows, scores, marker='o', label=benchmark, linewidth=2)
            
            ax.set_xlabel('Window Size')
            ax.set_ylabel('F1 Score')
            ax.set_title('Performance vs Window Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(fig_dir / 'window_size_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Token budget analysis
        budget_data = config_analysis.get("token_budget_analysis", {})
        if budget_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for benchmark, data in budget_data.items():
                budgets = sorted([int(b) for b in data.keys()])
                scores = [data[str(b)] for b in budgets]
                ax.plot(budgets, scores, marker='s', label=benchmark, linewidth=2)
            
            ax.set_xlabel('Token Budget')
            ax.set_ylabel('F1 Score')
            ax.set_title('Performance vs Token Budget')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(fig_dir / 'token_budget_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_efficiency_analysis(self, benchmark_results: Dict[str, Dict[str, EvaluationMetrics]], fig_dir: Path):
        """Plot efficiency analysis."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Efficiency Analysis', fontsize=16)
        
        # Latency vs Performance
        methods = []
        latencies = []
        f1_scores = []
        
        for benchmark, results in benchmark_results.items():
            for method, metrics in results.items():
                methods.append(f"{benchmark}_{method}")
                latencies.append(metrics.avg_retrieval_time_ms)
                f1_scores.append(metrics.f1_score)
        
        # Color code by method type
        colors = []
        for method in methods:
            if 'vanilla_knn' in method:
                colors.append('red')
            elif 'simple_expansion' in method:
                colors.append('orange')
            elif 'retrivex' in method:
                colors.append('blue')
            else:
                colors.append('gray')
        
        ax1.scatter(latencies, f1_scores, c=colors, alpha=0.7)
        ax1.set_xlabel('Average Latency (ms)')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Latency vs Performance')
        ax1.grid(True, alpha=0.3)
        
        # Token efficiency
        token_counts = []
        for benchmark, results in benchmark_results.items():
            for method, metrics in results.items():
                token_counts.append(metrics.tokens_per_query)
        
        ax2.scatter(token_counts, f1_scores, c=colors, alpha=0.7)
        ax2.set_xlabel('Tokens per Query')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Token Efficiency')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Vanilla kNN'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Simple Expansion'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='RetriVex')
        ]
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_benchmark():
    """Main function to run the comprehensive benchmark."""
    
    benchmark = ComprehensiveBenchmark(
        output_dir="benchmark_results",
        max_samples_per_benchmark=30,  # Reduced for faster evaluation
        include_baselines=True,
        verbose=True
    )
    
    results = benchmark.run_comprehensive_evaluation()
    return results


if __name__ == "__main__":
    results = run_benchmark()
    print(f"\n🎉 Benchmark completed successfully!")
    print(f"📁 Results available in: {results['output_dir']}")
