#!/usr/bin/env python3
"""
Comprehensive RetriVex Evaluation Script

This script runs the complete evaluation suite comparing RetriVex against
LongBench, RULER, and NIAH benchmarks with detailed reporting.

Usage:
    python -m retrivex.eval.run_evaluation [options]

Example:
    python -m retrivex.eval.run_evaluation --samples 50 --output benchmark_results
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from retrivex.eval.benchmark import ComprehensiveBenchmark


def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive RetriVex evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=30,
        help="Maximum samples per benchmark (default: 30)"
    )
    
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline methods evaluation"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation with reduced samples"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Adjust samples for quick evaluation
    if args.quick:
        args.samples = 10
    
    print("🚀 RetriVex Comprehensive Evaluation")
    print("=" * 50)
    print(f"📁 Output directory: {args.output}")
    print(f"📊 Samples per benchmark: {args.samples}")
    print(f"⚡ Include baselines: {not args.no_baselines}")
    print(f"🔍 Verbose: {args.verbose}")
    print()
    
    # Create benchmark runner
    benchmark = ComprehensiveBenchmark(
        output_dir=args.output,
        max_samples_per_benchmark=args.samples,
        include_baselines=not args.no_baselines,
        verbose=args.verbose
    )
    
    try:
        # Run evaluation
        results = benchmark.run_comprehensive_evaluation()
        
        print("\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {results['output_dir']}")
        print("\n📋 Summary:")
        
        # Print quick summary
        summary = results["analysis"]["summary"]
        print(f"🥇 Best overall method: {summary['best_overall_method']}")
        print(f"📈 Best F1 Score: {summary['best_f1_score']:.3f}")
        print(f"🎯 Best Recall@k: {summary['best_recall_at_k']:.3f}")
        
        # Print recommendations
        recommendations = results["analysis"]["recommendations"]
        print(f"\n💡 Recommendations:")
        for guideline in recommendations["general_guidelines"][:3]:
            print(f"   • {guideline}")
        
        print(f"\n📊 Detailed results and visualizations available in:")
        print(f"   {results['output_dir']}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
