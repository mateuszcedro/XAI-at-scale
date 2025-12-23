#!/usr/bin/env python3
"""
Create a comprehensive summary of XAI results across all seeds and models.

Aggregates XAI metrics (Saliency, IntegratedGradients, GradCAM, etc.) across multiple seeds
and models, computing mean and standard deviation for each XAI method.

Output:
- xai_summary_by_seed.csv: Results aggregated by seed
- xai_summary_by_model.csv: Results aggregated by model
- xai_summary_overall.csv: Overall summary across all seeds and models
- xai_analysis_report.txt: Detailed text report with interpretations
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_xai_metrics_for_seed(seed_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all XAI metrics for a given seed.
    
    Args:
        seed_dir: Path to seed directory (e.g., xai_results/seed_51)
        
    Returns:
        Dictionary mapping model names to their XAI metrics DataFrames
    """
    metrics = {}
    
    # Find all xai_metrics_*.csv files
    for csv_file in seed_dir.glob("xai_metrics_*.csv"):
        # Extract model name from filename (e.g., xai_metrics_ResNet50.csv -> ResNet50)
        model_name = csv_file.stem.replace("xai_metrics_", "")
        
        try:
            df = pd.read_csv(csv_file, index_col=0)
            metrics[model_name] = df
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    return metrics


def aggregate_xai_metrics_by_model(xai_results_dir: Path = Path("./xai_results")) -> Dict[str, Dict]:
    """
    Aggregate XAI metrics across all seeds for each model.
    
    Args:
        xai_results_dir: Root directory containing seed directories
        
    Returns:
        Dictionary mapping model names to aggregated metrics
    """
    
    # Dictionary to collect all metrics per model across seeds
    all_metrics_by_model = defaultdict(lambda: defaultdict(list))
    seeds = []
    
    # Iterate through seed directories
    for seed_dir in sorted(xai_results_dir.iterdir()):
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            seed_name = seed_dir.name
            seeds.append(seed_name)
            
            # Load metrics for this seed
            metrics = load_xai_metrics_for_seed(seed_dir)
            
            # Aggregate by model
            for model_name, df in metrics.items():
                for xai_method in df.index:
                    for metric_col in df.columns:
                        value = df.loc[xai_method, metric_col]
                        if pd.notna(value):  # Only include non-NaN values
                            all_metrics_by_model[model_name][(xai_method, metric_col)].append(value)
    
    # Calculate mean and std for each model
    aggregated = {}
    for model_name, metrics_dict in all_metrics_by_model.items():
        aggregated[model_name] = {}
        for (xai_method, metric_col), values in metrics_dict.items():
            aggregated[model_name][(xai_method, metric_col)] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values,
                'num_seeds': len(values)
            }
    
    return aggregated, seeds


def create_summary_by_model(aggregated: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a summary table aggregated by model across all seeds.
    
    Args:
        aggregated: Dictionary of aggregated metrics by model
        
    Returns:
        DataFrame with summary statistics
    """
    
    data = []
    
    for model_name in sorted(aggregated.keys()):
        model_data = aggregated[model_name]
        
        # Create row for this model
        row = {"Model": model_name}
        
        # Extract metrics for each XAI method
        xai_methods = sorted(set(method for method, _ in model_data.keys()))
        metric_types = sorted(set(metric for _, metric in model_data.keys()))
        
        for xai_method in xai_methods:
            for metric_type in metric_types:
                key = (xai_method, metric_type)
                if key in model_data:
                    stats = model_data[key]
                    col_name = f"{xai_method}_{metric_type}"
                    row[col_name] = f"{stats['mean']:.4f}±{stats['std']:.4f}"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_summary_overall(aggregated: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create an overall summary across all models and seeds.
    
    Args:
        aggregated: Dictionary of aggregated metrics by model
        
    Returns:
        DataFrame with overall statistics per XAI method
    """
    
    # Collect all metrics across all models
    all_metrics_by_xai = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_data in aggregated.items():
        for (xai_method, metric_type), stats in model_data.items():
            all_metrics_by_xai[xai_method][metric_type].extend(stats['values'])
    
    # Create summary
    data = []
    for xai_method in sorted(all_metrics_by_xai.keys()):
        row = {"XAI Method": xai_method}
        
        for metric_type in sorted(all_metrics_by_xai[xai_method].keys()):
            values = all_metrics_by_xai[xai_method][metric_type]
            row[f"{metric_type}_Mean"] = f"{np.mean(values):.4f}"
            row[f"{metric_type}_Std"] = f"{np.std(values):.4f}"
            row[f"{metric_type}_Min"] = f"{np.min(values):.4f}"
            row[f"{metric_type}_Max"] = f"{np.max(values):.4f}"
            row[f"{metric_type}_N"] = len(values)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_detailed_report(aggregated: Dict[str, Dict], seeds: List[str]) -> str:
    """
    Create a detailed text report with interpretations.
    
    Args:
        aggregated: Dictionary of aggregated metrics by model
        seeds: List of seed directories processed
        
    Returns:
        Formatted text report
    """
    
    report = []
    report.append("=" * 120)
    report.append("XAI RESULTS SUMMARY - ACROSS ALL SEEDS AND MODELS")
    report.append("=" * 120)
    report.append("")
    
    report.append(f"Number of seeds: {len(seeds)}")
    report.append(f"Seeds: {', '.join(seeds)}")
    report.append(f"Number of models: {len(aggregated)}")
    report.append("")
    
    report.append("=" * 120)
    report.append("XAI METHODS EVALUATED")
    report.append("=" * 120)
    report.append("")
    report.append("1. Saliency: Gradient-based attribution method")
    report.append("2. IntegratedGradients: Path integration-based attribution")
    report.append("3. GradCAM: Class Activation Mapping")
    report.append("4. GradientShap: Game theory-based attribution")
    report.append("5. FeaturePermutation: Importance by permutation")
    report.append("")
    
    report.append("=" * 120)
    report.append("EVALUATION METRICS")
    report.append("=" * 120)
    report.append("")
    report.append("RelevanceRankAccuracy: How well the XAI method ranks relevant features")
    report.append("  - Higher is better (range: 0-1)")
    report.append("  - Measures if important features get high attribution scores")
    report.append("")
    report.append("RelevanceMassAccuracy: How much attribution is concentrated on relevant features")
    report.append("  - Higher is better (range: 0-1)")
    report.append("  - Measures if explanations focus on important regions")
    report.append("")
    
    report.append("=" * 120)
    report.append("SUMMARY BY MODEL")
    report.append("=" * 120)
    report.append("")
    
    for model_name in sorted(aggregated.keys()):
        model_data = aggregated[model_name]
        report.append(f"\n{model_name}:")
        report.append("-" * 80)
        
        xai_methods = sorted(set(method for method, _ in model_data.keys()))
        
        for xai_method in xai_methods:
            report.append(f"\n  {xai_method}:")
            
            # RelevanceRankAccuracy
            key = (xai_method, "RelevanceRankAccuracy")
            if key in model_data:
                stats = model_data[key]
                report.append(f"    RelevanceRankAccuracy: {stats['mean']:.4f} ± {stats['std']:.4f}")
                report.append(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
            # RelevanceMassAccuracy
            key = (xai_method, "RelevanceMassAccuracy")
            if key in model_data:
                stats = model_data[key]
                report.append(f"    RelevanceMassAccuracy: {stats['mean']:.4f} ± {stats['std']:.4f}")
                report.append(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    report.append("")
    report.append("=" * 120)
    report.append("OVERALL RANKINGS")
    report.append("=" * 120)
    report.append("")
    
    # Calculate average scores for each XAI method
    xai_scores = defaultdict(list)
    for model_name, model_data in aggregated.items():
        for (xai_method, metric_type), stats in model_data.items():
            xai_scores[xai_method].append(stats['mean'])
    
    report.append("\nAverage RelevanceRankAccuracy by XAI Method:")
    report.append("-" * 80)
    rank_acc_scores = {}
    for model_name, model_data in aggregated.items():
        for (xai_method, metric_type), stats in model_data.items():
            if metric_type == "RelevanceRankAccuracy":
                if xai_method not in rank_acc_scores:
                    rank_acc_scores[xai_method] = []
                rank_acc_scores[xai_method].append(stats['mean'])
    
    for xai_method, scores in sorted(rank_acc_scores.items(), key=lambda x: -np.mean(x[1])):
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        report.append(f"  {xai_method}: {avg_score:.4f} ± {std_score:.4f}")
    
    report.append("\nAverage RelevanceMassAccuracy by XAI Method:")
    report.append("-" * 80)
    mass_acc_scores = {}
    for model_name, model_data in aggregated.items():
        for (xai_method, metric_type), stats in model_data.items():
            if metric_type == "RelevanceMassAccuracy":
                if xai_method not in mass_acc_scores:
                    mass_acc_scores[xai_method] = []
                mass_acc_scores[xai_method].append(stats['mean'])
    
    for xai_method, scores in sorted(mass_acc_scores.items(), key=lambda x: -np.mean(x[1])):
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        report.append(f"  {xai_method}: {avg_score:.4f} ± {std_score:.4f}")
    
    report.append("")
    report.append("=" * 120)
    report.append("TOP PERFORMING COMBINATIONS")
    report.append("=" * 120)
    report.append("")
    
    # Find top combinations for RelevanceRankAccuracy
    rank_acc_list = []
    for model_name, model_data in aggregated.items():
        for (xai_method, metric_type), stats in model_data.items():
            if metric_type == "RelevanceRankAccuracy":
                rank_acc_list.append({
                    'model': model_name,
                    'xai_method': xai_method,
                    'score': stats['mean'],
                    'std': stats['std']
                })
    
    report.append("\nTop 10 Model-XAI Method Combinations (RelevanceRankAccuracy):")
    report.append("-" * 80)
    for i, item in enumerate(sorted(rank_acc_list, key=lambda x: -x['score'])[:10], 1):
        report.append(f"{i:2d}. {item['model']:20s} + {item['xai_method']:20s}: {item['score']:.4f} ± {item['std']:.4f}")
    
    # Find top combinations for RelevanceMassAccuracy
    mass_acc_list = []
    for model_name, model_data in aggregated.items():
        for (xai_method, metric_type), stats in model_data.items():
            if metric_type == "RelevanceMassAccuracy":
                mass_acc_list.append({
                    'model': model_name,
                    'xai_method': xai_method,
                    'score': stats['mean'],
                    'std': stats['std']
                })
    
    report.append("\nTop 10 Model-XAI Method Combinations (RelevanceMassAccuracy):")
    report.append("-" * 80)
    for i, item in enumerate(sorted(mass_acc_list, key=lambda x: -x['score'])[:10], 1):
        report.append(f"{i:2d}. {item['model']:20s} + {item['xai_method']:20s}: {item['score']:.4f} ± {item['std']:.4f}")
    
    report.append("")
    report.append("=" * 120)
    report.append("INTERPRETATION GUIDE")
    report.append("=" * 120)
    report.append("")
    report.append("- Higher RelevanceRankAccuracy indicates the XAI method correctly identifies")
    report.append("  which features are important for the model's predictions")
    report.append("")
    report.append("- Higher RelevanceMassAccuracy indicates explanations are concentrated on")
    report.append("  the most important regions of the input")
    report.append("")
    report.append("- Standard deviation shows consistency across seeds:")
    report.append("  * Low std: Method is stable across different random initializations")
    report.append("  * High std: Method may be sensitive to training variations")
    report.append("")
    report.append("=" * 120)
    
    return "\n".join(report)


def main(xai_results_dir: str = "./xai_results", output_dir: str = "./xai_results"):
    """
    Main function to create XAI summary.
    
    Args:
        xai_results_dir: Directory containing XAI results organized by seed
        output_dir: Directory to save summary files
    """
    
    xai_results_path = Path(xai_results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading XAI metrics...")
    aggregated, seeds = aggregate_xai_metrics_by_model(xai_results_path)
    
    print(f"Aggregating results across {len(seeds)} seeds and {len(aggregated)} models...")
    
    # Create summaries
    summary_by_model = create_summary_by_model(aggregated)
    summary_overall = create_summary_overall(aggregated)
    detailed_report = create_detailed_report(aggregated, seeds)
    
    # Save summaries
    summary_by_model_path = output_path / "xai_summary_by_model.csv"
    summary_by_model.to_csv(summary_by_model_path, index=False)
    print(f"✓ Saved: {summary_by_model_path}")
    
    summary_overall_path = output_path / "xai_summary_overall.csv"
    summary_overall.to_csv(summary_overall_path, index=False)
    print(f"✓ Saved: {summary_overall_path}")
    
    report_path = output_path / "xai_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(detailed_report)
    print(f"✓ Saved: {report_path}")
    
    # Save summary tables to text files
    summary_by_model_txt_path = output_path / "xai_summary_by_model.txt"
    with open(summary_by_model_txt_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("XAI SUMMARY BY MODEL\n")
        f.write("=" * 120 + "\n\n")
        f.write(summary_by_model.to_string(index=False))
        f.write("\n\n")
    print(f"✓ Saved: {summary_by_model_txt_path}")
    
    summary_overall_txt_path = output_path / "xai_summary_overall.txt"
    with open(summary_overall_txt_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("XAI OVERALL SUMMARY\n")
        f.write("=" * 120 + "\n\n")
        f.write(summary_overall.to_string(index=False))
        f.write("\n\n")
    print(f"✓ Saved: {summary_overall_txt_path}")
    
    print("\n✓ All XAI summary files created successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create summary of XAI results across all seeds and models"
    )
    parser.add_argument(
        "--xai-dir",
        type=str,
        default="./xai_results",
        help="Directory containing XAI results (default: ./xai_results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./xai_results",
        help="Directory to save summary files (default: ./xai_results)"
    )
    
    args = parser.parse_args()
    main(args.xai_dir, args.output_dir)
