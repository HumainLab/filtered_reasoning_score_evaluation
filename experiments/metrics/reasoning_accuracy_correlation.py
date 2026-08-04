#!/usr/bin/env python3
"""
Experiment: Reasoning Score vs Accuracy Correlation Analysis

This script investigates:
1. How well reasoning scores from N samples predict full dataset accuracy
2. Within-model correlation: Do high reasoning scores predict correct answers?
3. Cross-model correlation: Do models with higher reasoning scores have higher accuracy?
4. Stability analysis: How many samples needed for reliable reasoning score estimate?

Usage:
    python experiments/reasoning_accuracy_correlation.py [--output-dir OUTPUT_DIR] [--no-plots]
"""

import json
import glob
import os
import argparse
import numpy as np
from scipy import stats
from collections import defaultdict
from datetime import datetime

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


def load_cot_analysis_files(base_path: str) -> list:
    """Load all COT analysis JSON files and extract relevant data."""
    json_files = glob.glob(os.path.join(base_path, '**/*.json'), recursive=True)
    
    results = []
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            model = data.get('model')
            dataset = data.get('dataset')
            prompt_type = data.get('prompt_type', 'unknown')
            
            # Skip if no model metadata
            if not model:
                continue
            
            samples = data.get('per_sample', [])
            if len(samples) < 10:
                continue
            
            # Extract per-sample data
            sample_data = []
            for s in samples:
                overall = s.get('overall', 0)
                faithfulness = s.get('faithfulness', 0)
                utility = s.get('utility', 0)
                coherence = s.get('coherence', 0)
                factuality = s.get('factuality', 0)
                
                evidence = s.get('evidence', {})
                correct = evidence.get('final_correct')
                
                if correct is not None and overall > 0:
                    sample_data.append({
                        'overall': overall,
                        'faithfulness': faithfulness,
                        'utility': utility,
                        'coherence': coherence,
                        'factuality': factuality,
                        'correct': 1 if correct else 0
                    })
            
            if len(sample_data) < 10:
                continue
            
            results.append({
                'model': model,
                'dataset': dataset,
                'prompt_type': prompt_type,
                'file': jf,
                'samples': sample_data,
                'n_samples': len(sample_data)
            })
            
        except Exception as e:
            print(f"Error loading {jf}: {e}")
    
    return results


def compute_statistics(data: list) -> dict:
    """Compute aggregate statistics for a model-dataset combo."""
    overall_scores = np.array([s['overall'] for s in data])
    correct_flags = np.array([s['correct'] for s in data])
    
    accuracy = np.mean(correct_flags)
    reasoning = np.mean(overall_scores)
    
    # Within-model correlation
    if np.std(overall_scores) > 0 and np.std(correct_flags) > 0:
        within_corr, within_p = stats.pearsonr(overall_scores, correct_flags)
    else:
        within_corr, within_p = 0, 1
    
    # Reasoning by correctness
    correct_reasoning = [s['overall'] for s in data if s['correct'] == 1]
    incorrect_reasoning = [s['overall'] for s in data if s['correct'] == 0]
    
    return {
        'accuracy': accuracy,
        'reasoning': reasoning,
        'within_correlation': within_corr,
        'within_p_value': within_p,
        'correct_reasoning_mean': np.mean(correct_reasoning) if correct_reasoning else 0,
        'incorrect_reasoning_mean': np.mean(incorrect_reasoning) if incorrect_reasoning else 0,
        'n_correct': len(correct_reasoning),
        'n_incorrect': len(incorrect_reasoning)
    }


def run_bootstrap_sampling(data: list, n_values: list, n_bootstrap: int = 100) -> dict:
    """Run bootstrap sampling experiment for different N values."""
    overall_scores = np.array([s['overall'] for s in data])
    correct_flags = np.array([s['correct'] for s in data])
    
    full_accuracy = np.mean(correct_flags)
    full_reasoning = np.mean(overall_scores)
    
    results = {}
    for N in n_values:
        if N > len(data):
            continue
        
        acc_samples = []
        reas_samples = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(data), size=N, replace=False)
            acc_samples.append(np.mean(correct_flags[indices]))
            reas_samples.append(np.mean(overall_scores[indices]))
        
        acc_samples = np.array(acc_samples)
        reas_samples = np.array(reas_samples)
        
        # Correlation between sampled reasoning and sampled accuracy
        if np.std(reas_samples) > 0 and np.std(acc_samples) > 0:
            sample_corr, _ = stats.pearsonr(reas_samples, acc_samples)
        else:
            sample_corr = 0
        
        results[N] = {
            'accuracy_mean': np.mean(acc_samples),
            'accuracy_std': np.std(acc_samples),
            'accuracy_error': np.abs(np.mean(acc_samples) - full_accuracy),
            'reasoning_mean': np.mean(reas_samples),
            'reasoning_std': np.std(reas_samples),
            'reasoning_error': np.abs(np.mean(reas_samples) - full_reasoning),
            'sample_correlation': sample_corr
        }
    
    return results


def generate_plots(all_results: list, output_dir: str):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots - matplotlib not available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scatter plot: Reasoning vs Accuracy across all model-dataset combos
    fig, ax = plt.subplots(figsize=(10, 8))
    
    datasets = list(set(r['dataset'] for r in all_results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    dataset_colors = {d: c for d, c in zip(datasets, colors)}
    
    for r in all_results:
        stats_data = r['stats']
        ax.scatter(stats_data['reasoning'] * 100, stats_data['accuracy'] * 100,
                   c=[dataset_colors[r['dataset']]], label=r['dataset'],
                   s=100, alpha=0.7)
        ax.annotate(r['model'][:15], 
                    (stats_data['reasoning'] * 100, stats_data['accuracy'] * 100),
                    fontsize=6, alpha=0.7)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    ax.set_xlabel('Reasoning Score (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Reasoning Score vs Accuracy Across Models', fontsize=14)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='y=x')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reasoning_vs_accuracy_scatter.png'), dpi=150)
    plt.close()
    
    # 2. Within-model correlation distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    correlations = [r['stats']['within_correlation'] for r in all_results]
    ax.hist(correlations, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(correlations), color='red', linestyle='--', 
               label=f'Mean: {np.mean(correlations):.3f}')
    ax.set_xlabel('Within-Model Correlation (Reasoning vs Correct)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Within-Model Correlations', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'within_model_correlation_dist.png'), dpi=150)
    plt.close()
    
    # 3. Bootstrap convergence plot (for a representative model)
    # Find a model with good data
    representative = None
    for r in all_results:
        if r['dataset'] == 'gsm8k' and r['n_samples'] > 1000:
            representative = r
            break
    
    if representative and representative.get('bootstrap'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        bootstrap = representative['bootstrap']
        n_values = sorted(bootstrap.keys())
        
        # Accuracy convergence
        ax = axes[0]
        means = [bootstrap[n]['accuracy_mean'] * 100 for n in n_values]
        stds = [bootstrap[n]['accuracy_std'] * 100 for n in n_values]
        true_acc = representative['stats']['accuracy'] * 100
        
        ax.errorbar(n_values, means, yerr=stds, fmt='o-', capsize=5, label='Sampled')
        ax.axhline(true_acc, color='red', linestyle='--', label=f'True: {true_acc:.1f}%')
        ax.set_xlabel('Sample Size (N)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Accuracy Convergence ({representative["model"]})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reasoning convergence
        ax = axes[1]
        means = [bootstrap[n]['reasoning_mean'] * 100 for n in n_values]
        stds = [bootstrap[n]['reasoning_std'] * 100 for n in n_values]
        true_reas = representative['stats']['reasoning'] * 100
        
        ax.errorbar(n_values, means, yerr=stds, fmt='o-', capsize=5, label='Sampled')
        ax.axhline(true_reas, color='red', linestyle='--', label=f'True: {true_reas:.1f}%')
        ax.set_xlabel('Sample Size (N)', fontsize=12)
        ax.set_ylabel('Reasoning Score (%)', fontsize=12)
        ax.set_title(f'Reasoning Score Convergence ({representative["model"]})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bootstrap_convergence.png'), dpi=150)
        plt.close()
    
    # 4. Reasoning score by correctness (box plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    correct_scores = []
    incorrect_scores = []
    labels = []
    
    for r in all_results[:10]:  # Limit to first 10 for readability
        stats_data = r['stats']
        if stats_data['n_correct'] > 0 and stats_data['n_incorrect'] > 0:
            correct_scores.append(stats_data['correct_reasoning_mean'] * 100)
            incorrect_scores.append(stats_data['incorrect_reasoning_mean'] * 100)
            labels.append(f"{r['model'][:12]}\n{r['dataset']}")
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, correct_scores, width, label='Correct', color='green', alpha=0.7)
    ax.bar(x + width/2, incorrect_scores, width, label='Incorrect', color='red', alpha=0.7)
    
    ax.set_xlabel('Model-Dataset', fontsize=12)
    ax.set_ylabel('Mean Reasoning Score (%)', fontsize=12)
    ax.set_title('Reasoning Score by Answer Correctness', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reasoning_by_correctness.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def generate_report(all_results: list, output_path: str):
    """Generate a text report of the analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("REASONING SCORE VS ACCURACY CORRELATION ANALYSIS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary statistics
    lines.append("1. SUMMARY STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total model-dataset combinations: {len(all_results)}")
    
    accuracies = [r['stats']['accuracy'] for r in all_results]
    reasonings = [r['stats']['reasoning'] for r in all_results]
    within_corrs = [r['stats']['within_correlation'] for r in all_results]
    
    lines.append(f"Accuracy range: {min(accuracies)*100:.1f}% - {max(accuracies)*100:.1f}%")
    lines.append(f"Reasoning range: {min(reasonings)*100:.1f}% - {max(reasonings)*100:.1f}%")
    lines.append(f"Within-model correlation range: {min(within_corrs):.3f} - {max(within_corrs):.3f}")
    lines.append("")
    
    # Cross-model correlation
    lines.append("2. CROSS-MODEL CORRELATION")
    lines.append("-" * 40)
    if len(accuracies) >= 3:
        corr, p = stats.pearsonr(accuracies, reasonings)
        lines.append(f"Pearson correlation (Reasoning vs Accuracy): {corr:.4f}")
        lines.append(f"P-value: {p:.4e}")
        
        slope, intercept, r_value, _, std_err = stats.linregress(reasonings, accuracies)
        lines.append(f"Linear regression: Accuracy = {slope:.3f} * Reasoning + {intercept:.3f}")
        lines.append(f"R-squared: {r_value**2:.4f}")
    lines.append("")
    
    # Per-dataset analysis
    lines.append("3. PER-DATASET ANALYSIS")
    lines.append("-" * 40)
    
    by_dataset = defaultdict(list)
    for r in all_results:
        by_dataset[r['dataset']].append(r)
    
    for dataset in sorted(by_dataset.keys()):
        dataset_results = by_dataset[dataset]
        lines.append(f"\n{dataset.upper()} ({len(dataset_results)} models):")
        
        if len(dataset_results) >= 3:
            acc = [r['stats']['accuracy'] for r in dataset_results]
            reas = [r['stats']['reasoning'] for r in dataset_results]
            corr, p = stats.pearsonr(acc, reas)
            lines.append(f"  Cross-model correlation: {corr:.3f} (p={p:.4f})")
        
        lines.append(f"  {'Model':<30} {'Accuracy':>10} {'Reasoning':>12} {'Within-Corr':>12}")
        lines.append(f"  {'-'*64}")
        
        for r in sorted(dataset_results, key=lambda x: -x['stats']['accuracy']):
            s = r['stats']
            lines.append(f"  {r['model']:<30} {s['accuracy']*100:>9.2f}% {s['reasoning']*100:>11.2f}% {s['within_correlation']:>11.3f}")
    
    lines.append("")
    
    # Bootstrap analysis
    lines.append("4. BOOTSTRAP SAMPLING ANALYSIS")
    lines.append("-" * 40)
    lines.append("How many samples needed for stable estimates?")
    lines.append("")
    
    # Find results with bootstrap data
    bootstrap_results = [r for r in all_results if r.get('bootstrap')]
    if bootstrap_results:
        r = bootstrap_results[0]
        lines.append(f"Example: {r['model']} on {r['dataset']} (N={r['n_samples']})")
        lines.append(f"Full accuracy: {r['stats']['accuracy']*100:.2f}%, Full reasoning: {r['stats']['reasoning']*100:.2f}%")
        lines.append("")
        lines.append(f"{'N':>6} | {'Acc Mean':>10} | {'Acc Std':>10} | {'Reas Mean':>10} | {'Reas Std':>10}")
        lines.append("-" * 60)
        
        for n in sorted(r['bootstrap'].keys()):
            b = r['bootstrap'][n]
            lines.append(f"{n:>6} | {b['accuracy_mean']*100:>9.2f}% | {b['accuracy_std']*100:>9.2f}% | {b['reasoning_mean']*100:>9.2f}% | {b['reasoning_std']*100:>9.2f}%")
    
    lines.append("")
    
    # Key findings
    lines.append("5. KEY FINDINGS")
    lines.append("-" * 40)
    
    # Find high/low correlation examples
    high_corr = [r for r in all_results if r['stats']['within_correlation'] > 0.7]
    low_corr = [r for r in all_results if r['stats']['within_correlation'] < 0.2]
    
    lines.append(f"High within-model correlation (>0.7): {len(high_corr)} combos")
    for r in high_corr[:5]:
        lines.append(f"  - {r['model']} on {r['dataset']}: {r['stats']['within_correlation']:.3f}")
    
    lines.append(f"\nLow within-model correlation (<0.2): {len(low_corr)} combos")
    for r in low_corr[:5]:
        lines.append(f"  - {r['model']} on {r['dataset']}: {r['stats']['within_correlation']:.3f}")
    
    # Anomalies (high reasoning, low accuracy)
    anomalies = [r for r in all_results 
                 if r['stats']['reasoning'] > 0.9 and r['stats']['accuracy'] < 0.3]
    if anomalies:
        lines.append(f"\nAnomalies (reasoning > 90%, accuracy < 30%): {len(anomalies)} combos")
        for r in anomalies:
            s = r['stats']
            lines.append(f"  - {r['model']} on {r['dataset']}: {s['accuracy']*100:.1f}% acc, {s['reasoning']*100:.1f}% reas")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Reasoning Score vs Accuracy Correlation Analysis')
    parser.add_argument('--cot-dir', type=str, 
                        default='evaluation/exports/cot_analysis',
                        help='Directory containing COT analysis JSON files')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/reasoning_accuracy',
                        help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--n-bootstrap', type=int, default=100,
                        help='Number of bootstrap samples')
    
    args = parser.parse_args()
    
    # Find the base path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Go up one level from experiments/
    cot_dir = os.path.join(base_dir, args.cot_dir)
    output_dir = os.path.join(base_dir, args.output_dir)
    
    print(f"Loading COT analysis files from: {cot_dir}")
    
    # Load data
    raw_results = load_cot_analysis_files(cot_dir)
    print(f"Loaded {len(raw_results)} model-dataset combinations")
    
    if not raw_results:
        print("No data found. Check the COT analysis directory.")
        return
    
    # Process each result
    n_values = [10, 25, 50, 100, 200, 500, 1000]
    all_results = []
    
    for r in raw_results:
        # Compute statistics
        stats_data = compute_statistics(r['samples'])
        
        # Run bootstrap sampling
        bootstrap = run_bootstrap_sampling(r['samples'], n_values, args.n_bootstrap)
        
        all_results.append({
            'model': r['model'],
            'dataset': r['dataset'],
            'prompt_type': r['prompt_type'],
            'n_samples': r['n_samples'],
            'stats': stats_data,
            'bootstrap': bootstrap
        })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    generate_report(all_results, report_path)
    
    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        generate_plots(all_results, output_dir)
    
    # Save raw results as JSON
    json_results = []
    for r in all_results:
        json_results.append({
            'model': r['model'],
            'dataset': r['dataset'],
            'prompt_type': r['prompt_type'],
            'n_samples': r['n_samples'],
            'accuracy': r['stats']['accuracy'],
            'reasoning': r['stats']['reasoning'],
            'within_correlation': r['stats']['within_correlation'],
            'correct_reasoning_mean': r['stats']['correct_reasoning_mean'],
            'incorrect_reasoning_mean': r['stats']['incorrect_reasoning_mean'],
            'bootstrap': {str(k): v for k, v in r['bootstrap'].items()}
        })
    
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == '__main__':
    main()
