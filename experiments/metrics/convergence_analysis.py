#!/usr/bin/env python3
"""
Experiment: Convergence Rate of Reasoning Scores vs Accuracy

Measures how quickly reasoning scores converge to their true values compared
to accuracy as sample size increases, using bootstrap resampling on CoT
Zero-shot evaluation data.

Key insight: Reasoning scores are continuous (0-1) while accuracy is binary
(0/1). The mean of continuous values should have lower variance than the mean
of Bernoulli variables at the same sample size, making reasoning a more
sample-efficient evaluation metric.

Usage:
    python experiments/convergence_analysis.py [--output-dir DIR] [--n-bootstrap N] [--no-plots]
"""

import json
import glob
import os
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cot_analysis_files(base_path: str) -> list:
    """Load all COT analysis JSON files and extract relevant data.

    Reuses the same loading logic from reasoning_accuracy_correlation.py.
    """
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


# ---------------------------------------------------------------------------
# Core bootstrap convergence analysis
# ---------------------------------------------------------------------------

REASONING_DIMS = ['overall', 'faithfulness', 'utility', 'coherence', 'factuality']


def run_convergence_bootstrap(samples: list, n_values: list,
                              n_bootstrap: int = 1000) -> dict:
    """Run bootstrap convergence analysis for a single model-dataset combo.

    For each sample size N, draw N items without replacement B times and
    compute convergence statistics for accuracy and each reasoning dimension.

    Returns a dict keyed by N, each containing metrics for accuracy and
    reasoning dimensions.
    """
    # Pre-extract arrays for speed
    correct_flags = np.array([s['correct'] for s in samples])
    dim_arrays = {}
    for dim in REASONING_DIMS:
        dim_arrays[dim] = np.array([s[dim] for s in samples])

    total_n = len(samples)

    # True (full-dataset) values
    true_accuracy = float(np.mean(correct_flags))
    true_reasoning = {dim: float(np.mean(dim_arrays[dim])) for dim in REASONING_DIMS}

    results = {}
    for N in n_values:
        if N > total_n:
            continue

        # Bootstrap draws
        acc_draws = np.empty(n_bootstrap)
        reas_draws = {dim: np.empty(n_bootstrap) for dim in REASONING_DIMS}

        for b in range(n_bootstrap):
            indices = np.random.choice(total_n, size=N, replace=False)
            acc_draws[b] = np.mean(correct_flags[indices])
            for dim in REASONING_DIMS:
                reas_draws[dim][b] = np.mean(dim_arrays[dim][indices])

        # --- Accuracy metrics ---
        acc_std = float(np.std(acc_draws))
        acc_mae = float(np.mean(np.abs(acc_draws - true_accuracy)))
        acc_rel_err = acc_mae / true_accuracy if true_accuracy > 0 else float('inf')
        acc_ci_half = 1.96 * acc_std
        acc_within_5 = float(np.mean(np.abs(acc_draws - true_accuracy) <= 0.05 * max(true_accuracy, 1e-9)))
        acc_within_10 = float(np.mean(np.abs(acc_draws - true_accuracy) <= 0.10 * max(true_accuracy, 1e-9)))

        acc_metrics = {
            'mean': float(np.mean(acc_draws)),
            'std': acc_std,
            'mae': acc_mae,
            'relative_error': acc_rel_err,
            'ci95_half': acc_ci_half,
            'p_within_5pct': acc_within_5,
            'p_within_10pct': acc_within_10,
        }

        # --- Reasoning metrics (per dimension) ---
        reas_metrics = {}
        for dim in REASONING_DIMS:
            draws = reas_draws[dim]
            true_val = true_reasoning[dim]
            r_std = float(np.std(draws))
            r_mae = float(np.mean(np.abs(draws - true_val)))
            r_rel_err = r_mae / true_val if true_val > 0 else float('inf')
            r_ci_half = 1.96 * r_std
            r_within_5 = float(np.mean(np.abs(draws - true_val) <= 0.05 * max(true_val, 1e-9)))
            r_within_10 = float(np.mean(np.abs(draws - true_val) <= 0.10 * max(true_val, 1e-9)))

            reas_metrics[dim] = {
                'mean': float(np.mean(draws)),
                'std': r_std,
                'mae': r_mae,
                'relative_error': r_rel_err,
                'ci95_half': r_ci_half,
                'p_within_5pct': r_within_5,
                'p_within_10pct': r_within_10,
            }

        results[N] = {
            'accuracy': acc_metrics,
            'reasoning': reas_metrics,
        }

    return {
        'true_accuracy': true_accuracy,
        'true_reasoning': true_reasoning,
        'bootstrap': results,
    }


def find_convergence_n(bootstrap_results: dict, threshold: float = 0.02) -> dict:
    """Find the smallest N where 95% CI half-width < threshold for each metric.

    Returns dict with keys 'accuracy' and each reasoning dimension, values
    are the N needed (or None if never reached).
    """
    sorted_ns = sorted(bootstrap_results.keys())

    conv = {}

    # Accuracy
    conv['accuracy'] = None
    for n in sorted_ns:
        if bootstrap_results[n]['accuracy']['ci95_half'] < threshold:
            conv['accuracy'] = n
            break

    # Reasoning dimensions
    for dim in REASONING_DIMS:
        conv[dim] = None
        for n in sorted_ns:
            if bootstrap_results[n]['reasoning'][dim]['ci95_half'] < threshold:
                conv[dim] = n
                break

    return conv


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_across_combos(all_results: list) -> dict:
    """Aggregate convergence metrics across all model-dataset combos.

    For each N, compute median and IQR of std / relative_error / ci95_half
    across all combos that have that N.
    """
    # Collect per-N data
    per_n_acc_std = defaultdict(list)
    per_n_acc_rel = defaultdict(list)
    per_n_acc_ci = defaultdict(list)
    per_n_reas_std = {dim: defaultdict(list) for dim in REASONING_DIMS}
    per_n_reas_rel = {dim: defaultdict(list) for dim in REASONING_DIMS}
    per_n_reas_ci = {dim: defaultdict(list) for dim in REASONING_DIMS}

    for r in all_results:
        for n, data in r['convergence']['bootstrap'].items():
            per_n_acc_std[n].append(data['accuracy']['std'])
            per_n_acc_rel[n].append(data['accuracy']['relative_error'])
            per_n_acc_ci[n].append(data['accuracy']['ci95_half'])
            for dim in REASONING_DIMS:
                per_n_reas_std[dim][n].append(data['reasoning'][dim]['std'])
                per_n_reas_rel[dim][n].append(data['reasoning'][dim]['relative_error'])
                per_n_reas_ci[dim][n].append(data['reasoning'][dim]['ci95_half'])

    def summarize(data_dict):
        result = {}
        for n in sorted(data_dict.keys()):
            arr = np.array(data_dict[n])
            result[n] = {
                'median': float(np.median(arr)),
                'q25': float(np.percentile(arr, 25)),
                'q75': float(np.percentile(arr, 75)),
                'mean': float(np.mean(arr)),
                'count': len(arr),
            }
        return result

    agg = {
        'accuracy': {
            'std': summarize(per_n_acc_std),
            'relative_error': summarize(per_n_acc_rel),
            'ci95_half': summarize(per_n_acc_ci),
        },
        'reasoning': {}
    }
    for dim in REASONING_DIMS:
        agg['reasoning'][dim] = {
            'std': summarize(per_n_reas_std[dim]),
            'relative_error': summarize(per_n_reas_rel[dim]),
            'ci95_half': summarize(per_n_reas_ci[dim]),
        }

    return agg


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_aggregate_convergence(agg: dict, output_dir: str):
    """Plot 1: Aggregate convergence curves (std) -- accuracy vs reasoning overall."""
    fig, ax = plt.subplots(figsize=(10, 6))

    acc_data = agg['accuracy']['std']
    reas_data = agg['reasoning']['overall']['std']
    ns = sorted(set(acc_data.keys()) & set(reas_data.keys()))

    # Accuracy
    acc_med = [acc_data[n]['median'] * 100 for n in ns]
    acc_q25 = [acc_data[n]['q25'] * 100 for n in ns]
    acc_q75 = [acc_data[n]['q75'] * 100 for n in ns]
    ax.plot(ns, acc_med, 'o-', color='#e74c3c', linewidth=2, label='Accuracy (median)')
    ax.fill_between(ns, acc_q25, acc_q75, alpha=0.2, color='#e74c3c', label='Accuracy (IQR)')

    # Reasoning overall
    reas_med = [reas_data[n]['median'] * 100 for n in ns]
    reas_q25 = [reas_data[n]['q25'] * 100 for n in ns]
    reas_q75 = [reas_data[n]['q75'] * 100 for n in ns]
    ax.plot(ns, reas_med, 's-', color='#2980b9', linewidth=2, label='Reasoning (median)')
    ax.fill_between(ns, reas_q25, reas_q75, alpha=0.2, color='#2980b9', label='Reasoning (IQR)')

    ax.set_xlabel('Sample Size (N)', fontsize=13)
    ax.set_ylabel('Std of Bootstrap Estimates (pp)', fontsize=13)
    ax.set_title('Convergence Rate: Accuracy vs Reasoning Score\n(CoT Zero-shot, aggregated across all model-dataset combos)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_aggregate_convergence.png'), dpi=150)
    plt.close()
    print("  -> 1_aggregate_convergence.png")


def plot_relative_error(agg: dict, output_dir: str):
    """Plot 2: Relative error comparison -- accuracy vs reasoning overall."""
    fig, ax = plt.subplots(figsize=(10, 6))

    acc_data = agg['accuracy']['relative_error']
    reas_data = agg['reasoning']['overall']['relative_error']
    ns = sorted(set(acc_data.keys()) & set(reas_data.keys()))

    acc_med = [acc_data[n]['median'] * 100 for n in ns]
    acc_q25 = [acc_data[n]['q25'] * 100 for n in ns]
    acc_q75 = [acc_data[n]['q75'] * 100 for n in ns]
    ax.plot(ns, acc_med, 'o-', color='#e74c3c', linewidth=2, label='Accuracy (median)')
    ax.fill_between(ns, acc_q25, acc_q75, alpha=0.2, color='#e74c3c', label='Accuracy (IQR)')

    reas_med = [reas_data[n]['median'] * 100 for n in ns]
    reas_q25 = [reas_data[n]['q25'] * 100 for n in ns]
    reas_q75 = [reas_data[n]['q75'] * 100 for n in ns]
    ax.plot(ns, reas_med, 's-', color='#2980b9', linewidth=2, label='Reasoning (median)')
    ax.fill_between(ns, reas_q25, reas_q75, alpha=0.2, color='#2980b9', label='Reasoning (IQR)')

    ax.set_xlabel('Sample Size (N)', fontsize=13)
    ax.set_ylabel('Relative Error (%)', fontsize=13)
    ax.set_title('Relative Error Convergence: Accuracy vs Reasoning Score\n(CoT Zero-shot, aggregated across all model-dataset combos)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_relative_error.png'), dpi=150)
    plt.close()
    print("  -> 2_relative_error.png")


def plot_samples_to_converge(all_results: list, output_dir: str):
    """Plot 3: Grouped bar chart -- samples to converge for each model, averaged across datasets.

    When accuracy never converges within the tested range for a dataset, that
    dataset contributes the cap value (max_n) to the average.  The bar is drawn
    with hatching and labelled ">N" to indicate it is a lower bound.
    """
    # Determine max tested N from the data
    max_n = max(n for r in all_results for n in r['convergence']['bootstrap'].keys())

    # Group by model
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r['model']].append(r)

    models = sorted(by_model.keys())
    acc_means = []
    reas_means = []
    acc_has_cap = []   # True if any dataset was capped at max_n
    reas_has_cap = []

    for model in models:
        acc_ns = []
        reas_ns = []
        a_capped = False
        r_capped = False
        for r in by_model[model]:
            cn = r['convergence_n']
            if cn['accuracy'] is not None:
                acc_ns.append(cn['accuracy'])
            else:
                acc_ns.append(max_n)  # cap at max tested N
                a_capped = True
            if cn['overall'] is not None:
                reas_ns.append(cn['overall'])
            else:
                reas_ns.append(max_n)
                r_capped = True
        acc_means.append(np.mean(acc_ns))
        reas_means.append(np.mean(reas_ns))
        acc_has_cap.append(a_capped)
        reas_has_cap.append(r_capped)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35

    # Draw accuracy bars -- hatched if capped
    for i, model in enumerate(models):
        hatch = '///' if acc_has_cap[i] else None
        ax.bar(x[i] - width / 2, acc_means[i], width, color='#e74c3c',
               alpha=0.8, hatch=hatch, edgecolor='white' if hatch else None,
               label='Accuracy' if i == 0 else None)

    # Draw reasoning bars -- hatched if capped
    for i, model in enumerate(models):
        hatch = '///' if reas_has_cap[i] else None
        ax.bar(x[i] + width / 2, reas_means[i], width, color='#2980b9',
               alpha=0.8, hatch=hatch, edgecolor='white' if hatch else None,
               label='Reasoning' if i == 0 else None)

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Avg Samples to Converge (N)', fontsize=13)
    ax.set_title('Samples Needed for Convergence (95% CI < 2pp)\nAvg across datasets, CoT Zero-shot  (hatched = includes >max estimates)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate bar values
    for i in range(len(models)):
        # Accuracy annotation
        h = acc_means[i]
        prefix = ">" if acc_has_cap[i] else ""
        ax.annotate(f'{prefix}{h:.0f}', xy=(x[i] - width / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8,
                    fontstyle='italic' if acc_has_cap[i] else 'normal')
        # Reasoning annotation
        h = reas_means[i]
        prefix = ">" if reas_has_cap[i] else ""
        ax.annotate(f'{prefix}{h:.0f}', xy=(x[i] + width / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8,
                    fontstyle='italic' if reas_has_cap[i] else 'normal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_samples_to_converge.png'), dpi=150)
    plt.close()
    print("  -> 3_samples_to_converge.png")


def plot_per_dataset_convergence(all_results: list, output_dir: str):
    """Plot 4: Per-dataset convergence -- one subplot per dataset."""
    by_dataset = defaultdict(list)
    for r in all_results:
        by_dataset[r['dataset']].append(r)

    datasets = sorted(by_dataset.keys())
    n_datasets = len(datasets)
    cols = 3
    rows = (n_datasets + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_2d(axes)

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        # Collect all N values and per-N std across models
        acc_per_n = defaultdict(list)
        reas_per_n = defaultdict(list)
        for r in by_dataset[dataset]:
            for n, data in r['convergence']['bootstrap'].items():
                acc_per_n[n].append(data['accuracy']['std'])
                reas_per_n[n].append(data['reasoning']['overall']['std'])

        ns = sorted(set(acc_per_n.keys()) & set(reas_per_n.keys()))
        if not ns:
            ax.set_title(f'{dataset} (no data)')
            continue

        acc_means = [np.mean(acc_per_n[n]) * 100 for n in ns]
        reas_means = [np.mean(reas_per_n[n]) * 100 for n in ns]

        ax.plot(ns, acc_means, 'o-', color='#e74c3c', linewidth=2, label='Accuracy')
        ax.plot(ns, reas_means, 's-', color='#2980b9', linewidth=2, label='Reasoning')

        n_models = len(by_dataset[dataset])
        avg_samples = int(np.mean([r['n_samples'] for r in by_dataset[dataset]]))
        ax.set_title(f'{dataset}\n({n_models} models, ~{avg_samples} samples)', fontsize=11)
        ax.set_xlabel('N', fontsize=10)
        ax.set_ylabel('Std (pp)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(datasets), rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle('Per-Dataset Convergence: Accuracy vs Reasoning Std\n(averaged across models, CoT Zero-shot)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_per_dataset_convergence.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  -> 4_per_dataset_convergence.png")


def plot_subscore_convergence(agg: dict, output_dir: str):
    """Plot 5: Reasoning sub-score convergence -- one line per dimension."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'overall': '#2c3e50',
        'faithfulness': '#e74c3c',
        'utility': '#27ae60',
        'coherence': '#f39c12',
        'factuality': '#8e44ad',
    }
    markers = {
        'overall': 's',
        'faithfulness': 'o',
        'utility': '^',
        'coherence': 'D',
        'factuality': 'v',
    }

    for dim in REASONING_DIMS:
        data = agg['reasoning'][dim]['std']
        ns = sorted(data.keys())
        medians = [data[n]['median'] * 100 for n in ns]
        ax.plot(ns, medians, f'{markers[dim]}-', color=colors[dim],
                linewidth=2, label=dim.capitalize(), markersize=7)

    ax.set_xlabel('Sample Size (N)', fontsize=13)
    ax.set_ylabel('Median Std of Bootstrap Estimates (pp)', fontsize=13)
    ax.set_title('Convergence by Reasoning Dimension\n(median across all model-dataset combos, CoT Zero-shot)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_subscore_convergence.png'), dpi=150)
    plt.close()
    print("  -> 5_subscore_convergence.png")


def plot_convergence_ratio_heatmap(all_results: list, output_dir: str):
    """Plot 6: Heatmap of N_converge(accuracy) / N_converge(reasoning).

    Higher ratios mean reasoning is relatively more sample-efficient.
    """
    # Build model x dataset matrix
    models = sorted(set(r['model'] for r in all_results))
    datasets = sorted(set(r['dataset'] for r in all_results))

    # Create lookup
    lookup = {}
    for r in all_results:
        lookup[(r['model'], r['dataset'])] = r['convergence_n']

    matrix = np.full((len(models), len(datasets)), np.nan)
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            cn = lookup.get((model, dataset))
            if cn is None:
                continue
            acc_n = cn['accuracy']
            reas_n = cn['overall']
            if acc_n is not None and reas_n is not None and reas_n > 0:
                matrix[i, j] = acc_n / reas_n
            elif acc_n is None and reas_n is not None:
                # Accuracy never converged within tested range
                matrix[i, j] = float('nan')

    fig, ax = plt.subplots(figsize=(10, 7))

    # Mask NaN for display
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='#f0f0f0')

    im = ax.imshow(masked, cmap=cmap, aspect='auto',
                   vmin=1, vmax=max(np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 5, 2))

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha='right', fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(datasets)):
            val = matrix[i, j]
            if not np.isnan(val):
                text = f'{val:.1f}x'
                ax.text(j, i, text, ha='center', va='center', fontsize=9,
                        color='white' if val > (np.nanmax(matrix) * 0.6) else 'black')
            else:
                cn = lookup.get((models[i], datasets[j]))
                if cn is not None:
                    ax.text(j, i, 'N/A', ha='center', va='center',
                            fontsize=8, color='gray')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Ratio: N_accuracy / N_reasoning', fontsize=11)

    ax.set_title('Convergence Speedup: Reasoning vs Accuracy\n(ratio > 1 means reasoning converges faster)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_convergence_ratio_heatmap.png'), dpi=150)
    plt.close()
    print("  -> 6_convergence_ratio_heatmap.png")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_results: list, agg: dict, output_path: str):
    """Generate a text report of the convergence analysis."""
    lines = []
    lines.append("=" * 90)
    lines.append("CONVERGENCE ANALYSIS: REASONING SCORE vs ACCURACY")
    lines.append("Prompt variation: CoT Zero-shot (temp=0.7, top_p=0.95)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)
    lines.append("")

    # ---- Section 1: Overview ----
    lines.append("1. OVERVIEW")
    lines.append("-" * 50)
    lines.append(f"Total model-dataset combinations: {len(all_results)}")
    models = sorted(set(r['model'] for r in all_results))
    datasets = sorted(set(r['dataset'] for r in all_results))
    lines.append(f"Models ({len(models)}): {', '.join(models)}")
    lines.append(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    sample_counts = [r['n_samples'] for r in all_results]
    lines.append(f"Samples per combo: {min(sample_counts)} - {max(sample_counts)} (median {int(np.median(sample_counts))})")
    n_bootstrap = None
    for r in all_results:
        bs = r['convergence']['bootstrap']
        if bs:
            first_n = next(iter(bs.values()))
            # Can't determine n_bootstrap from saved stats; use arg
            break
    lines.append("")

    # ---- Section 2: Aggregate convergence table ----
    lines.append("2. AGGREGATE CONVERGENCE (median across all combos)")
    lines.append("-" * 50)
    lines.append(f"{'N':>6} | {'Acc Std':>9} | {'Reas Std':>9} | {'Acc RelErr':>10} | {'Reas RelErr':>11} | {'Acc CI95':>9} | {'Reas CI95':>9} | {'Speedup':>8}")
    lines.append("-" * 90)

    acc_std_data = agg['accuracy']['std']
    reas_std_data = agg['reasoning']['overall']['std']
    acc_rel_data = agg['accuracy']['relative_error']
    reas_rel_data = agg['reasoning']['overall']['relative_error']
    acc_ci_data = agg['accuracy']['ci95_half']
    reas_ci_data = agg['reasoning']['overall']['ci95_half']

    ns = sorted(set(acc_std_data.keys()) & set(reas_std_data.keys()))
    for n in ns:
        a_std = acc_std_data[n]['median'] * 100
        r_std = reas_std_data[n]['median'] * 100
        a_rel = acc_rel_data[n]['median'] * 100
        r_rel = reas_rel_data[n]['median'] * 100
        a_ci = acc_ci_data[n]['median'] * 100
        r_ci = reas_ci_data[n]['median'] * 100
        speedup = a_std / r_std if r_std > 0 else float('inf')
        lines.append(f"{n:>6} | {a_std:>8.2f}% | {r_std:>8.2f}% | {a_rel:>9.2f}% | {r_rel:>10.2f}% | {a_ci:>8.2f}% | {r_ci:>8.2f}% | {speedup:>7.2f}x")
    lines.append("")
    lines.append("  Speedup = Accuracy Std / Reasoning Std (higher = reasoning converges faster)")
    lines.append("")

    # ---- Section 3: Sub-score convergence ----
    lines.append("3. REASONING SUB-SCORE CONVERGENCE (median std in pp)")
    lines.append("-" * 50)
    header = f"{'N':>6}"
    for dim in REASONING_DIMS:
        header += f" | {dim[:8]:>9}"
    lines.append(header)
    lines.append("-" * (6 + len(REASONING_DIMS) * 12))
    for n in ns:
        row = f"{n:>6}"
        for dim in REASONING_DIMS:
            val = agg['reasoning'][dim]['std'][n]['median'] * 100
            row += f" | {val:>8.2f}%"
        lines.append(row)
    lines.append("")

    # ---- Section 4: Convergence threshold analysis ----
    lines.append("4. SAMPLES TO CONVERGE (95% CI half-width < 2pp)")
    lines.append("-" * 50)
    lines.append(f"{'Model':<35} {'Dataset':<18} {'N(Acc)':>8} {'N(Reas)':>8} {'Speedup':>8}")
    lines.append("-" * 80)

    speedups = []
    for r in sorted(all_results, key=lambda x: (x['model'], x['dataset'])):
        cn = r['convergence_n']
        acc_n = cn['accuracy']
        reas_n = cn['overall']
        acc_str = str(acc_n) if acc_n is not None else '>max'
        reas_str = str(reas_n) if reas_n is not None else '>max'
        if acc_n is not None and reas_n is not None and reas_n > 0:
            sp = acc_n / reas_n
            sp_str = f'{sp:.1f}x'
            speedups.append(sp)
        else:
            sp_str = 'N/A'
        lines.append(f"{r['model']:<35} {r['dataset']:<18} {acc_str:>8} {reas_str:>8} {sp_str:>8}")

    lines.append("")
    if speedups:
        lines.append(f"Average speedup: {np.mean(speedups):.2f}x (reasoning converges {np.mean(speedups):.1f}x faster)")
        lines.append(f"Median speedup: {np.median(speedups):.2f}x")
        lines.append(f"Range: {min(speedups):.1f}x - {max(speedups):.1f}x")
    lines.append("")

    # ---- Section 5: Per-dataset summary ----
    lines.append("5. PER-DATASET SUMMARY")
    lines.append("-" * 50)
    by_dataset = defaultdict(list)
    for r in all_results:
        by_dataset[r['dataset']].append(r)

    for dataset in sorted(by_dataset.keys()):
        combos = by_dataset[dataset]
        acc_ns = [r['convergence_n']['accuracy'] for r in combos if r['convergence_n']['accuracy'] is not None]
        reas_ns = [r['convergence_n']['overall'] for r in combos if r['convergence_n']['overall'] is not None]
        lines.append(f"\n  {dataset.upper()} ({len(combos)} models, ~{combos[0]['n_samples']} samples):")
        if acc_ns:
            lines.append(f"    Accuracy converges at: median N={int(np.median(acc_ns))}, mean N={np.mean(acc_ns):.0f}")
        else:
            lines.append(f"    Accuracy: does not converge within tested range for any model")
        if reas_ns:
            lines.append(f"    Reasoning converges at: median N={int(np.median(reas_ns))}, mean N={np.mean(reas_ns):.0f}")
        else:
            lines.append(f"    Reasoning: does not converge within tested range for any model")

    lines.append("")

    # ---- Section 6: Key findings ----
    lines.append("6. KEY FINDINGS")
    lines.append("-" * 50)

    # Count how often reasoning converges strictly before accuracy
    faster_count = sum(1 for r in all_results
                       if r['convergence_n']['overall'] is not None
                       and r['convergence_n']['accuracy'] is not None
                       and r['convergence_n']['overall'] < r['convergence_n']['accuracy'])
    same_count = sum(1 for r in all_results
                     if r['convergence_n']['overall'] is not None
                     and r['convergence_n']['accuracy'] is not None
                     and r['convergence_n']['overall'] == r['convergence_n']['accuracy'])
    total_both = sum(1 for r in all_results
                     if r['convergence_n']['overall'] is not None
                     and r['convergence_n']['accuracy'] is not None)

    if total_both > 0:
        lines.append(f"Reasoning converges strictly faster in {faster_count}/{total_both} combos ({faster_count/total_both*100:.0f}%)")
        lines.append(f"Reasoning and accuracy converge at same N in {same_count}/{total_both} combos")
    else:
        lines.append("Not enough data to compare convergence points.")

    # Find the combo with largest speedup
    best = None
    best_sp = 0
    for r in all_results:
        cn = r['convergence_n']
        if cn['accuracy'] is not None and cn['overall'] is not None and cn['overall'] > 0:
            sp = cn['accuracy'] / cn['overall']
            if sp > best_sp:
                best_sp = sp
                best = r
    if best:
        lines.append(f"\nLargest speedup: {best['model']} on {best['dataset']} "
                      f"({best_sp:.1f}x -- accuracy needs N={best['convergence_n']['accuracy']}, "
                      f"reasoning needs N={best['convergence_n']['overall']})")

    lines.append("")
    lines.append("=" * 90)
    lines.append("END OF REPORT")
    lines.append("=" * 90)

    report_text = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convergence Analysis: Reasoning Score vs Accuracy (CoT Zero-shot)')
    parser.add_argument('--cot-dir', type=str,
                        default='evaluation/exports/cot_analysis',
                        help='Directory containing COT analysis JSON files')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/convergence',
                        help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Go up one level from experiments/
    cot_dir = os.path.join(base_dir, args.cot_dir)
    output_dir = os.path.join(base_dir, args.output_dir)

    print(f"Loading COT analysis files from: {cot_dir}")
    raw_results = load_cot_analysis_files(cot_dir)
    print(f"Loaded {len(raw_results)} total model-dataset combinations")

    # Filter to CoT Zero-shot only (prompt_type == "cot")
    cot_results = [r for r in raw_results if r['prompt_type'] == 'cot']
    print(f"Filtered to {len(cot_results)} CoT Zero-shot combinations")

    if not cot_results:
        print("No CoT Zero-shot data found. Check the directory.")
        return

    # Sample sizes to test
    n_values = [10, 25, 50, 75, 100, 150, 200, 300, 500]

    # Run convergence analysis for each combo
    print(f"\nRunning bootstrap convergence (B={args.n_bootstrap}) ...")
    all_results = []
    for i, r in enumerate(cot_results):
        print(f"  [{i+1}/{len(cot_results)}] {r['model']} / {r['dataset']} "
              f"(N={r['n_samples']})")

        conv = run_convergence_bootstrap(r['samples'], n_values, args.n_bootstrap)
        conv_n = find_convergence_n(conv['bootstrap'], threshold=0.02)

        all_results.append({
            'model': r['model'],
            'dataset': r['dataset'],
            'n_samples': r['n_samples'],
            'convergence': conv,
            'convergence_n': conv_n,
        })

    # Aggregate across all combos
    print("\nAggregating results ...")
    agg = aggregate_across_combos(all_results)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate report
    print("\n")
    report_path = os.path.join(output_dir, 'convergence_report.txt')
    generate_report(all_results, agg, report_path)

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("\nGenerating plots ...")
        plot_aggregate_convergence(agg, output_dir)
        plot_relative_error(agg, output_dir)
        plot_samples_to_converge(all_results, output_dir)
        plot_per_dataset_convergence(all_results, output_dir)
        plot_subscore_convergence(agg, output_dir)
        plot_convergence_ratio_heatmap(all_results, output_dir)
        print(f"\nAll plots saved to {output_dir}")

    # Save JSON results
    json_output = {
        'metadata': {
            'prompt_variation': 'cot_zero_shot',
            'n_bootstrap': args.n_bootstrap,
            'n_values': n_values,
            'seed': args.seed,
            'n_combos': len(all_results),
            'generated': datetime.now().isoformat(),
        },
        'aggregate': _serialize_agg(agg),
        'per_combo': [_serialize_combo(r) for r in all_results],
    }

    json_path = os.path.join(output_dir, 'convergence_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Results saved to {json_path}")


def _serialize_agg(agg: dict) -> dict:
    """Convert aggregate dict for JSON serialization (int keys -> str keys)."""
    out = {'accuracy': {}, 'reasoning': {}}
    for metric in ['std', 'relative_error', 'ci95_half']:
        out['accuracy'][metric] = {str(k): v for k, v in agg['accuracy'][metric].items()}
    for dim in REASONING_DIMS:
        out['reasoning'][dim] = {}
        for metric in ['std', 'relative_error', 'ci95_half']:
            out['reasoning'][dim][metric] = {str(k): v for k, v in agg['reasoning'][dim][metric].items()}
    return out


def _serialize_combo(r: dict) -> dict:
    """Convert a single combo result for JSON serialization."""
    conv = r['convergence']
    bootstrap_out = {}
    for n, data in conv['bootstrap'].items():
        bootstrap_out[str(n)] = {
            'accuracy': data['accuracy'],
            'reasoning': {dim: data['reasoning'][dim] for dim in REASONING_DIMS},
        }
    return {
        'model': r['model'],
        'dataset': r['dataset'],
        'n_samples': r['n_samples'],
        'true_accuracy': conv['true_accuracy'],
        'true_reasoning': conv['true_reasoning'],
        'convergence_n': r['convergence_n'],
        'bootstrap': bootstrap_out,
    }


if __name__ == '__main__':
    main()
