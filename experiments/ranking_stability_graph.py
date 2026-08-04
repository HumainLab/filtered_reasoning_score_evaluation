#!/usr/bin/env python3
"""
Generate "Ranking stability across prompt and temperature variations" bar chart.

Shows average Spearman rank correlation (across datasets) for Reasoning score vs Accuracy,
grouped by prompting strategy: Direct, Few-shot, CoT temp=0, CoT temp=0.7.

Usage:
    python experiments/ranking_stability_graph.py [--cot-dir DIR] [--output PATH]
"""

import json
import glob
import os
import argparse
import numpy as np
from scipy import stats
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_job_db(job_db_path: str) -> dict:
    """Load job database to get temperature per job_id."""
    if not os.path.exists(job_db_path):
        return {}
    try:
        with open(job_db_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def map_to_condition(prompt_type: str, job_id: str, job_db: dict) -> Optional[str]:
    """Map (prompt_type, job) to chart condition: Direct, Few-shot, CoT temp=0, CoT temp=0.7."""
    # Direct: zero-shot, no few-shot examples
    if prompt_type == 'direct':
        return 'Direct'

    # Few-shot: any *_fewshot prompt
    if 'fewshot' in (prompt_type or '').lower():
        return 'Few-shot'

    # CoT zero-shot: prompt_type is 'cot', distinguish by temperature
    if prompt_type == 'cot':
        job = job_db.get(job_id, {})
        req = job.get('request', {})
        temp = req.get('temperature', 0.0)
        if abs(temp - 0.7) < 0.01:
            return 'CoT temp=0.7'
        return 'CoT temp=0'  # default to temp=0 for cot

    return None  # unknown, skip


def load_cot_analysis_files(base_path: str, job_db: dict) -> list:
    """Load COT analysis files with condition mapping."""
    json_files = glob.glob(os.path.join(base_path, '**/*.json'), recursive=True)
    results = []

    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)

            model = data.get('model')
            dataset = data.get('dataset')
            prompt_type = data.get('prompt_type', 'unknown')
            job_id = data.get('job_id', '')

            if not model:
                continue

            condition = map_to_condition(prompt_type, job_id, job_db)
            if condition is None:
                continue

            samples = data.get('per_sample', [])
            if len(samples) < 50:
                continue

            sample_data = []
            for s in samples:
                overall = s.get('overall', 0)
                evidence = s.get('evidence', {})
                correct = evidence.get('final_correct')
                if correct is not None and overall > 0:
                    sample_data.append({'overall': overall, 'correct': 1 if correct else 0})

            if len(sample_data) < 50:
                continue

            results.append({
                'model': model,
                'dataset': dataset,
                'condition': condition,
                'samples': sample_data,
                'true_accuracy': np.mean([x['correct'] for x in sample_data]),
                'true_reasoning': np.mean([x['overall'] for x in sample_data]),
            })
        except Exception as e:
            pass

    return results


def run_ranking_experiment(results_by_condition_dataset: dict, n_bootstrap: int = 100) -> dict:
    """Compute Spearman rank correlation per (condition, dataset), then average across datasets."""
    # results_by_condition_dataset: condition -> dataset -> list of model dicts
    condition_metrics = defaultdict(lambda: {'reasoning_corrs': [], 'accuracy_corrs': []})

    for condition, by_dataset in results_by_condition_dataset.items():
        for dataset, models in by_dataset.items():
            if len(models) < 3:
                continue

            true_accs = [m['true_accuracy'] for m in models]
            true_reas = [m['true_reasoning'] for m in models]
            min_samples = min(len(m['samples']) for m in models)
            n_sample = min(25, min_samples)  # use N=25 for stability, or less if not enough data

            if n_sample < 10:
                continue

            reas_corrs = []
            acc_corrs = []

            for _ in range(n_bootstrap):
                sampled_reasoning = []
                sampled_accuracy = []
                for m in models:
                    idx = np.random.choice(len(m['samples']), size=n_sample, replace=False)
                    sampled_reasoning.append(np.mean([m['samples'][i]['overall'] for i in idx]))
                    sampled_accuracy.append(np.mean([m['samples'][i]['correct'] for i in idx]))

                r_reas, _ = stats.spearmanr(sampled_reasoning, true_accs)
                r_acc, _ = stats.spearmanr(sampled_accuracy, true_accs)
                if not np.isnan(r_reas):
                    reas_corrs.append(r_reas)
                if not np.isnan(r_acc):
                    acc_corrs.append(r_acc)

            if reas_corrs:
                condition_metrics[condition]['reasoning_corrs'].append(np.mean(reas_corrs))
            if acc_corrs:
                condition_metrics[condition]['accuracy_corrs'].append(np.mean(acc_corrs))

    return dict(condition_metrics)


def plot_ranking_stability(condition_metrics: dict, output_path: str):
    """Generate bar chart: Ranking stability across prompt and temperature variations."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    order = ['Direct', 'Few-shot', 'CoT temp=0', 'CoT temp=0.7']
    conditions = [c for c in order if c in condition_metrics]
    if not conditions:
        print("No data for any condition, cannot plot")
        return

    reasoning_means = []
    accuracy_means = []
    for c in conditions:
        m = condition_metrics[c]
        reasoning_means.append(np.mean(m['reasoning_corrs']) if m['reasoning_corrs'] else 0)
        accuracy_means.append(np.mean(m['accuracy_corrs']) if m['accuracy_corrs'] else 0)

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, reasoning_means, width, label='Reasoning score', color='#1f77b4', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, accuracy_means, width, label='Accuracy', color='#ff7f0e', edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Avg Spearman rank correlation (across datasets)', fontsize=12)
    ax.set_title('Ranking stability across prompt and temperature variations', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Ranking stability bar chart')
    parser.add_argument('--cot-dir', type=str,
                        default='evaluation/exports/cot_analysis',
                        help='COT analysis directory')
    parser.add_argument('--job-db', type=str,
                        default='backend/job_db.json',
                        help='Job database for temperature lookup')
    parser.add_argument('--output', type=str,
                        default='experiments/results/ranking_stability.png',
                        help='Output image path')
    parser.add_argument('--n-bootstrap', type=int, default=100,
                        help='Bootstrap iterations per dataset')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    cot_dir = base_dir / args.cot_dir
    job_db_path = base_dir / args.job_db
    output_path = base_dir / args.output

    print("Loading job_db...")
    job_db = load_job_db(str(job_db_path))
    print(f"  Loaded {len(job_db)} jobs")

    print("Loading COT analysis files...")
    all_results = load_cot_analysis_files(str(cot_dir), job_db)
    print(f"  Loaded {len(all_results)} model-dataset-condition entries")

    # Group by (condition, dataset)
    by_cond_dataset = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        by_cond_dataset[r['condition']][r['dataset']].append(r)

    print("\nConditions and datasets:")
    for c in ['Direct', 'Few-shot', 'CoT temp=0', 'CoT temp=0.7']:
        if c in by_cond_dataset:
            datasets = list(by_cond_dataset[c].keys())
            n_models = sum(len(by_cond_dataset[c][d]) for d in datasets)
            print(f"  {c}: {len(datasets)} datasets, {n_models} model-dataset combos")

    print("\nComputing Spearman rank correlations...")
    condition_metrics = run_ranking_experiment(by_cond_dataset, n_bootstrap=args.n_bootstrap)

    print("\nResults (avg Spearman across datasets):")
    for c in ['Direct', 'Few-shot', 'CoT temp=0', 'CoT temp=0.7']:
        if c in condition_metrics:
            m = condition_metrics[c]
            r_reas = np.mean(m['reasoning_corrs']) if m['reasoning_corrs'] else 0
            r_acc = np.mean(m['accuracy_corrs']) if m['accuracy_corrs'] else 0
            print(f"  {c}: Reasoning={r_reas:.3f}, Accuracy={r_acc:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_ranking_stability(condition_metrics, str(output_path))


if __name__ == '__main__':
    main()
