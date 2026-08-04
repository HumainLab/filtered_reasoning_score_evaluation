#!/usr/bin/env python3
"""
Experiment: Can reasoning scores predict model ranking with fewer samples?

Question: If we sample N examples and rank models by reasoning score,
how well does that ranking match the true accuracy ranking?
"""

import argparse
import json
import glob
import os
from pathlib import Path
import numpy as np
from scipy import stats
from collections import defaultdict

def load_cot_analysis_files(base_path: str) -> list:
    """Load all COT analysis JSON files."""
    json_files = glob.glob(os.path.join(base_path, '**/*.json'), recursive=True)
    
    results = []
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            model = data.get('model')
            dataset = data.get('dataset')
            if not model:
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
                    sample_data.append({
                        'overall': overall,
                        'correct': 1 if correct else 0
                    })
            
            if len(sample_data) >= 50:
                results.append({
                    'model': model,
                    'dataset': dataset,
                    'samples': sample_data,
                    'true_accuracy': np.mean([s['correct'] for s in sample_data]),
                    'true_reasoning': np.mean([s['overall'] for s in sample_data])
                })
        except Exception as e:
            pass
    
    return results


def run_ranking_experiment(results_by_dataset: dict, n_values: list, n_bootstrap: int = 100):
    """For each dataset, test if sampled reasoning predicts true accuracy ranking."""
    
    all_findings = {}
    
    for dataset, models in results_by_dataset.items():
        if len(models) < 3:
            continue
        
        # True rankings (by accuracy)
        true_acc_ranking = np.argsort([-m['true_accuracy'] for m in models])
        true_reas_ranking = np.argsort([-m['true_reasoning'] for m in models])
        
        # How well does full reasoning predict full accuracy ranking?
        full_reas_vs_acc, _ = stats.spearmanr(
            [models[i]['true_reasoning'] for i in range(len(models))],
            [models[i]['true_accuracy'] for i in range(len(models))]
        )
        
        findings = {
            'n_models': len(models),
            'full_reasoning_vs_accuracy_rank_corr': full_reas_vs_acc,
            'by_sample_size': {}
        }
        
        min_samples = min(len(m['samples']) for m in models)
        
        for N in n_values:
            if N > min_samples:
                continue
            
            # Bootstrap: sample N from each model, rank by reasoning, compare to true accuracy rank
            reas_rank_matches = []
            acc_rank_matches = []
            
            for _ in range(n_bootstrap):
                sampled_reasoning = []
                sampled_accuracy = []
                
                for m in models:
                    indices = np.random.choice(len(m['samples']), size=N, replace=False)
                    sampled_reas = np.mean([m['samples'][i]['overall'] for i in indices])
                    sampled_acc = np.mean([m['samples'][i]['correct'] for i in indices])
                    sampled_reasoning.append(sampled_reas)
                    sampled_accuracy.append(sampled_acc)
                
                # Rank correlation: sampled reasoning vs true accuracy
                reas_corr, _ = stats.spearmanr(sampled_reasoning, [m['true_accuracy'] for m in models])
                # Rank correlation: sampled accuracy vs true accuracy  
                acc_corr, _ = stats.spearmanr(sampled_accuracy, [m['true_accuracy'] for m in models])
                
                reas_rank_matches.append(reas_corr)
                acc_rank_matches.append(acc_corr)
            
            findings['by_sample_size'][N] = {
                'reasoning_rank_corr_mean': np.mean(reas_rank_matches),
                'reasoning_rank_corr_std': np.std(reas_rank_matches),
                'accuracy_rank_corr_mean': np.mean(acc_rank_matches),
                'accuracy_rank_corr_std': np.std(acc_rank_matches),
            }
        
        all_findings[dataset] = findings
    
    return all_findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cot-dir",
        default=str(Path(__file__).resolve().parent.parent / "evaluation" / "exports" / "cot_analysis"),
        help="Directory of cot_analysis_*.json exports (default: evaluation/exports/cot_analysis)",
    )
    base_path = parser.parse_args().cot_dir

    print("Loading data...")
    all_results = load_cot_analysis_files(base_path)
    print(f"Loaded {len(all_results)} model-dataset combinations")
    
    # Group by dataset
    by_dataset = defaultdict(list)
    for r in all_results:
        by_dataset[r['dataset']].append(r)
    
    print(f"\nDatasets with 3+ models: {[d for d, m in by_dataset.items() if len(m) >= 3]}")
    
    # Run experiment
    n_values = [10, 25, 50, 100, 200, 500]
    findings = run_ranking_experiment(by_dataset, n_values, n_bootstrap=200)
    
    # Print results
    print("\n" + "="*80)
    print("CAN REASONING SCORES PREDICT MODEL RANKING WITH FEWER SAMPLES?")
    print("="*80)
    
    print("\nSpearman rank correlation with TRUE ACCURACY RANKING")
    print("(Higher = better prediction of which models are actually more accurate)")
    print()
    
    for dataset in sorted(findings.keys()):
        f = findings[dataset]
        print(f"\n{dataset.upper()} ({f['n_models']} models)")
        print(f"  Full reasoning vs full accuracy rank correlation: {f['full_reasoning_vs_accuracy_rank_corr']:.3f}")
        print()
        print(f"  {'N':>6} | {'Reasoning→Rank':>18} | {'Accuracy→Rank':>18} | {'Winner':>10}")
        print(f"  {'-'*60}")
        
        for N in sorted(f['by_sample_size'].keys()):
            s = f['by_sample_size'][N]
            reas = f"{s['reasoning_rank_corr_mean']:.3f} ± {s['reasoning_rank_corr_std']:.3f}"
            acc = f"{s['accuracy_rank_corr_mean']:.3f} ± {s['accuracy_rank_corr_std']:.3f}"
            
            # Which is better at predicting true ranking?
            if s['reasoning_rank_corr_mean'] > s['accuracy_rank_corr_mean'] + 0.05:
                winner = "REASONING"
            elif s['accuracy_rank_corr_mean'] > s['reasoning_rank_corr_mean'] + 0.05:
                winner = "ACCURACY"
            else:
                winner = "TIE"
            
            print(f"  {N:>6} | {reas:>18} | {acc:>18} | {winner:>10}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: When is reasoning better than accuracy for ranking?")
    print("="*80)
    
    reasoning_wins = []
    accuracy_wins = []
    
    for dataset, f in findings.items():
        for N, s in f['by_sample_size'].items():
            if s['reasoning_rank_corr_mean'] > s['accuracy_rank_corr_mean'] + 0.05:
                reasoning_wins.append((dataset, N, s['reasoning_rank_corr_mean'] - s['accuracy_rank_corr_mean']))
            elif s['accuracy_rank_corr_mean'] > s['reasoning_rank_corr_mean'] + 0.05:
                accuracy_wins.append((dataset, N, s['accuracy_rank_corr_mean'] - s['reasoning_rank_corr_mean']))
    
    print(f"\nReasoning wins: {len(reasoning_wins)} cases")
    print(f"Accuracy wins: {len(accuracy_wins)} cases")
    
    if reasoning_wins:
        print("\nBest cases for reasoning:")
        for d, n, diff in sorted(reasoning_wins, key=lambda x: -x[2])[:5]:
            print(f"  {d} at N={n}: +{diff:.3f} advantage")
    
    # Key insight
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    
    # Check at small N
    small_n_comparison = []
    for dataset, f in findings.items():
        if 25 in f['by_sample_size']:
            s = f['by_sample_size'][25]
            small_n_comparison.append({
                'dataset': dataset,
                'reasoning': s['reasoning_rank_corr_mean'],
                'accuracy': s['accuracy_rank_corr_mean'],
                'diff': s['reasoning_rank_corr_mean'] - s['accuracy_rank_corr_mean']
            })
    
    if small_n_comparison:
        avg_reas = np.mean([x['reasoning'] for x in small_n_comparison])
        avg_acc = np.mean([x['accuracy'] for x in small_n_comparison])
        print(f"\nAt N=25 samples:")
        print(f"  Average rank correlation using REASONING: {avg_reas:.3f}")
        print(f"  Average rank correlation using ACCURACY:  {avg_acc:.3f}")
        
        if avg_reas > avg_acc:
            print(f"\n  → Reasoning is {(avg_reas-avg_acc):.3f} better at predicting true ranking with few samples!")
        else:
            print(f"\n  → Accuracy is {(avg_acc-avg_reas):.3f} better at predicting true ranking")


if __name__ == '__main__':
    main()
