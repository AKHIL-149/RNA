#!/usr/bin/env python3
"""
Evaluate Baseline Predictions using RMSD and Approximate TM-score

Alternative evaluation method that doesn't rely on US-align.
Uses direct coordinate comparison with BioPython.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import evaluate_structure_similarity

PROJECT_DIR = Path(__file__).parent.parent
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
PREDICTIONS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'
RESULTS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'

print("=" * 80)
print("BASELINE EVALUATION - RMSD & APPROXIMATE TM-SCORE")
print("=" * 80)
print()

# Step 1: Load validation labels
print("[1/3] Loading validation labels...")
print("-" * 80)

valid_labels = pd.read_csv(STANFORD_DIR / 'validation_labels.csv')
test_seqs = pd.read_csv(STANFORD_DIR / 'test_sequences.csv')

print(f"  ✓ Loaded {len(valid_labels)} label rows")
print(f"  ✓ Test sequences: {len(test_seqs)}")
print()

# Step 2: Process labels into coordinate dictionaries
print("[2/3] Processing validation coordinates...")
print("-" * 80)

# Group by target
valid_coords = {}
for target_id in test_seqs['target_id']:
    target_labels = valid_labels[valid_labels['ID'].str.startswith(target_id)]

    if len(target_labels) == 0:
        continue

    # Extract coordinates (x_1, y_1, z_1 are the C1' atom positions)
    coords = []
    for _, row in target_labels.sort_values('resid').iterrows():
        if pd.notna(row['x_1']) and pd.notna(row['y_1']) and pd.notna(row['z_1']):
            coords.append([row['x_1'], row['y_1'], row['z_1']])
        else:
            coords.append([np.nan, np.nan, np.nan])

    if len(coords) > 0:
        valid_coords[target_id] = np.array(coords)

print(f"  ✓ Processed {len(valid_coords)} validation structures")
print()

# Step 3: Evaluate predictions
print("[3/3] Evaluating predictions...")
print("-" * 80)
print()

results = []

for target_id in tqdm(sorted(valid_coords.keys()), desc="Evaluating"):
    true_coords = valid_coords[target_id]

    # Find all predictions for this target
    pred_files = sorted(PREDICTIONS_DIR.glob(f"{target_id}_pred_*.pdb"))

    if len(pred_files) == 0:
        results.append({
            'target_id': target_id,
            'num_predictions': 0,
            'best_tm_score': None,
            'best_rmsd': None,
            'mean_tm_score': None,
            'mean_rmsd': None,
            'status': 'no_predictions'
        })
        continue

    # Evaluate each prediction
    pred_metrics = []

    for pred_file in pred_files:
        # Read predicted coordinates from PDB
        pred_coords = []
        with open(pred_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    pred_coords.append([x, y, z])

        if len(pred_coords) == 0:
            continue

        pred_coords = np.array(pred_coords)

        # Evaluate similarity
        metrics = evaluate_structure_similarity(pred_coords, true_coords)
        metrics['pred_file'] = pred_file.name
        pred_metrics.append(metrics)

    if len(pred_metrics) > 0:
        # Extract scores
        tm_scores = [m['tm_score'] for m in pred_metrics if m['tm_score'] is not None]
        rmsd_scores = [m['aligned_rmsd'] for m in pred_metrics if m['aligned_rmsd'] is not None]

        results.append({
            'target_id': target_id,
            'num_predictions': len(pred_files),
            'best_tm_score': max(tm_scores) if tm_scores else None,
            'best_rmsd': min(rmsd_scores) if rmsd_scores else None,
            'mean_tm_score': np.mean(tm_scores) if tm_scores else None,
            'mean_rmsd': np.mean(rmsd_scores) if rmsd_scores else None,
            'all_tm_scores': tm_scores,
            'all_rmsd_scores': rmsd_scores,
            'coverage': pred_metrics[0]['coverage'],
            'num_aligned': pred_metrics[0]['num_aligned'],
            'status': 'success'
        })
    else:
        results.append({
            'target_id': target_id,
            'num_predictions': len(pred_files),
            'best_tm_score': None,
            'best_rmsd': None,
            'mean_tm_score': None,
            'mean_rmsd': None,
            'status': 'evaluation_failed'
        })

print()
print("=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)

# Save results
eval_csv = RESULTS_DIR / 'rmsd_evaluation.csv'
results_df.to_csv(eval_csv, index=False)
print(f"\n✓ Results saved to: {eval_csv}")

# Statistics
successful = results_df[results_df['status'] == 'success']

print(f"\nTotal targets: {len(results_df)}")
print(f"  ✓ Successful: {len(successful)}")
print(f"  ✗ Failed: {len(results_df) - len(successful)}")

if len(successful) > 0:
    best_tm_scores = successful['best_tm_score'].values
    best_rmsd_scores = successful['best_rmsd'].values
    mean_tm_scores = successful['mean_tm_score'].values
    mean_rmsd_scores = successful['mean_rmsd'].values

    print(f"\n{'='*80}")
    print("TM-SCORE STATISTICS (Approximate, Best-of-N)")
    print(f"{'='*80}")
    print(f"  Mean:   {np.mean(best_tm_scores):.3f}")
    print(f"  Median: {np.median(best_tm_scores):.3f}")
    print(f"  Min:    {np.min(best_tm_scores):.3f}")
    print(f"  Max:    {np.max(best_tm_scores):.3f}")
    print(f"  Std:    {np.std(best_tm_scores):.3f}")

    print(f"\n{'='*80}")
    print("RMSD STATISTICS (Angstroms, Best-of-N)")
    print(f"{'='*80}")
    print(f"  Mean:   {np.mean(best_rmsd_scores):.3f} Å")
    print(f"  Median: {np.median(best_rmsd_scores):.3f} Å")
    print(f"  Min:    {np.min(best_rmsd_scores):.3f} Å")
    print(f"  Max:    {np.max(best_rmsd_scores):.3f} Å")
    print(f"  Std:    {np.std(best_rmsd_scores):.3f} Å")

    print(f"\n{'='*80}")
    print("TM-SCORE DISTRIBUTION")
    print(f"{'='*80}")
    bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ['Poor (<0.3)', 'Fair (0.3-0.5)', 'Good (0.5-0.7)',
              'Very Good (0.7-0.9)', 'Excellent (>0.9)']

    for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], labels)):
        count = sum((best_tm_scores >= low) & (best_tm_scores < high))
        if i == len(labels) - 1:  # Last bin includes upper bound
            count = sum((best_tm_scores >= low) & (best_tm_scores <= high))
        pct = count / len(best_tm_scores) * 100
        print(f"  {label:20s}: {count:2d} ({pct:5.1f}%)")

    print(f"\n{'='*80}")
    print("RMSD DISTRIBUTION")
    print(f"{'='*80}")
    rmsd_bins = [0.0, 1.0, 2.0, 5.0, 10.0, float('inf')]
    rmsd_labels = ['Excellent (<1Å)', 'Very Good (1-2Å)', 'Good (2-5Å)',
                   'Fair (5-10Å)', 'Poor (>10Å)']

    for i, (low, high, label) in enumerate(zip(rmsd_bins[:-1], rmsd_bins[1:], rmsd_labels)):
        count = sum((best_rmsd_scores >= low) & (best_rmsd_scores < high))
        pct = count / len(best_rmsd_scores) * 100
        print(f"  {label:20s}: {count:2d} ({pct:5.1f}%)")

    print(f"\n{'='*80}")
    print("TOP 5 PREDICTIONS (by TM-score)")
    print(f"{'='*80}")
    top_5 = successful.nlargest(5, 'best_tm_score')[
        ['target_id', 'best_tm_score', 'best_rmsd', 'num_predictions']
    ]
    for idx, row in top_5.iterrows():
        print(f"  {row['target_id']:10s}: TM={row['best_tm_score']:.3f}, "
              f"RMSD={row['best_rmsd']:.3f}Å ({int(row['num_predictions'])} preds)")

    print(f"\n{'='*80}")
    print("BOTTOM 5 PREDICTIONS (by TM-score)")
    print(f"{'='*80}")
    bottom_5 = successful.nsmallest(5, 'best_tm_score')[
        ['target_id', 'best_tm_score', 'best_rmsd', 'num_predictions']
    ]
    for idx, row in bottom_5.iterrows():
        print(f"  {row['target_id']:10s}: TM={row['best_tm_score']:.3f}, "
              f"RMSD={row['best_rmsd']:.3f}Å ({int(row['num_predictions'])} preds)")

    # Additional insights
    print(f"\n{'='*80}")
    print("COVERAGE STATISTICS")
    print(f"{'='*80}")
    coverage_values = successful['coverage'].values
    print(f"  Mean coverage:   {np.mean(coverage_values):.1%}")
    print(f"  Median coverage: {np.median(coverage_values):.1%}")
    print(f"  Min coverage:    {np.min(coverage_values):.1%}")
    print(f"  Max coverage:    {np.max(coverage_values):.1%}")

print()
print("=" * 80)
print()
