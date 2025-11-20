#!/usr/bin/env python3
"""
Evaluate Baseline Predictions

Calculate TM-scores for baseline predictions against validation labels.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import calculate_tm_score_from_pdbs

PROJECT_DIR = Path(__file__).parent.parent
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
PREDICTIONS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'
RESULTS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'

print("=" * 80)
print("BASELINE TM-SCORE EVALUATION")
print("=" * 80)
print()

# Step 1: Load validation labels
print("[1/4] Loading validation labels...")
print("-" * 80)

valid_labels = pd.read_csv(STANFORD_DIR / 'validation_labels.csv')
test_seqs = pd.read_csv(STANFORD_DIR / 'test_sequences.csv')

print(f"  ✓ Loaded {len(valid_labels)} label rows")
print(f"  ✓ Test sequences: {len(test_seqs)}")
print()

# Step 2: Process labels into coordinate dictionaries
print("[2/4] Processing validation coordinates...")
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

    if len(coords) > 0:
        valid_coords[target_id] = np.array(coords)

print(f"  ✓ Processed {len(valid_coords)} validation structures")
print()

# Step 3: Save validation structures as PDB files for US-align
print("[3/4] Creating validation PDB files...")
print("-" * 80)

valid_pdb_dir = RESULTS_DIR / 'validation_pdbs'
valid_pdb_dir.mkdir(exist_ok=True)

for target_id, coords in valid_coords.items():
    # Get sequence
    seq_row = test_seqs[test_seqs['target_id'] == target_id].iloc[0]
    sequence = seq_row['sequence']

    # Write PDB
    pdb_path = valid_pdb_dir / f"{target_id}_true.pdb"
    with open(pdb_path, 'w') as f:
        f.write(f"HEADER    RNA VALIDATION    {target_id}\n")
        for i, (coord, base) in enumerate(zip(coords, sequence), start=1):
            if np.isnan(coord).any():
                continue
            line = (
                f"ATOM  {i:5d}  C1'  {base:>3} A{i:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           C\n"
            )
            f.write(line)
        f.write("END\n")

print(f"  ✓ Created {len(valid_coords)} validation PDB files")
print()

# Step 4: Calculate TM-scores
print("[4/4] Calculating TM-scores...")
print("-" * 80)
print()

results = []

for target_id in tqdm(valid_coords.keys(), desc="Evaluating"):
    true_pdb = valid_pdb_dir / f"{target_id}_true.pdb"

    # Find all predictions for this target
    pred_files = sorted(PREDICTIONS_DIR.glob(f"{target_id}_pred_*.pdb"))

    if len(pred_files) == 0:
        results.append({
            'target_id': target_id,
            'num_predictions': 0,
            'best_tm_score': None,
            'mean_tm_score': None,
            'status': 'no_predictions'
        })
        continue

    tm_scores = []

    for pred_file in pred_files:
        result = calculate_tm_score_from_pdbs(
            str(pred_file),
            str(true_pdb),
            usalign_path='USalign/USalign'
        )

        if result['tm_score'] is not None:
            tm_scores.append(result['tm_score'])

    if len(tm_scores) > 0:
        results.append({
            'target_id': target_id,
            'num_predictions': len(pred_files),
            'best_tm_score': max(tm_scores),
            'mean_tm_score': np.mean(tm_scores),
            'all_scores': tm_scores,
            'status': 'success'
        })
    else:
        results.append({
            'target_id': target_id,
            'num_predictions': len(pred_files),
            'best_tm_score': None,
            'mean_tm_score': None,
            'status': 'tm_score_failed'
        })

print()
print("=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)

# Save results
eval_csv = RESULTS_DIR / 'tm_score_evaluation.csv'
results_df.to_csv(eval_csv, index=False)
print(f"\n✓ Results saved to: {eval_csv}")

# Statistics
successful = results_df[results_df['status'] == 'success']

print(f"\nTotal targets: {len(results_df)}")
print(f"  ✓ Successful: {len(successful)}")
print(f"  ✗ Failed: {len(results_df) - len(successful)}")

if len(successful) > 0:
    best_scores = successful['best_tm_score'].values
    mean_scores = successful['mean_tm_score'].values

    print(f"\nTM-Score Statistics (Best-of-N):")
    print(f"  Mean:   {np.mean(best_scores):.3f}")
    print(f"  Median: {np.median(best_scores):.3f}")
    print(f"  Min:    {np.min(best_scores):.3f}")
    print(f"  Max:    {np.max(best_scores):.3f}")
    print(f"  Std:    {np.std(best_scores):.3f}")

    print(f"\nTM-Score Distribution:")
    bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ['Poor (<0.3)', 'Fair (0.3-0.5)', 'Good (0.5-0.7)', 'Very Good (0.7-0.9)', 'Excellent (>0.9)']

    for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], labels)):
        count = sum((best_scores >= low) & (best_scores < high))
        if i == len(labels) - 1:  # Last bin includes upper bound
            count = sum((best_scores >= low) & (best_scores <= high))
        pct = count / len(best_scores) * 100
        print(f"  {label:20s}: {count:2d} ({pct:5.1f}%)")

    print(f"\nTop 5 Predictions:")
    top_5 = successful.nlargest(5, 'best_tm_score')[['target_id', 'best_tm_score', 'num_predictions']]
    for idx, row in top_5.iterrows():
        print(f"  {row['target_id']:10s}: {row['best_tm_score']:.3f} ({int(row['num_predictions'])} predictions)")

    print(f"\nBottom 5 Predictions:")
    bottom_5 = successful.nsmallest(5, 'best_tm_score')[['target_id', 'best_tm_score', 'num_predictions']]
    for idx, row in bottom_5.iterrows():
        print(f"  {row['target_id']:10s}: {row['best_tm_score']:.3f} ({int(row['num_predictions'])} predictions)")

print()
print("=" * 80)
print()
