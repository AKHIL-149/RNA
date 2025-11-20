#!/usr/bin/env python3
"""
Diagnose Failing Predictions

Investigate why predictions with high template identity are failing.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline
from src.evaluation import evaluate_structure_similarity

PROJECT_DIR = Path(__file__).parent.parent
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
RESULTS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'

print("=" * 80)
print("DIAGNOSING FAILING PREDICTIONS")
print("=" * 80)
print()

# Load data
print("[1/4] Loading data...")
with open(PROJECT_DIR / 'data' / 'train_coords_dict.pkl', 'rb') as f:
    train_coords = pickle.load(f)
with open(PROJECT_DIR / 'data' / 'train_sequences_dict.pkl', 'rb') as f:
    train_sequences = pickle.load(f)

valid_labels = pd.read_csv(STANFORD_DIR / 'validation_labels.csv')
test_seqs = pd.read_csv(STANFORD_DIR / 'test_sequences.csv')
eval_results = pd.read_csv(RESULTS_DIR / 'rmsd_evaluation.csv')
pred_summary = pd.read_csv(RESULTS_DIR / 'prediction_summary.csv')

print(f"  ✓ Loaded {len(train_coords)} templates")
print(f"  ✓ Loaded {len(test_seqs)} test sequences")
print()

# Initialize pipeline
pipeline = TBMPipeline(train_coords, train_sequences)

# Process validation coordinates
print("[2/4] Processing validation coordinates...")
valid_coords = {}
for target_id in test_seqs['target_id']:
    target_labels = valid_labels[valid_labels['ID'].str.startswith(target_id)]
    if len(target_labels) == 0:
        continue

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

# Focus on failures with high template identity
print("[3/4] Analyzing failures with high template identity...")
print("-" * 80)

failures = eval_results[
    (eval_results['status'] == 'success') &
    (eval_results['best_tm_score'] < 0.3)
]

merged = failures.merge(pred_summary[['target_id', 'best_template', 'best_identity']],
                        on='target_id', how='left')

print(f"\nFound {len(merged)} predictions with TM < 0.3")
print()

for idx, row in merged.iterrows():
    target_id = row['target_id']
    best_template = row['best_template']
    template_identity = row['best_identity']
    tm_score = row['best_tm_score']

    print(f"\n{'='*80}")
    print(f"TARGET: {target_id}")
    print(f"{'='*80}")

    # Get sequence
    test_seq = test_seqs[test_seqs['target_id'] == target_id].iloc[0]['sequence']
    print(f"Sequence length: {len(test_seq)}")
    print(f"Best template: {best_template}")
    print(f"Template identity: {template_identity*100:.1f}%")
    print(f"TM-score: {tm_score:.3f}")

    # Check if template exists and has coordinates
    if best_template not in train_coords:
        print(f"  ⚠️  Template {best_template} NOT FOUND in training data!")
        continue

    template_coords = train_coords[best_template]
    template_seq = train_sequences.get(best_template, "UNKNOWN")

    print(f"\nTemplate info:")
    print(f"  Sequence length: {len(template_seq)}")
    print(f"  Coordinates shape: {template_coords.shape}")
    print(f"  Non-NaN coords: {(~np.isnan(template_coords).any(axis=1)).sum()}")

    # Check validation coordinates
    if target_id in valid_coords:
        true_coords = valid_coords[target_id]
        print(f"\nValidation info:")
        print(f"  Coordinates shape: {true_coords.shape}")
        print(f"  Non-NaN coords: {(~np.isnan(true_coords).any(axis=1)).sum()}")

        # Check coordinate ranges
        template_valid = template_coords[~np.isnan(template_coords).any(axis=1)]
        true_valid = true_coords[~np.isnan(true_coords).any(axis=1)]

        if len(template_valid) > 0 and len(true_valid) > 0:
            print(f"\nCoordinate ranges:")
            print(f"  Template: X[{template_valid[:, 0].min():.2f}, {template_valid[:, 0].max():.2f}], "
                  f"Y[{template_valid[:, 1].min():.2f}, {template_valid[:, 1].max():.2f}], "
                  f"Z[{template_valid[:, 2].min():.2f}, {template_valid[:, 2].max():.2f}]")
            print(f"  True:     X[{true_valid[:, 0].min():.2f}, {true_valid[:, 0].max():.2f}], "
                  f"Y[{true_valid[:, 1].min():.2f}, {true_valid[:, 1].max():.2f}], "
                  f"Z[{true_valid[:, 2].min():.2f}, {true_valid[:, 2].max():.2f}]")

            # Calculate coordinate scale difference
            template_scale = np.sqrt(np.sum(template_valid**2, axis=1)).mean()
            true_scale = np.sqrt(np.sum(true_valid**2, axis=1)).mean()
            print(f"\nCoordinate scales:")
            print(f"  Template mean distance from origin: {template_scale:.2f}")
            print(f"  True mean distance from origin: {true_scale:.2f}")
            print(f"  Scale ratio: {template_scale/true_scale:.2f}")

            # Check if scales are very different (possible issue)
            if abs(template_scale/true_scale - 1.0) > 0.1:
                print(f"  ⚠️  WARNING: Coordinate scales differ by >10%!")

            # Check center of mass
            template_com = template_valid.mean(axis=0)
            true_com = true_valid.mean(axis=0)
            com_distance = np.linalg.norm(template_com - true_com)
            print(f"\nCenter of mass:")
            print(f"  Template: ({template_com[0]:.2f}, {template_com[1]:.2f}, {template_com[2]:.2f})")
            print(f"  True:     ({true_com[0]:.2f}, {true_com[1]:.2f}, {true_com[2]:.2f})")
            print(f"  Distance: {com_distance:.2f}")

    else:
        print(f"  ⚠️  No validation coordinates found for {target_id}")

    # Load and check prediction
    pred_files = list(RESULTS_DIR.glob(f"{target_id}_pred_*.pdb"))
    if pred_files:
        print(f"\nPrediction info:")
        print(f"  Number of predictions: {len(pred_files)}")

        # Read first prediction
        pred_coords = []
        with open(pred_files[0], 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    pred_coords.append([x, y, z])

        pred_coords = np.array(pred_coords)
        print(f"  Predicted coords shape: {pred_coords.shape}")
        print(f"  Non-NaN coords: {(~np.isnan(pred_coords).any(axis=1)).sum()}")

        if len(pred_coords) > 0:
            print(f"  Prediction range: X[{pred_coords[:, 0].min():.2f}, {pred_coords[:, 0].max():.2f}], "
                  f"Y[{pred_coords[:, 1].min():.2f}, {pred_coords[:, 1].max():.2f}], "
                  f"Z[{pred_coords[:, 2].min():.2f}, {pred_coords[:, 2].max():.2f}]")

            pred_scale = np.sqrt(np.sum(pred_coords**2, axis=1)).mean()
            print(f"  Prediction scale: {pred_scale:.2f}")

            # Compare prediction to template
            if len(pred_coords) == len(template_valid):
                direct_rmsd = np.sqrt(np.mean(np.sum((pred_coords - template_valid)**2, axis=1)))
                print(f"\nDirect RMSD (pred vs template): {direct_rmsd:.3f} Å")
            else:
                print(f"\n  ⚠️  Length mismatch: pred={len(pred_coords)}, template={len(template_valid)}")

print()
print("=" * 80)
print("[4/4] SUMMARY OF FINDINGS")
print("=" * 80)
print()

# Summarize common issues
print("Common patterns in failures:")
print("1. Check if coordinate scales differ significantly")
print("2. Check if templates have missing/invalid coordinates")
print("3. Check if sequence alignment is producing correct mapping")
print("4. Check if there are coordinate system inconsistencies")
print()
print("=" * 80)
