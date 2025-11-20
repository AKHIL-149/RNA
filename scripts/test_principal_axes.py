#!/usr/bin/env python3
"""
Test Principal Axis Alignment

Test the impact of principal axis alignment on challenging cases.
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

print("=" * 80)
print("TESTING PRINCIPAL AXIS ALIGNMENT")
print("=" * 80)
print()

# Load data
print("[1/3] Loading data...")
with open(PROJECT_DIR / 'data' / 'train_coords_dict.pkl', 'rb') as f:
    train_coords = pickle.load(f)
with open(PROJECT_DIR / 'data' / 'train_sequences_dict.pkl', 'rb') as f:
    train_sequences = pickle.load(f)

valid_labels = pd.read_csv(STANFORD_DIR / 'validation_labels.csv')
test_seqs = pd.read_csv(STANFORD_DIR / 'test_sequences.csv')

print(f"  ✓ Loaded {len(train_coords)} templates")
print()

# Initialize pipeline
pipeline = TBMPipeline(train_coords, train_sequences)

# Test cases with coordinate system issues
test_targets = [
    ('R1156', 0.556, 'Moderate - worst case'),
    ('R1190', 0.694, 'Good - needs improvement'),
    ('R1149', 0.848, 'Good - near excellent'),
    ('R1189', 0.865, 'Good - near excellent'),
]

print("[2/3] Testing on challenging cases...")
print()

results = []

for target_id, baseline_tm, description in test_targets:
    print(f"\n{'='*80}")
    print(f"TARGET: {target_id} - {description}")
    print(f"  Baseline TM: {baseline_tm:.3f}")
    print(f"{'='*80}")

    # Get sequence
    test_row = test_seqs[test_seqs['target_id'] == target_id]
    if len(test_row) == 0:
        print(f"  ⚠️  Target not found")
        continue

    query_seq = test_row.iloc[0]['sequence']
    print(f"  Sequence length: {len(query_seq)}")

    # Find template
    templates = pipeline.find_templates(query_seq, top_n=1, min_identity=0.5)
    if len(templates) == 0:
        print(f"  ⚠️  No templates found")
        continue

    best_template = templates[0]['template_id']
    print(f"  Best template: {best_template} ({templates[0]['identity']*100:.1f}% identity)")

    # Predict
    pred_coords = pipeline.predict_single_template(query_seq, best_template)

    # Get validation coordinates
    target_labels = valid_labels[valid_labels['ID'].str.startswith(target_id)]
    true_coords = []
    for _, row in target_labels.sort_values('resid').iterrows():
        if pd.notna(row['x_1']) and pd.notna(row['y_1']) and pd.notna(row['z_1']):
            true_coords.append([row['x_1'], row['y_1'], row['z_1']])
        else:
            true_coords.append([np.nan, np.nan, np.nan])
    true_coords = np.array(true_coords)

    # Evaluate with both methods
    print(f"\n  [Method 1] Standard centering...")
    standard_metrics = evaluate_structure_similarity(pred_coords, true_coords,
                                                      use_principal_axes=False)
    standard_tm = standard_metrics['tm_score']
    standard_rmsd = standard_metrics['aligned_rmsd']
    print(f"    TM = {standard_tm:.3f}, RMSD = {standard_rmsd:.3f} Å")

    print(f"  [Method 2] Principal axis alignment...")
    pca_metrics = evaluate_structure_similarity(pred_coords, true_coords,
                                                 use_principal_axes=True)
    pca_tm = pca_metrics['tm_score']
    pca_rmsd = pca_metrics['aligned_rmsd']
    print(f"    TM = {pca_tm:.3f}, RMSD = {pca_rmsd:.3f} Å")

    # Calculate improvement
    improvement = pca_tm - standard_tm
    print(f"\n  Improvement: {improvement:+.3f} TM-score", end='')
    if improvement > 0.01:
        print(" ✓ SIGNIFICANT")
    elif improvement > 0:
        print(" ✓")
    else:
        print()

    results.append({
        'target_id': target_id,
        'baseline_tm': baseline_tm,
        'standard_tm': standard_tm,
        'pca_tm': pca_tm,
        'improvement': improvement,
        'pca_rmsd': pca_rmsd,
        'standard_rmsd': standard_rmsd
    })

print()
print("=" * 80)
print("[3/3] SUMMARY")
print("=" * 80)
print()

results_df = pd.DataFrame(results)

print("Performance Comparison:")
print(f"{'─'*80}")
print(f"{'Target':<10} {'Baseline':<10} {'Standard':<10} {'PCA':<10} {'Δ TM':<10} {'Best':<10}")
print(f"{'─'*80}")

for _, row in results_df.iterrows():
    target = row['target_id']
    baseline = f"{row['baseline_tm']:.3f}"
    standard = f"{row['standard_tm']:.3f}"
    pca = f"{row['pca_tm']:.3f}"
    delta = f"{row['improvement']:+.3f}"

    # Determine best method
    if row['pca_tm'] > row['standard_tm']:
        best = "PCA ✓"
    elif row['standard_tm'] > row['pca_tm']:
        best = "Standard"
    else:
        best = "Equal"

    print(f"{target:<10} {baseline:<10} {standard:<10} {pca:<10} {delta:<10} {best:<10}")

print(f"{'─'*80}")
print()

# Calculate average improvement
if len(results_df) > 0:
    avg_improvement = results_df['improvement'].mean()
    print(f"Average Improvement: {avg_improvement:+.3f} TM-score")

    # Project to full dataset
    baseline_mean = 0.834
    n_total = 10  # Clean sequences
    n_tested = len(results_df)

    if avg_improvement > 0:
        # Estimate impact on full dataset
        estimated_new_mean = baseline_mean + (avg_improvement * n_tested / n_total)
        print(f"\nProjected Full Dataset Impact:")
        print(f"  Current mean TM-score: {baseline_mean:.3f}")
        print(f"  Estimated improvement: +{avg_improvement * n_tested / n_total:.3f}")
        print(f"  Projected new mean: {estimated_new_mean:.3f}")

        if estimated_new_mean >= 0.87:
            print(f"\n✓ GOAL ACHIEVED! Projected mean ≥ 0.87")
        else:
            gap = 0.87 - estimated_new_mean
            print(f"\n⚠️  Still {gap:.3f} below 0.87 target")
    else:
        print(f"\n⚠️  Principal axis alignment not helping")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if avg_improvement > 0.01:
    print("✓ Principal axis alignment shows significant improvement!")
    print("✓ Deploy this method for full evaluation.")
elif avg_improvement > 0:
    print("✓ Principal axis alignment shows minor improvement.")
    print("⚠️  Consider combining with other optimizations.")
else:
    print("⚠️  Principal axis alignment not effective for this dataset.")
    print("⚠️  The coordinate centering we already have is sufficient.")

print()
print("=" * 80)
