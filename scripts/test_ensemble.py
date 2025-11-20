#!/usr/bin/env python3
"""
Test Ensemble Module

Quick test of multi-template ensemble on challenging cases.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline, predict_multi_template_weighted, quality_weighted_ensemble
from src.evaluation import evaluate_structure_similarity
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'

print("=" * 80)
print("TESTING MULTI-TEMPLATE ENSEMBLE")
print("=" * 80)
print()

# Load data
print("[1/3] Loading data...")
with open(PROJECT_DIR / 'data' / 'train_coords_dict.pkl', 'rb') as f:
    train_coords = pickle.load(f)
with open(PROJECT_DIR / 'data' / 'train_sequences_dict.pkl', 'rb') as f:
    train_sequences = pickle.load(f)

# Load validation data
valid_labels = pd.read_csv(STANFORD_DIR / 'validation_labels.csv')
test_seqs = pd.read_csv(STANFORD_DIR / 'test_sequences.csv')

print(f"  ✓ Loaded {len(train_coords)} templates")
print()

# Initialize pipeline
pipeline = TBMPipeline(train_coords, train_sequences)

# Test on challenging cases
test_targets = [
    ('R1156', 'Moderate performer (TM=0.556)'),
    ('R1190', 'Good performer (TM=0.694)'),
    ('R1149', 'Good performer (TM=0.848)')
]

print("[2/3] Testing ensemble on challenging cases...")
print()

results = []

for target_id, description in test_targets:
    print(f"\n{'='*80}")
    print(f"TARGET: {target_id} - {description}")
    print(f"{'='*80}")

    # Get sequence
    test_row = test_seqs[test_seqs['target_id'] == target_id]
    if len(test_row) == 0:
        print(f"  ⚠️  Target not found")
        continue

    query_seq = test_row.iloc[0]['sequence']
    print(f"  Sequence length: {len(query_seq)}")

    # Find templates
    templates = pipeline.find_templates(query_seq, top_n=10, min_identity=0.5)
    print(f"  Templates found: {len(templates)}")

    if len(templates) == 0:
        print(f"  ⚠️  No templates found")
        continue

    print(f"\n  Top 5 templates:")
    for i, t in enumerate(templates[:5], 1):
        print(f"    {i}. {t['template_id']}: {t['identity']*100:.1f}% identity")

    # Test 1: Single template (baseline)
    print(f"\n  [Test 1] Single template prediction...")
    single_pred = pipeline.predict_single_template(query_seq, templates[0]['template_id'])

    # Test 2: Multi-template weighted ensemble
    print(f"  [Test 2] Multi-template weighted ensemble (top 3)...")
    try:
        ensemble_pred = predict_multi_template_weighted(
            query_seq,
            templates,
            train_coords,
            lambda seq, tid: pipeline.predict_single_template(seq, tid),
            top_n=3,
            weighting='squared'
        )
        ensemble_success = True
    except Exception as e:
        print(f"    ⚠️  Ensemble failed: {e}")
        ensemble_pred = None
        ensemble_success = False

    # Test 3: Quality-weighted ensemble
    print(f"  [Test 3] Quality-weighted ensemble (top 5)...")
    try:
        quality_pred = quality_weighted_ensemble(
            query_seq,
            templates,
            train_coords,
            train_sequences,
            lambda seq, tid: pipeline.predict_single_template(seq, tid),
            top_n=5
        )
        quality_success = True
    except Exception as e:
        print(f"    ⚠️  Quality ensemble failed: {e}")
        quality_pred = None
        quality_success = False

    # Get validation coordinates
    target_labels = valid_labels[valid_labels['ID'].str.startswith(target_id)]
    true_coords = []
    for _, row in target_labels.sort_values('resid').iterrows():
        if pd.notna(row['x_1']) and pd.notna(row['y_1']) and pd.notna(row['z_1']):
            true_coords.append([row['x_1'], row['y_1'], row['z_1']])
        else:
            true_coords.append([np.nan, np.nan, np.nan])

    true_coords = np.array(true_coords)

    # Evaluate
    print(f"\n  Evaluation Results:")
    print(f"  {'─'*76}")

    # Single template
    if single_pred is not None:
        single_metrics = evaluate_structure_similarity(single_pred, true_coords)
        single_tm = single_metrics['tm_score']
        single_rmsd = single_metrics['aligned_rmsd']
        print(f"    Single template:  TM = {single_tm:.3f}, RMSD = {single_rmsd:.3f} Å")
    else:
        single_tm = None
        single_rmsd = None
        print(f"    Single template:  FAILED")

    # Ensemble
    if ensemble_success and ensemble_pred is not None:
        ensemble_metrics = evaluate_structure_similarity(ensemble_pred, true_coords)
        ensemble_tm = ensemble_metrics['tm_score']
        ensemble_rmsd = ensemble_metrics['aligned_rmsd']
        print(f"    Ensemble (3):     TM = {ensemble_tm:.3f}, RMSD = {ensemble_rmsd:.3f} Å", end='')

        if single_tm is not None:
            improvement = ensemble_tm - single_tm
            if improvement > 0:
                print(f"  [+{improvement:.3f}] ✓")
            else:
                print(f"  [{improvement:.3f}]")
        else:
            print()
    else:
        ensemble_tm = None
        ensemble_rmsd = None
        print(f"    Ensemble (3):     FAILED")

    # Quality ensemble
    if quality_success and quality_pred is not None:
        quality_metrics = evaluate_structure_similarity(quality_pred, true_coords)
        quality_tm = quality_metrics['tm_score']
        quality_rmsd = quality_metrics['aligned_rmsd']
        print(f"    Quality (5):      TM = {quality_tm:.3f}, RMSD = {quality_rmsd:.3f} Å", end='')

        if single_tm is not None:
            improvement = quality_tm - single_tm
            if improvement > 0:
                print(f"  [+{improvement:.3f}] ✓")
            else:
                print(f"  [{improvement:.3f}]")
        else:
            print()
    else:
        quality_tm = None
        quality_rmsd = None
        print(f"    Quality (5):      FAILED")

    results.append({
        'target_id': target_id,
        'single_tm': single_tm,
        'ensemble_tm': ensemble_tm,
        'quality_tm': quality_tm,
        'ensemble_improvement': ensemble_tm - single_tm if (ensemble_tm and single_tm) else None,
        'quality_improvement': quality_tm - single_tm if (quality_tm and single_tm) else None
    })

print()
print("=" * 80)
print("[3/3] SUMMARY")
print("=" * 80)
print()

# Summary table
results_df = pd.DataFrame(results)

print("Performance Comparison:")
print(f"{'─'*80}")
print(f"{'Target':<10} {'Single':<10} {'Ensemble(3)':<12} {'Quality(5)':<12} {'Best Method':<15}")
print(f"{'─'*80}")

for _, row in results_df.iterrows():
    target = row['target_id']
    single = f"{row['single_tm']:.3f}" if row['single_tm'] else "N/A"
    ensemble = f"{row['ensemble_tm']:.3f}" if row['ensemble_tm'] else "N/A"
    quality = f"{row['quality_tm']:.3f}" if row['quality_tm'] else "N/A"

    # Find best
    scores = {
        'Single': row['single_tm'],
        'Ensemble': row['ensemble_tm'],
        'Quality': row['quality_tm']
    }
    valid_scores = {k: v for k, v in scores.items() if v is not None}

    if valid_scores:
        best_method = max(valid_scores, key=valid_scores.get)
        best_score = valid_scores[best_method]
    else:
        best_method = "None"

    print(f"{target:<10} {single:<10} {ensemble:<12} {quality:<12} {best_method:<15}")

print(f"{'─'*80}")
print()

# Calculate improvements
valid_improvements = [r for r in results if r['ensemble_improvement'] is not None]

if valid_improvements:
    avg_ensemble_improvement = np.mean([r['ensemble_improvement'] for r in valid_improvements])
    avg_quality_improvement = np.mean([r['quality_improvement'] for r in valid_improvements if r['quality_improvement'] is not None])

    print(f"Average Improvements:")
    print(f"  Ensemble (3 templates):  {avg_ensemble_improvement:+.3f} TM-score")
    if not np.isnan(avg_quality_improvement):
        print(f"  Quality (5 templates):   {avg_quality_improvement:+.3f} TM-score")
    print()

    if avg_ensemble_improvement > 0:
        print(f"✓ Ensemble method shows improvement!")
    else:
        print(f"⚠️  Ensemble method not improving performance")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if valid_improvements and avg_ensemble_improvement > 0.01:
    print("✓ Multi-template ensemble is working and improving performance.")
    print("✓ Proceed with full evaluation using ensemble method.")
    print()
    print("Expected impact on full dataset:")
    print(f"  Current mean TM-score: 0.834")
    print(f"  Estimated improvement: +{avg_ensemble_improvement:.3f}")
    print(f"  Expected new mean: {0.834 + avg_ensemble_improvement:.3f}")
else:
    print("⚠️  Ensemble method needs tuning before full deployment.")
    print("⚠️  Consider adjusting:")
    print("    - Weighting scheme")
    print("    - Number of templates")
    print("    - Diversity selection threshold")

print()
print("=" * 80)
