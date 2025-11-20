#!/usr/bin/env python3
"""
Evaluate All Sequences with Smart Ensemble

Runs full evaluation using the smart ensemble method.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline, predict_multi_template_weighted
from src.evaluation import evaluate_structure_similarity

PROJECT_DIR = Path(__file__).parent.parent
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
RESULTS_DIR = PROJECT_DIR / 'results' / 'tbm_ensemble'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("EVALUATING WITH SMART ENSEMBLE")
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

print(f"  ✓ Loaded {len(train_coords)} templates")
print()

# Initialize pipeline
pipeline = TBMPipeline(train_coords, train_sequences)

# Prepare validation coordinates
print("[2/4] Preparing validation coordinates...")
valid_coords = {}
for target_id in test_seqs['target_id']:
    target_labels = valid_labels[valid_labels['ID'].str.startswith(target_id)]
    coords = []
    for _, row in target_labels.sort_values('resid').iterrows():
        if pd.notna(row['x_1']) and pd.notna(row['y_1']) and pd.notna(row['z_1']):
            coords.append([row['x_1'], row['y_1'], row['z_1']])
        else:
            coords.append([np.nan, np.nan, np.nan])
    valid_coords[target_id] = np.array(coords)

print(f"  ✓ Prepared {len(valid_coords)} validation structures")
print()

# Evaluate
print("[3/4] Running predictions with ensemble...")
results = []

for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    query_seq = row['sequence']
    
    print(f"\n  [{idx+1}/{len(test_seqs)}] {target_id} ({len(query_seq)}nt)...", end=' ')
    
    # Find templates
    templates = pipeline.find_templates(query_seq, top_n=10, min_identity=0.5)
    
    if len(templates) == 0:
        print("No templates")
        results.append({
            'target_id': target_id,
            'status': 'no_templates',
            'tm_score': None,
            'rmsd': None
        })
        continue
    
    # Predict with ensemble
    try:
        pred_coords = predict_multi_template_weighted(
            query_seq,
            templates,
            train_coords,
            lambda seq, tid: pipeline.predict_single_template(seq, tid),
            top_n=3,
            weighting='squared',
            min_identity_for_ensemble=0.99  # Smart ensemble
        )
        
        if pred_coords is None:
            print("Prediction failed")
            results.append({
                'target_id': target_id,
                'status': 'prediction_failed',
                'tm_score': None,
                'rmsd': None
            })
            continue
        
        # Evaluate
        true_coords = valid_coords[target_id]
        metrics = evaluate_structure_similarity(pred_coords, true_coords)
        
        print(f"TM={metrics['tm_score']:.3f}")
        
        results.append({
            'target_id': target_id,
            'status': 'success',
            'tm_score': metrics['tm_score'],
            'rmsd': metrics['aligned_rmsd'],
            'num_templates': len(templates),
            'best_identity': templates[0]['identity']
        })
        
    except Exception as e:
        print(f"Error: {e}")
        results.append({
            'target_id': target_id,
            'status': 'error',
            'tm_score': None,
            'rmsd': None,
            'error': str(e)
        })

print()
print()

# Save results
print("[4/4] Saving results...")
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / 'ensemble_evaluation.csv', index=False)
print(f"  ✓ Saved to {RESULTS_DIR / 'ensemble_evaluation.csv'}")
print()

# Calculate metrics
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

successful = results_df[results_df['status'] == 'success']

# Overall stats
print("Overall Performance:")
print(f"  Total sequences: {len(results_df)}")
print(f"  Successful: {len(successful)}/{len(results_df)} ({len(successful)/len(results_df)*100:.1f}%)")
print()

if len(successful) > 0:
    # Clean metrics (exclude R1116 if present)
    clean = successful[successful['target_id'] != 'R1116']
    
    print("Clean Performance (excluding R1116):")
    print(f"  Mean TM-score: {clean['tm_score'].mean():.3f}")
    print(f"  Median TM-score: {clean['tm_score'].median():.3f}")
    print(f"  Min TM-score: {clean['tm_score'].min():.3f}")
    print(f"  Max TM-score: {clean['tm_score'].max():.3f}")
    print()
    
    # Quality breakdown
    excellent = (clean['tm_score'] >= 0.9).sum()
    high = (clean['tm_score'] >= 0.7).sum()
    acceptable = (clean['tm_score'] >= 0.5).sum()
    
    print("Quality Distribution:")
    print(f"  Excellent (≥0.9): {excellent}/{len(clean)} ({excellent/len(clean)*100:.1f}%)")
    print(f"  High (≥0.7): {high}/{len(clean)} ({high/len(clean)*100:.1f}%)")
    print(f"  Acceptable (≥0.5): {acceptable}/{len(clean)} ({acceptable/len(clean)*100:.1f}%)")
    print()
    
    # Compare to baseline
    baseline_mean = 0.834
    improvement = clean['tm_score'].mean() - baseline_mean
    print(f"Comparison to Baseline:")
    print(f"  Baseline mean TM-score: {baseline_mean:.3f}")
    print(f"  Ensemble mean TM-score: {clean['tm_score'].mean():.3f}")
    print(f"  Improvement: {improvement:+.3f} ({improvement/baseline_mean*100:+.1f}%)")

print()
print("=" * 80)
