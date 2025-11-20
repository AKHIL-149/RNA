#!/usr/bin/env python3
"""
Test Fragment Assembly on R1138

Test the fragment assembly approach on R1138 (720nt long sequence).
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline, predict_long_sequence
from src.evaluation import evaluate_structure_similarity

PROJECT_DIR = Path(__file__).parent.parent
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'

print("=" * 80)
print("TESTING FRAGMENT ASSEMBLY ON R1138")
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

# Target: R1138 (720nt)
target_id = 'R1138'
print(f"[2/4] Testing on {target_id}...")

# Get sequence
test_row = test_seqs[test_seqs['target_id'] == target_id].iloc[0]
query_seq = test_row['sequence']
print(f"  Sequence length: {len(query_seq)} nt")
print()

# Get validation coordinates
print("[3/4] Preparing validation data...")
target_labels = valid_labels[valid_labels['ID'].str.startswith(target_id)]
true_coords = []
for _, row in target_labels.sort_values('resid').iterrows():
    if pd.notna(row['x_1']) and pd.notna(row['y_1']) and pd.notna(row['z_1']):
        true_coords.append([row['x_1'], row['y_1'], row['z_1']])
    else:
        true_coords.append([np.nan, np.nan, np.nan])
true_coords = np.array(true_coords)
print(f"  ✓ Loaded {len(true_coords)} validation coordinates")
print()

# Test different approaches
print("[4/4] Testing predictions...")
print()

results = []

# Method 1: Single template (baseline)
print("Method 1: Single Template (Baseline)")
print("-" * 80)
templates = pipeline.find_templates(query_seq, top_n=1, min_identity=0.5)
if len(templates) > 0:
    best_template = templates[0]['template_id']
    print(f"  Best template: {best_template} ({templates[0]['identity']*100:.1f}% identity)")
    
    try:
        pred_single = pipeline.predict_single_template(query_seq, best_template)
        metrics_single = evaluate_structure_similarity(pred_single, true_coords)
        tm_single = metrics_single['tm_score']
        rmsd_single = metrics_single['aligned_rmsd']
        
        if tm_single is not None:
            print(f"  TM-score: {tm_single:.3f}")
            print(f"  RMSD: {rmsd_single:.3f} Å")
            
            results.append({
                'method': 'Single Template',
                'tm_score': tm_single,
                'rmsd': rmsd_single
            })
        else:
            print(f"  ✗ Evaluation failed (returned None)")
            tm_single = None
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        tm_single = None
else:
    print(f"  ✗ No templates found")
    tm_single = None

print()

# Method 2: Fragment assembly (100nt fragments, 20nt overlap)
print("Method 2: Fragment Assembly (100nt fragments, 20nt overlap)")
print("-" * 80)
try:
    pred_frag100 = predict_long_sequence(
        query_seq,
        pipeline.find_templates,
        train_coords,
        train_sequences,
        lambda seq, tid: pipeline.predict_single_template(seq, tid),
        fragment_size=100,
        overlap=20
    )
    
    # Check if prediction is valid
    if pred_frag100 is None:
        print(f"  ✗ Prediction returned None")
        tm_frag100 = None
    elif np.isnan(pred_frag100).all():
        print(f"  ✗ All coordinates are NaN")
        tm_frag100 = None
    else:
        valid_count = (~np.isnan(pred_frag100).any(axis=1)).sum()
        print(f"  Valid coordinates: {valid_count}/{len(pred_frag100)}")
        
        metrics_frag100 = evaluate_structure_similarity(pred_frag100, true_coords)
        tm_frag100 = metrics_frag100['tm_score']
        rmsd_frag100 = metrics_frag100['aligned_rmsd']
        
        if tm_frag100 is not None:
            print(f"  TM-score: {tm_frag100:.3f}")
            print(f"  RMSD: {rmsd_frag100:.3f} Å")
            
            if tm_single is not None:
                improvement = tm_frag100 - tm_single
                print(f"  Improvement: {improvement:+.3f}", end='')
                if improvement > 0.05:
                    print(" ✓ SIGNIFICANT")
                elif improvement > 0:
                    print(" ✓")
                else:
                    print()
            
            results.append({
                'method': 'Fragment 100/20',
                'tm_score': tm_frag100,
                'rmsd': rmsd_frag100
            })
        else:
            print(f"  ✗ Evaluation returned None")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    tm_frag100 = None

print()

# Method 3: Fragment assembly (150nt fragments, 30nt overlap)  
print("Method 3: Fragment Assembly (150nt fragments, 30nt overlap)")
print("-" * 80)
try:
    pred_frag150 = predict_long_sequence(
        query_seq,
        pipeline.find_templates,
        train_coords,
        train_sequences,
        lambda seq, tid: pipeline.predict_single_template(seq, tid),
        fragment_size=150,
        overlap=30
    )
    
    # Check if prediction is valid
    if pred_frag150 is None:
        print(f"  ✗ Prediction returned None")
        tm_frag150 = None
    elif np.isnan(pred_frag150).all():
        print(f"  ✗ All coordinates are NaN")
        tm_frag150 = None
    else:
        valid_count = (~np.isnan(pred_frag150).any(axis=1)).sum()
        print(f"  Valid coordinates: {valid_count}/{len(pred_frag150)}")
        
        metrics_frag150 = evaluate_structure_similarity(pred_frag150, true_coords)
        tm_frag150 = metrics_frag150['tm_score']
        rmsd_frag150 = metrics_frag150['aligned_rmsd']
        
        if tm_frag150 is not None:
            print(f"  TM-score: {tm_frag150:.3f}")
            print(f"  RMSD: {rmsd_frag150:.3f} Å")
            
            if tm_single is not None:
                improvement = tm_frag150 - tm_single
                print(f"  Improvement: {improvement:+.3f}", end='')
                if improvement > 0.05:
                    print(" ✓ SIGNIFICANT")
                elif improvement > 0:
                    print(" ✓")
                else:
                    print()
            
            results.append({
                'method': 'Fragment 150/30',
                'tm_score': tm_frag150,
                'rmsd': rmsd_frag150
            })
        else:
            print(f"  ✗ Evaluation returned None")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    tm_frag150 = None

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

if len(results) > 0:
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    
    if tm_single is not None and len(results) > 1:
        # Find best method
        best_row = results_df.loc[results_df['tm_score'].idxmax()]
        print(f"Best Method: {best_row['method']}")
        print(f"  TM-score: {best_row['tm_score']:.3f}")
        print(f"  Improvement over baseline: {best_row['tm_score'] - tm_single:+.3f}")
        print()
        
        # Project impact on full dataset
        if best_row['tm_score'] > tm_single:
            baseline_mean = 0.834
            r1138_contribution_old = tm_single / 10  # 1/10 of dataset
            r1138_contribution_new = best_row['tm_score'] / 10
            improvement = r1138_contribution_new - r1138_contribution_old
            new_mean = baseline_mean + improvement
            
            print("Projected Impact on Full Dataset:")
            print(f"  Current mean TM-score: {baseline_mean:.3f}")
            print(f"  R1138 improvement: {best_row['tm_score'] - tm_single:+.3f}")
            print(f"  Dataset improvement: +{improvement:.3f}")
            print(f"  New mean: {new_mean:.3f}")
            print()
            
            if new_mean >= 0.85:
                print("✓ Would significantly improve overall performance!")
            else:
                gap = 0.87 - new_mean
                print(f"⚠️  Still {gap:.3f} below 0.87 target")
        else:
            print("⚠️  Fragment assembly not improving R1138")
    elif len(results) == 1:
        print("Only baseline method succeeded")
else:
    print("No methods succeeded")

print()
print("=" * 80)
