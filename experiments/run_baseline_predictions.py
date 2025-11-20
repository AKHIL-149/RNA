#!/usr/bin/env python3
"""
Run Baseline TBM Predictions

This script runs the TBM pipeline on all test sequences and generates predictions.
Uses an optimized approach for faster template matching.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline, save_prediction_to_pdb

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
RESULTS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BASELINE TBM PREDICTIONS")
print("=" * 80)
print()

# Load training data
print("[1/4] Loading training data...")
print("-" * 80)

coords_path = DATA_DIR / 'train_coords_dict.pkl'
sequences_path = DATA_DIR / 'train_sequences_dict.pkl'

print(f"Loading coordinates: {coords_path}")
with open(coords_path, 'rb') as f:
    train_coords = pickle.load(f)
print(f"  ✓ Loaded {len(train_coords)} template structures")

print(f"Loading sequences: {sequences_path}")
with open(sequences_path, 'rb') as f:
    train_sequences = pickle.load(f)
print(f"  ✓ Loaded {len(train_sequences)} template sequences")
print()

# Initialize pipeline
print("[2/4] Initializing TBM pipeline...")
print("-" * 80)
pipeline = TBMPipeline(train_coords, train_sequences)
print("  ✓ Pipeline ready")
print()

# Load test sequences
print("[3/4] Loading test sequences...")
print("-" * 80)

test_seqs = pd.read_csv(STANFORD_DIR / 'test_sequences.csv')
print(f"  ✓ Loaded {len(test_seqs)} test sequences")
print()

# Process test sequences
print("[4/4] Generating predictions...")
print("-" * 80)
print()

results = []
prediction_times = []

for idx, row in tqdm(test_seqs.iterrows(), total=len(test_seqs), desc="Processing"):
    target_id = row['target_id']
    query_seq = row['sequence']

    start_time = time.time()

    try:
        # Find templates (with lower threshold for faster processing)
        templates = pipeline.find_templates(query_seq, top_n=5, min_identity=0.25)

        if len(templates) == 0:
            # Try with even lower threshold
            templates = pipeline.find_templates(query_seq, top_n=5, min_identity=0.15)

        if len(templates) > 0:
            # Generate single prediction using best template
            pred = pipeline.predict_single_template(
                query_seq,
                templates[0]['template_id']
            )

            # Save prediction
            output_path = RESULTS_DIR / f"{target_id}_pred_1.pdb"
            save_prediction_to_pdb(pred, query_seq, str(output_path), target_id)

            # Try to generate multiple predictions if we have enough templates
            if len(templates) >= 3:
                try:
                    # Use multi-template approach for prediction 2
                    pred2 = pipeline.predict_multi_template(query_seq, templates, use_top_n=3)
                    output_path = RESULTS_DIR / f"{target_id}_pred_2.pdb"
                    save_prediction_to_pdb(pred2, query_seq, str(output_path), target_id)

                    # Use different template for prediction 3
                    if len(templates) >= 2:
                        pred3 = pipeline.predict_single_template(query_seq, templates[1]['template_id'])
                        output_path = RESULTS_DIR / f"{target_id}_pred_3.pdb"
                        save_prediction_to_pdb(pred3, query_seq, str(output_path), target_id)
                except Exception as e:
                    pass  # Skip additional predictions if error

            elapsed = time.time() - start_time
            prediction_times.append(elapsed)

            results.append({
                'target_id': target_id,
                'sequence_length': len(query_seq),
                'num_templates': len(templates),
                'best_template': templates[0]['template_id'],
                'best_identity': templates[0]['identity'],
                'time_seconds': elapsed,
                'status': 'success'
            })

        else:
            results.append({
                'target_id': target_id,
                'sequence_length': len(query_seq),
                'num_templates': 0,
                'best_template': None,
                'best_identity': 0.0,
                'time_seconds': time.time() - start_time,
                'status': 'no_templates'
            })

    except Exception as e:
        results.append({
            'target_id': target_id,
            'sequence_length': len(query_seq),
            'num_templates': 0,
            'best_template': None,
            'best_identity': 0.0,
            'time_seconds': time.time() - start_time,
            'status': f'error: {str(e)[:100]}'
        })

print()
print("=" * 80)
print("PREDICTION SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)

# Save results
results_csv = RESULTS_DIR / 'prediction_summary.csv'
results_df.to_csv(results_csv, index=False)
print(f"\n✓ Results saved to: {results_csv}")

# Statistics
successful = results_df[results_df['status'] == 'success']
no_templates = results_df[results_df['status'] == 'no_templates']
errors = results_df[~results_df['status'].isin(['success', 'no_templates'])]

print(f"\nTotal sequences: {len(results_df)}")
print(f"  ✓ Successful: {len(successful)}")
print(f"  ⚠ No templates: {len(no_templates)}")
print(f"  ✗ Errors: {len(errors)}")

if len(successful) > 0:
    print(f"\nTemplate statistics (successful predictions):")
    print(f"  Mean templates found: {successful['num_templates'].mean():.1f}")
    print(f"  Mean identity: {successful['best_identity'].mean():.2%}")

    print(f"\nTiming:")
    print(f"  Mean time: {np.mean(prediction_times):.1f}s per sequence")
    print(f"  Total time: {np.sum(prediction_times):.1f}s")

print(f"\nPredictions saved to: {RESULTS_DIR}")
print("=" * 80)
print()
