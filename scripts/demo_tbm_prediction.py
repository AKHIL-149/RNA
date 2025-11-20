#!/usr/bin/env python3
"""
Demo: TBM Prediction Pipeline

This script demonstrates the complete TBM workflow on a single test sequence.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline, save_prediction_to_pdb

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
RESULTS_DIR = PROJECT_DIR / 'results' / 'demo'

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TBM PREDICTION DEMO")
print("=" * 60)
print()

# Step 1: Load training data
print("[Step 1] Loading training data...")
print("-" * 60)

coords_path = DATA_DIR / 'train_coords_dict.pkl'
sequences_path = DATA_DIR / 'train_sequences_dict.pkl'

print(f"Loading coordinates from {coords_path}...")
with open(coords_path, 'rb') as f:
    train_coords = pickle.load(f)
print(f"  ✓ Loaded {len(train_coords)} template structures")

print(f"\nLoading sequences from {sequences_path}...")
with open(sequences_path, 'rb') as f:
    train_sequences = pickle.load(f)
print(f"  ✓ Loaded {len(train_sequences)} template sequences")
print()

# Step 2: Initialize pipeline
print("[Step 2] Initializing TBM pipeline...")
print("-" * 60)
pipeline = TBMPipeline(train_coords, train_sequences)
print("  ✓ Pipeline initialized")
print()

# Step 3: Load a test sequence
print("[Step 3] Loading test sequence...")
print("-" * 60)

test_seqs = pd.read_csv(STANFORD_DIR / 'test_sequences.csv')
print(f"Available test sequences: {len(test_seqs)}")

# Use the first test sequence
test_row = test_seqs.iloc[0]
target_id = test_row['target_id']
query_seq = test_row['sequence']

print(f"\nSelected: {target_id}")
print(f"Sequence length: {len(query_seq)} nt")
print(f"Sequence: {query_seq[:50]}..." if len(query_seq) > 50 else f"Sequence: {query_seq}")
print()

# Step 4: Find templates
print("[Step 4] Finding similar templates...")
print("-" * 60)

template_matches = pipeline.find_templates(query_seq, top_n=10, min_identity=0.3)

if len(template_matches) == 0:
    print("⚠ No templates found with identity >= 30%")
    print("Trying with lower threshold...")
    template_matches = pipeline.find_templates(query_seq, top_n=10, min_identity=0.2)

print(f"Found {len(template_matches)} matching templates:")
for i, match in enumerate(template_matches[:5], 1):
    print(f"  {i}. {match['template_id']}: {match['identity']:.1%} identity")
print()

# Step 5: Generate predictions
print("[Step 5] Generating predictions...")
print("-" * 60)

if len(template_matches) > 0:
    try:
        # Single template prediction
        print("Using single best template...")
        pred_single = pipeline.predict_single_template(
            query_seq,
            template_matches[0]['template_id']
        )
        print(f"  ✓ Generated prediction (shape: {pred_single.shape})")

        # Save prediction
        output_path = RESULTS_DIR / f"{target_id}_single_template.pdb"
        save_prediction_to_pdb(pred_single, query_seq, str(output_path), target_id)
        print(f"  ✓ Saved to {output_path}")

        # Multi-template prediction (if we have multiple templates)
        if len(template_matches) >= 3:
            print("\nUsing multiple templates...")
            pred_multi = pipeline.predict_multi_template(
                query_seq,
                template_matches,
                use_top_n=min(5, len(template_matches))
            )
            print(f"  ✓ Generated prediction (shape: {pred_multi.shape})")

            # Save prediction
            output_path = RESULTS_DIR / f"{target_id}_multi_template.pdb"
            save_prediction_to_pdb(pred_multi, query_seq, str(output_path), target_id)
            print(f"  ✓ Saved to {output_path}")

        print()

        # Generate best-of-5 predictions
        print("\nGenerating best-of-5 predictions...")
        try:
            predictions = pipeline.predict(query_seq, n_predictions=5, strategy='multi_template')
            print(f"  ✓ Generated {len(predictions)} predictions")

            # Save all predictions
            for i, pred in enumerate(predictions, 1):
                output_path = RESULTS_DIR / f"{target_id}_pred_{i}.pdb"
                save_prediction_to_pdb(pred, query_seq, str(output_path), target_id)
            print(f"  ✓ Saved all predictions to {RESULTS_DIR}")

        except Exception as e:
            print(f"  ⚠ Could not generate best-of-5: {e}")

    except Exception as e:
        print(f"  ✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  ✗ No templates available for prediction")

print()

# Summary
print("=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
print()
print(f"Target: {target_id}")
print(f"Sequence length: {len(query_seq)} nt")
print(f"Templates found: {len(template_matches)}")
if len(template_matches) > 0:
    print(f"Best match: {template_matches[0]['template_id']} ({template_matches[0]['identity']:.1%})")
    print(f"\nPredictions saved to: {RESULTS_DIR}")
print()
