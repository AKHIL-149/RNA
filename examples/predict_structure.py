#!/usr/bin/env python3
"""
Example: Predict RNA 3D Structure

This example demonstrates how to use the TBM pipeline to predict
RNA 3D structures from sequence.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline
from src.evaluation import evaluate_structure_similarity

print("=" * 80)
print("RNA 3D STRUCTURE PREDICTION - EXAMPLE")
print("=" * 80)
print()

# Step 1: Load training data
print("[Step 1] Loading training data...")
PROJECT_DIR = Path(__file__).parent.parent

with open(PROJECT_DIR / 'data' / 'train_coords_dict.pkl', 'rb') as f:
    train_coords = pickle.load(f)
with open(PROJECT_DIR / 'data' / 'train_sequences_dict.pkl', 'rb') as f:
    train_sequences = pickle.load(f)

print(f"  ✓ Loaded {len(train_coords)} template structures")
print()

# Step 2: Initialize pipeline
print("[Step 2] Initializing TBM pipeline...")
pipeline = TBMPipeline(train_coords, train_sequences)
print(f"  ✓ Pipeline ready with {len(pipeline.templates)} templates indexed")
print()

# Step 3: Define query sequence (example: short RNA)
print("[Step 3] Defining query sequence...")
query_seq = "GCGGAUUUAGCUCAGDDGGGAGAGCMCCAGACUGAAGAUCUGGAGGGUCCUGUAGUACUUGUUAACCCC"
print(f"  Sequence: {query_seq[:50]}...")
print(f"  Length: {len(query_seq)} nucleotides")
print()

# Step 4: Find templates
print("[Step 4] Searching for templates...")
templates = pipeline.find_templates(query_seq, top_n=5, min_identity=0.5)

if len(templates) == 0:
    print("  ✗ No templates found!")
    sys.exit(1)

print(f"  ✓ Found {len(templates)} template(s)")
print()
print("  Top 3 templates:")
for i, template in enumerate(templates[:3], 1):
    print(f"    {i}. {template['template_id']}: "
          f"{template['identity']*100:.1f}% identity, "
          f"{template['coverage']*100:.1f}% coverage")
print()

# Step 5: Predict structure
print("[Step 5] Predicting 3D structure...")
best_template_id = templates[0]['template_id']
print(f"  Using template: {best_template_id}")

predicted_coords = pipeline.predict_single_template(query_seq, best_template_id)

if predicted_coords is None:
    print("  ✗ Prediction failed!")
    sys.exit(1)

print(f"  ✓ Prediction complete")
print(f"  Predicted coordinates shape: {predicted_coords.shape}")
print(f"  Valid residues: {(~np.isnan(predicted_coords).any(axis=1)).sum()}/{len(predicted_coords)}")
print()

# Step 6: Save prediction (optional)
print("[Step 6] Saving prediction...")
output_file = PROJECT_DIR / 'examples' / 'example_prediction.pdb'

from src.tbm import save_prediction_to_pdb
save_prediction_to_pdb(predicted_coords, query_seq, output_file)
print(f"  ✓ Saved to {output_file}")
print()

# Step 7: Evaluate (if validation data available)
print("[Step 7] Evaluation summary...")
print(f"  Predicted structure statistics:")
print(f"    - Length: {len(predicted_coords)} residues")
print(f"    - Valid coordinates: {(~np.isnan(predicted_coords).any(axis=1)).sum()}")
print(f"    - Coordinate range: {np.nanmin(predicted_coords):.2f} to {np.nanmax(predicted_coords):.2f} Å")

# Calculate center of mass
valid_mask = ~np.isnan(predicted_coords).any(axis=1)
if valid_mask.any():
    com = predicted_coords[valid_mask].mean(axis=0)
    print(f"    - Center of mass: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}) Å")
print()

print("=" * 80)
print("EXAMPLE COMPLETE")
print("=" * 80)
print()
print("Next steps:")
print("  1. Visualize structure: Open example_prediction.pdb in PyMOL or UCSF Chimera")
print("  2. Try different sequences")
print("  3. Experiment with multi-template ensemble (see README.md)")
print("  4. Evaluate against known structures")
print()
print("For more information:")
print("  - README.md: Project overview and usage")
print("  - FINAL_CONCLUSIONS.md: Complete analysis")
print("  - src/tbm/pipeline.py: Pipeline implementation")
print()
