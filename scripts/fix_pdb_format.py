#!/usr/bin/env python3
"""
Fix PDB Format

Re-generate PDB files with correct format for US-align compatibility.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import save_prediction_to_pdb

PROJECT_DIR = Path(__file__).parent.parent
PREDICTIONS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'

print("Fixing PDB files for US-align compatibility...")
print()

# Load test sequences to get sequences
test_seqs = pd.read_csv(STANFORD_DIR / 'test_sequences.csv')

# Find all prediction PDB files
pdb_files = sorted(PREDICTIONS_DIR.glob("*_pred_*.pdb"))

print(f"Found {len(pdb_files)} prediction files")

# Read and rewrite each file
for pdb_file in pdb_files:
    # Extract target_id from filename
    filename = pdb_file.stem  # e.g., "R1107_pred_1"
    target_id = filename.rsplit('_pred_', 1)[0]

    # Get sequence
    seq_row = test_seqs[test_seqs['target_id'] == target_id]
    if len(seq_row) == 0:
        print(f"  Warning: No sequence found for {target_id}")
        continue

    sequence = seq_row.iloc[0]['sequence']

    # Read coordinates from existing file
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    if len(coords) == 0:
        print(f"  Warning: No coordinates in {pdb_file.name}")
        continue

    # Pad with NaN to match sequence length
    coords_array = np.full((len(sequence), 3), np.nan)
    for i, coord in enumerate(coords):
        if i < len(coords_array):
            coords_array[i] = coord

    # Rewrite with correct format
    save_prediction_to_pdb(coords_array, sequence, str(pdb_file), target_id)

print(f"\n✓ Fixed {len(pdb_files)} PDB files")
print()
