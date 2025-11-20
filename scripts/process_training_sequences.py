#!/usr/bin/env python3
"""
Process Training Sequences

This script loads training sequences and creates a dictionary mapping
target IDs to sequences for template-based modeling.
"""

import pandas as pd
import pickle
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
OUTPUT_DIR = PROJECT_DIR / 'data'

print("=" * 60)
print("PROCESSING TRAINING SEQUENCES")
print("=" * 60)
print()

# Try to load v2 first, fall back to v1
sequences_file = DATA_DIR / 'train_sequences.v2.csv'
if not sequences_file.exists():
    sequences_file = DATA_DIR / 'train_sequences.csv'
    print(f"Using v1 sequences: {sequences_file}")
else:
    print(f"Using v2 sequences: {sequences_file}")

# Load sequences
print(f"\nLoading sequences from {sequences_file}...")
train_seqs = pd.read_csv(sequences_file)
print(f"Loaded {len(train_seqs)} sequences")

# Create dictionary
sequences_dict = {}
for _, row in train_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    sequences_dict[target_id] = sequence

print(f"\nCreated dictionary with {len(sequences_dict)} sequences")

# Sample statistics
seq_lengths = [len(seq) for seq in sequences_dict.values()]
print(f"\nSequence length statistics:")
print(f"  Min: {min(seq_lengths)} nt")
print(f"  Max: {max(seq_lengths)} nt")
print(f"  Mean: {sum(seq_lengths)/len(seq_lengths):.1f} nt")

# Save dictionary
output_path = OUTPUT_DIR / 'train_sequences_dict.pkl'
print(f"\nSaving to {output_path}...")
with open(output_path, 'wb') as f:
    pickle.dump(sequences_dict, f)

print(f"✓ Saved successfully")
print()

print("=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)
print(f"Sequences dictionary saved to: {output_path}")
print()
