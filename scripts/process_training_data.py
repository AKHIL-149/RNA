#!/usr/bin/env python3
"""
Process training labels into coordinate dictionary

This script converts the training labels CSV into a pickled dictionary
mapping target IDs to coordinate arrays for fast lookup during TBM.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

def process_labels(labels_df):
    """
    Convert labels DataFrame to coordinate dictionary

    Args:
        labels_df: DataFrame with columns [ID, resname, resid, x_1, y_1, z_1]

    Returns:
        dict: {target_id: np.array([[x1, y1, z1], [x2, y2, z2], ...])}
    """
    coords_dict = {}

    # Extract base ID (remove trailing _residue_number)
    print("Extracting base IDs...")
    base_ids = labels_df['ID'].str.rsplit('_', n=1).str[0]

    # Group by target ID and wrap with tqdm for progress tracking
    print("Processing structures...")
    id_groups = labels_df.groupby(base_ids)

    for id_prefix, group in tqdm(id_groups, desc="Processing structures"):
        # Extract coordinates for the first structure (x_1, y_1, z_1)
        coords = []
        for _, row in group.sort_values('resid').iterrows():
            # Check for missing coordinates
            if pd.notna(row['x_1']) and pd.notna(row['y_1']) and pd.notna(row['z_1']):
                coords.append([row['x_1'], row['y_1'], row['z_1']])
            else:
                # Skip structures with missing coordinates
                print(f"Warning: {id_prefix} has missing coordinates at residue {row['resid']}")
                coords = []
                break

        if coords:  # Only add if we have complete coordinates
            coords_dict[id_prefix] = np.array(coords)

    return coords_dict

def main():
    # File paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try to load v2 labels first (larger dataset)
    labels_path_v2 = os.path.join(project_root, 'stanford-rna-3d-folding', 'train_labels.v2.csv')
    labels_path_v1 = os.path.join(project_root, 'stanford-rna-3d-folding', 'train_labels.csv')
    output_path = os.path.join(project_root, 'data', 'train_coords_dict.pkl')

    # Check which file exists
    if os.path.exists(labels_path_v2):
        labels_path = labels_path_v2
        print(f"Using v2 labels: {labels_path}")
    elif os.path.exists(labels_path_v1):
        labels_path = labels_path_v1
        print(f"Using v1 labels: {labels_path}")
    else:
        print("Error: Could not find train_labels.csv or train_labels.v2.csv")
        print(f"Looked in: {os.path.dirname(labels_path_v1)}")
        sys.exit(1)

    # Load labels
    print(f"Loading labels from {labels_path}...")
    try:
        train_labels = pd.read_csv(labels_path)
        print(f"Loaded {len(train_labels)} label rows")
    except Exception as e:
        print(f"Error loading labels: {e}")
        sys.exit(1)

    # Process labels
    train_coords_dict = process_labels(train_labels)

    # Save to pickle file
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(train_coords_dict, f)

    # Summary
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Processed {len(train_coords_dict)} training structures")
    print(f"Output saved to: {output_path}")

    # Show some statistics
    if train_coords_dict:
        lengths = [len(coords) for coords in train_coords_dict.values()]
        print(f"\nSequence length statistics:")
        print(f"  Min: {min(lengths)} residues")
        print(f"  Max: {max(lengths)} residues")
        print(f"  Mean: {np.mean(lengths):.1f} residues")
        print(f"  Median: {np.median(lengths):.1f} residues")

    print("="*50)

if __name__ == "__main__":
    main()
