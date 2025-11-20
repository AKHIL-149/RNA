"""
Competition Submission Generator

Generates a properly formatted submission.csv file for the Stanford RNA 3D Folding
competition with 5 diverse predictions per sequence using template-based modeling.

Submission format: ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,...,x_5,y_5,z_5
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline
from src.tbm.ensemble import (
    predict_multi_template_weighted,
    quality_weighted_ensemble
)
from src.tbm.fragment_assembly import predict_long_sequence


def load_test_sequences(test_csv_path):
    """
    Load test sequences from the competition CSV file.

    Args:
        test_csv_path: Path to test_sequences.csv

    Returns:
        Dictionary mapping sequence ID to sequence string
    """
    df = pd.read_csv(test_csv_path)

    sequences = {}
    for _, row in df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        sequences[target_id] = sequence

    return sequences


def extract_resnames_from_sequence(sequence):
    """Extract residue names (A, U, G, C) from sequence."""
    return list(sequence)


def generate_diverse_predictions(pipeline, query_seq, n_predictions=5):
    """
    Generate 5 diverse predictions using different strategies.

    Strategy breakdown:
    1. Best single template (most reliable)
    2. Top-3 ensemble with identity weighting
    3. Top-5 ensemble with squared weighting (emphasizes quality)
    4. Quality-weighted diverse ensemble
    5. Top-7 ensemble or fragment assembly for long sequences

    Args:
        pipeline: TBMPipeline instance
        query_seq: Query RNA sequence
        n_predictions: Number of predictions (default 5)

    Returns:
        List of 5 coordinate arrays (each n_residues x 3)
    """
    predictions = []

    # Find templates once
    templates = pipeline.find_templates(query_seq, top_n=20)

    if len(templates) == 0:
        print(f"WARNING: No templates found! Generating null predictions.")
        # Return 5 identical null predictions
        null_coords = np.full((len(query_seq), 3), 0.0)
        return [null_coords.copy() for _ in range(5)]

    best_template_id = templates[0]['template_id']
    best_identity = templates[0]['identity']

    print(f"  Best template: {best_template_id} (identity: {best_identity:.3f})")

    # Strategy 1: Best single template (most reliable)
    pred1 = pipeline.predict_single_template(query_seq, best_template_id)
    predictions.append(pred1)
    print(f"  [1/5] Best single template: done")

    # Strategy 2: Top-3 ensemble with identity weighting
    try:
        pred2 = predict_multi_template_weighted(
            query_seq,
            templates,
            pipeline.train_coords,
            lambda seq, tid: pipeline.predict_single_template(seq, tid),
            top_n=3,
            weighting='identity',
            min_identity_for_ensemble=0.999  # Lower threshold to allow ensemble
        )
        if pred2 is not None:
            predictions.append(pred2)
            print(f"  [2/5] Top-3 identity ensemble: done")
        else:
            predictions.append(pred1.copy())
            print(f"  [2/5] Top-3 ensemble failed, using fallback")
    except Exception as e:
        print(f"  [2/5] Top-3 ensemble error: {e}, using fallback")
        predictions.append(pred1.copy())

    # Strategy 3: Top-5 ensemble with squared weighting (emphasizes quality)
    try:
        pred3 = predict_multi_template_weighted(
            query_seq,
            templates,
            pipeline.train_coords,
            lambda seq, tid: pipeline.predict_single_template(seq, tid),
            top_n=5,
            weighting='squared',
            min_identity_for_ensemble=0.999
        )
        if pred3 is not None:
            predictions.append(pred3)
            print(f"  [3/5] Top-5 squared ensemble: done")
        else:
            predictions.append(pred1.copy())
            print(f"  [3/5] Top-5 ensemble failed, using fallback")
    except Exception as e:
        print(f"  [3/5] Top-5 ensemble error: {e}, using fallback")
        predictions.append(pred1.copy())

    # Strategy 4: Quality-weighted diverse ensemble
    try:
        pred4 = quality_weighted_ensemble(
            query_seq,
            templates,
            pipeline.train_coords,
            pipeline.train_sequences,
            lambda seq, tid: pipeline.predict_single_template(seq, tid),
            top_n=5,
            min_identity_for_ensemble=0.999
        )
        if pred4 is not None:
            predictions.append(pred4)
            print(f"  [4/5] Quality-weighted ensemble: done")
        else:
            predictions.append(pred1.copy())
            print(f"  [4/5] Quality ensemble failed, using fallback")
    except Exception as e:
        print(f"  [4/5] Quality ensemble error: {e}, using fallback")
        predictions.append(pred1.copy())

    # Strategy 5: Adaptive - fragment assembly for long sequences, else Top-7 ensemble
    if len(query_seq) > 500:
        # Try fragment assembly for very long sequences
        try:
            pred5 = predict_long_sequence(
                query_seq,
                pipeline,
                fragment_size=150,
                overlap=30
            )
            if pred5 is not None and not np.isnan(pred5).all():
                predictions.append(pred5)
                print(f"  [5/5] Fragment assembly: done")
            else:
                predictions.append(pred1.copy())
                print(f"  [5/5] Fragment assembly failed, using fallback")
        except Exception as e:
            print(f"  [5/5] Fragment assembly error: {e}, using fallback")
            predictions.append(pred1.copy())
    else:
        # Top-7 ensemble for normal sequences
        try:
            pred5 = predict_multi_template_weighted(
                query_seq,
                templates,
                pipeline.train_coords,
                lambda seq, tid: pipeline.predict_single_template(seq, tid),
                top_n=7,
                weighting='identity',
                min_identity_for_ensemble=0.999
            )
            if pred5 is not None:
                predictions.append(pred5)
                print(f"  [5/5] Top-7 ensemble: done")
            else:
                predictions.append(pred1.copy())
                print(f"  [5/5] Top-7 ensemble failed, using fallback")
        except Exception as e:
            print(f"  [5/5] Top-7 ensemble error: {e}, using fallback")
            predictions.append(pred1.copy())

    # Ensure we have exactly 5 predictions
    while len(predictions) < 5:
        predictions.append(pred1.copy())

    return predictions[:5]


def predictions_to_submission_format(target_id, sequence, predictions):
    """
    Convert predictions to competition submission format.

    Args:
        target_id: Sequence ID (e.g., 'R1107')
        sequence: RNA sequence string
        predictions: List of 5 coordinate arrays (each n_residues x 3)

    Returns:
        List of dictionaries for DataFrame rows
    """
    rows = []
    resnames = list(sequence)

    for resid, resname in enumerate(resnames, start=1):
        row = {
            'ID': f'{target_id}_{resid}',
            'resname': resname,
            'resid': resid
        }

        # Add 5 predictions (x, y, z for each)
        for pred_idx, pred_coords in enumerate(predictions, start=1):
            if resid - 1 < len(pred_coords):
                coords = pred_coords[resid - 1]

                # Handle NaN or invalid coordinates
                if np.isnan(coords).any():
                    x, y, z = 0.0, 0.0, 0.0
                else:
                    x, y, z = coords[0], coords[1], coords[2]
            else:
                x, y, z = 0.0, 0.0, 0.0

            row[f'x_{pred_idx}'] = x
            row[f'y_{pred_idx}'] = y
            row[f'z_{pred_idx}'] = z

        rows.append(row)

    return rows


def generate_submission(
    train_coords_path,
    train_sequences_path,
    test_sequences_path,
    output_path='submission.csv'
):
    """
    Generate complete competition submission file.

    Args:
        train_coords_path: Path to train_coords_dict.pkl
        train_sequences_path: Path to train_sequences_dict.pkl
        test_sequences_path: Path to test_sequences.csv
        output_path: Output submission CSV path
    """
    print("=" * 80)
    print("Stanford RNA 3D Folding - Competition Submission Generator")
    print("=" * 80)

    # Load training data
    print("\n[1/4] Loading training data...")
    with open(train_coords_path, 'rb') as f:
        train_coords = pickle.load(f)
    with open(train_sequences_path, 'rb') as f:
        train_sequences = pickle.load(f)

    print(f"  Loaded {len(train_coords)} template structures")
    print(f"  Loaded {len(train_sequences)} template sequences")

    # Load test sequences
    print("\n[2/4] Loading test sequences...")
    test_sequences = load_test_sequences(test_sequences_path)
    print(f"  Loaded {len(test_sequences)} test sequences")

    for seq_id in sorted(test_sequences.keys()):
        print(f"    {seq_id}: {len(test_sequences[seq_id])} nt")

    # Initialize pipeline
    print("\n[3/4] Initializing TBM pipeline...")
    pipeline = TBMPipeline(train_coords, train_sequences)
    print("  Pipeline ready")

    # Generate predictions for all test sequences
    print("\n[4/4] Generating predictions...")
    all_rows = []

    for seq_id in sorted(test_sequences.keys()):
        query_seq = test_sequences[seq_id]
        print(f"\n{seq_id} ({len(query_seq)} nt):")

        # Generate 5 diverse predictions
        predictions = generate_diverse_predictions(pipeline, query_seq, n_predictions=5)

        # Convert to submission format
        rows = predictions_to_submission_format(seq_id, query_seq, predictions)
        all_rows.extend(rows)

    # Create DataFrame and save
    print("\n" + "=" * 80)
    print("Creating submission file...")

    columns = [
        'ID', 'resname', 'resid',
        'x_1', 'y_1', 'z_1',
        'x_2', 'y_2', 'z_2',
        'x_3', 'y_3', 'z_3',
        'x_4', 'y_4', 'z_4',
        'x_5', 'y_5', 'z_5'
    ]

    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Submission saved to: {output_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Sequences: {len(test_sequences)}")
    print(f"  Format: {len(columns)} columns × {len(df)} rows")

    # Validation
    print("\nValidation:")
    print(f"  ✓ Column count: {len(df.columns)} (expected 18)")
    print(f"  ✓ Column names match format: {list(df.columns) == columns}")
    print(f"  ✓ No missing values in ID/resname/resid: {df[['ID', 'resname', 'resid']].isna().sum().sum() == 0}")

    print("\n" + "=" * 80)
    print("SUBMISSION READY FOR UPLOAD!")
    print("=" * 80)

    return df


def main():
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent.parent
    train_coords_path = base_dir / 'data' / 'train_coords_dict.pkl'
    train_sequences_path = base_dir / 'data' / 'train_sequences_dict.pkl'
    test_sequences_path = base_dir / 'stanford-rna-3d-folding' / 'test_sequences.csv'
    output_path = base_dir / 'submission.csv'

    # Verify files exist
    if not train_coords_path.exists():
        print(f"ERROR: {train_coords_path} not found!")
        sys.exit(1)
    if not train_sequences_path.exists():
        print(f"ERROR: {train_sequences_path} not found!")
        sys.exit(1)
    if not test_sequences_path.exists():
        print(f"ERROR: {test_sequences_path} not found!")
        sys.exit(1)

    # Generate submission
    df = generate_submission(
        train_coords_path,
        train_sequences_path,
        test_sequences_path,
        output_path
    )

    # Display sample
    print("\nFirst 5 rows of submission:")
    print(df.head())

    print("\nLast 5 rows of submission:")
    print(df.tail())


if __name__ == '__main__':
    main()
