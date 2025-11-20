"""
TM-Score Evaluation Module

This module provides functions to calculate TM-scores for RNA structure predictions
using the US-align tool.

TM-score is the primary metric for the competition (ranges 0-1, higher is better).
"""

import subprocess
import tempfile
import os
import numpy as np
from pathlib import Path


def write_coords_to_pdb(coords, sequence, output_path, chain_id='A'):
    """
    Write coordinates to PDB format file.

    Args:
        coords (np.ndarray): Coordinates array, shape (n, 3)
        sequence (str): RNA sequence
        output_path (str): Path to output PDB file
        chain_id (str): Chain identifier
    """
    with open(output_path, 'w') as f:
        for i, (coord, base) in enumerate(zip(coords, sequence), start=1):
            if np.isnan(coord).any():
                continue

            # Write C1' atom for each nucleotide
            line = (
                f"ATOM  {i:5d}  C1'  {base:>3} {chain_id}{i:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           C\n"
            )
            f.write(line)
        f.write("END\n")


def calculate_tm_score(pred_coords, true_coords, pred_seq, true_seq, usalign_path='USalign/USalign'):
    """
    Calculate TM-score between predicted and true structures using US-align.

    Args:
        pred_coords (np.ndarray): Predicted coordinates, shape (n, 3)
        true_coords (np.ndarray): True coordinates, shape (m, 3)
        pred_seq (str): Predicted sequence
        true_seq (str): True sequence
        usalign_path (str): Path to USalign executable

    Returns:
        dict: {
            'tm_score': float,
            'rmsd': float,
            'aligned_length': int
        }
    """
    # Create temporary PDB files
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_pdb = os.path.join(tmpdir, 'pred.pdb')
        true_pdb = os.path.join(tmpdir, 'true.pdb')

        # Write PDB files
        write_coords_to_pdb(pred_coords, pred_seq, pred_pdb)
        write_coords_to_pdb(true_coords, true_seq, true_pdb)

        # Run US-align
        try:
            result = subprocess.run(
                [usalign_path, pred_pdb, true_pdb],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse output
            tm_score = None
            rmsd = None
            aligned_length = None

            for line in result.stdout.split('\n'):
                if 'TM-score=' in line and 'Chain_1' in line:
                    # Extract TM-score from line like:
                    # TM-score= 0.12345 (if normalized by length of Chain_1)
                    parts = line.split('TM-score=')[1].split()
                    tm_score = float(parts[0])
                elif 'RMSD=' in line:
                    parts = line.split('RMSD=')[1].split(',')[0].strip()
                    rmsd = float(parts)
                elif 'Aligned length=' in line:
                    parts = line.split('Aligned length=')[1].split(',')[0].strip()
                    aligned_length = int(parts)

            return {
                'tm_score': tm_score,
                'rmsd': rmsd,
                'aligned_length': aligned_length,
                'output': result.stdout
            }

        except subprocess.TimeoutExpired:
            return {
                'tm_score': None,
                'rmsd': None,
                'aligned_length': None,
                'error': 'US-align timeout'
            }
        except Exception as e:
            return {
                'tm_score': None,
                'rmsd': None,
                'aligned_length': None,
                'error': str(e)
            }


def calculate_tm_score_from_pdbs(pred_pdb_path, true_pdb_path, usalign_path='USalign/USalign'):
    """
    Calculate TM-score between two PDB files using US-align.

    Args:
        pred_pdb_path (str): Path to predicted structure PDB
        true_pdb_path (str): Path to true structure PDB
        usalign_path (str): Path to USalign executable

    Returns:
        dict: TM-score results
    """
    try:
        result = subprocess.run(
            [usalign_path, pred_pdb_path, true_pdb_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Parse output
        tm_score = None
        rmsd = None
        aligned_length = None

        for line in result.stdout.split('\n'):
            if 'TM-score=' in line and 'Chain_1' in line:
                parts = line.split('TM-score=')[1].split()
                tm_score = float(parts[0])
            elif 'RMSD=' in line:
                parts = line.split('RMSD=')[1].split(',')[0].strip()
                rmsd = float(parts)
            elif 'Aligned length=' in line:
                parts = line.split('Aligned length=')[1].split(',')[0].strip()
                aligned_length = int(parts)

        return {
            'tm_score': tm_score,
            'rmsd': rmsd,
            'aligned_length': aligned_length,
            'output': result.stdout
        }

    except Exception as e:
        return {
            'tm_score': None,
            'rmsd': None,
            'aligned_length': None,
            'error': str(e)
        }


def evaluate_predictions(predictions_list, true_coords, pred_seq, true_seq, usalign_path='USalign/USalign'):
    """
    Evaluate multiple predictions and return best TM-score.

    The competition uses best-of-5 predictions.

    Args:
        predictions_list (list): List of predicted coordinate arrays
        true_coords (np.ndarray): True coordinates
        pred_seq (str): Predicted sequence
        true_seq (str): True sequence
        usalign_path (str): Path to USalign executable

    Returns:
        dict: {
            'best_tm_score': float,
            'best_prediction_idx': int,
            'all_scores': list of TM-scores
        }
    """
    all_scores = []

    for i, pred_coords in enumerate(predictions_list):
        result = calculate_tm_score(
            pred_coords,
            true_coords,
            pred_seq,
            true_seq,
            usalign_path
        )
        all_scores.append(result['tm_score'])

    best_idx = np.argmax(all_scores) if all_scores else 0
    best_score = all_scores[best_idx] if all_scores else None

    return {
        'best_tm_score': best_score,
        'best_prediction_idx': best_idx,
        'all_scores': all_scores
    }


def batch_evaluate(predictions_dict, true_coords_dict, sequences_dict, usalign_path='USalign/USalign'):
    """
    Evaluate predictions for multiple targets.

    Args:
        predictions_dict (dict): Maps target_id to list of predictions
        true_coords_dict (dict): Maps target_id to true coordinates
        sequences_dict (dict): Maps target_id to sequence
        usalign_path (str): Path to USalign executable

    Returns:
        dict: Evaluation results for each target
    """
    results = {}

    for target_id in predictions_dict.keys():
        if target_id not in true_coords_dict:
            continue

        predictions = predictions_dict[target_id]
        true_coords = true_coords_dict[target_id]
        sequence = sequences_dict.get(target_id, 'N' * len(true_coords))

        eval_result = evaluate_predictions(
            predictions,
            true_coords,
            sequence,
            sequence,
            usalign_path
        )

        results[target_id] = eval_result

    return results


if __name__ == "__main__":
    print("TM-Score Evaluation Module")
    print("\nUsage:")
    print("""
    # Evaluate single prediction
    result = calculate_tm_score(pred_coords, true_coords, pred_seq, true_seq)
    print(f"TM-score: {result['tm_score']:.3f}")

    # Evaluate multiple predictions (best-of-5)
    results = evaluate_predictions(predictions_list, true_coords, seq, seq)
    print(f"Best TM-score: {results['best_tm_score']:.3f}")
    """)
