"""
Evaluation Module

Provides functions for evaluating RNA structure predictions using TM-score
and other structural metrics.
"""

from .tm_score import (
    calculate_tm_score,
    calculate_tm_score_from_pdbs,
    evaluate_predictions,
    batch_evaluate,
    write_coords_to_pdb
)

from .rmsd_calculator import (
    calculate_rmsd,
    calculate_tm_score_approx,
    evaluate_structure_similarity,
    align_to_principal_axes
)

__all__ = [
    'calculate_tm_score',
    'calculate_tm_score_from_pdbs',
    'evaluate_predictions',
    'batch_evaluate',
    'write_coords_to_pdb',
    'calculate_rmsd',
    'calculate_tm_score_approx',
    'evaluate_structure_similarity',
    'align_to_principal_axes',
]
