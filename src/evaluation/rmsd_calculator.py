"""
RMSD-based Structure Evaluation

Alternative to TM-score using BioPython and direct coordinate comparison.
Provides multiple structural similarity metrics.
"""

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from typing import Dict, Tuple, Optional


def align_to_principal_axes(coords: np.ndarray) -> np.ndarray:
    """
    Align coordinates to principal axes for canonical orientation.

    This creates a standard reference frame by:
    1. Centering at origin
    2. Computing principal components (eigenvectors of covariance)
    3. Rotating to align with coordinate axes

    This helps when comparing structures in different reference frames.

    Args:
        coords: Coordinates to align (n, 3)

    Returns:
        Aligned coordinates in canonical orientation
    """
    # Remove NaN coords for PCA
    valid_mask = ~np.isnan(coords).any(axis=1)
    valid_coords = coords[valid_mask]

    if len(valid_coords) < 3:
        # Not enough points for PCA
        return coords

    # Center at origin
    centered = valid_coords - valid_coords.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Get principal components (eigenvectors)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

        # Rotate to principal axes
        aligned_valid = centered @ eigenvectors

        # Apply to all coords (including NaN)
        result = coords.copy()
        result[valid_mask] = aligned_valid

        return result

    except np.linalg.LinAlgError:
        # PCA failed, return centered coords
        result = coords.copy()
        result[valid_mask] = centered
        return result


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray,
                   align: bool = True, use_principal_axes: bool = False) -> Dict[str, float]:
    """
    Calculate RMSD between two coordinate sets.

    Args:
        coords1: First coordinate array (n, 3)
        coords2: Second coordinate array (n, 3)
        align: Whether to perform optimal superimposition first
        use_principal_axes: Whether to align to principal axes before comparison

    Returns:
        Dictionary with RMSD and alignment metrics
    """
    # Remove NaN and invalid coordinates
    valid_mask = ~(np.isnan(coords1).any(axis=1) | np.isnan(coords2).any(axis=1))

    # Also filter out extremely large coordinates (likely corrupted data)
    max_coord = 1e6  # Reasonable upper bound for biological structures
    valid_mask &= (np.abs(coords1).max(axis=1) < max_coord)
    valid_mask &= (np.abs(coords2).max(axis=1) < max_coord)

    valid_coords1 = coords1[valid_mask]
    valid_coords2 = coords2[valid_mask]

    if len(valid_coords1) == 0:
        return {
            'rmsd': None,
            'aligned_rmsd': None,
            'num_aligned': 0,
            'coverage': 0.0
        }

    # Calculate coverage
    coverage = len(valid_coords1) / max(len(coords1), len(coords2))

    # Align to principal axes for canonical orientation (optional)
    if use_principal_axes:
        # Create full coord arrays with valid coords
        full_coords1 = np.full((len(coords1), 3), np.nan)
        full_coords2 = np.full((len(coords2), 3), np.nan)
        full_coords1[valid_mask] = valid_coords1
        full_coords2[valid_mask] = valid_coords2

        # Align to principal axes
        aligned1 = align_to_principal_axes(full_coords1)
        aligned2 = align_to_principal_axes(full_coords2)

        # Extract valid coords
        centered1 = aligned1[valid_mask]
        centered2 = aligned2[valid_mask]
    else:
        # Center coordinates at origin (critical for proper alignment)
        com1 = valid_coords1.mean(axis=0)
        com2 = valid_coords2.mean(axis=0)
        centered1 = valid_coords1 - com1
        centered2 = valid_coords2 - com2

    # Direct RMSD (without alignment, but centered)
    diff = centered1 - centered2
    direct_rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    # Aligned RMSD (with optimal superimposition)
    aligned_rmsd = direct_rmsd
    if align:
        try:
            sup = SVDSuperimposer()
            # SVDSuperimposer works best with centered coordinates
            sup.set(centered1, centered2)
            sup.run()
            aligned_rmsd = sup.get_rms()
        except Exception as e:
            print(f"Warning: SVD alignment failed: {e}, using centered RMSD")
            # Fall back to centered RMSD
            aligned_rmsd = direct_rmsd

    return {
        'rmsd': direct_rmsd,
        'aligned_rmsd': aligned_rmsd,
        'num_aligned': len(valid_coords1),
        'coverage': coverage
    }


def calculate_tm_score_approx(coords1: np.ndarray, coords2: np.ndarray,
                               d0: Optional[float] = None,
                               use_principal_axes: bool = False) -> Dict[str, float]:
    """
    Approximate TM-score calculation.

    TM-score formula: TM = (1/L) * Σ 1/(1 + (di/d0)^2)
    where L is the length of the target structure,
    di is the distance between aligned residues,
    and d0 is a length-dependent scaling factor.

    Args:
        coords1: Predicted coordinates (n, 3)
        coords2: True coordinates (n, 3)
        d0: Scaling factor (default: 1.24 * (L-15)^(1/3) - 1.8 for L>15)
        use_principal_axes: Whether to align to principal axes before comparison

    Returns:
        Dictionary with TM-score and related metrics
    """
    # Remove NaN and invalid coordinates
    valid_mask = ~(np.isnan(coords1).any(axis=1) | np.isnan(coords2).any(axis=1))

    # Filter out extremely large coordinates (likely corrupted)
    max_coord = 1e6
    valid_mask &= (np.abs(coords1).max(axis=1) < max_coord)
    valid_mask &= (np.abs(coords2).max(axis=1) < max_coord)

    valid_coords1 = coords1[valid_mask]
    valid_coords2 = coords2[valid_mask]

    if len(valid_coords1) == 0:
        return {
            'tm_score': None,
            'tm_score_normalized': None,
            'd0': None,
            'num_aligned': 0
        }

    L = len(coords2)  # Length of target structure

    # Calculate d0 (Zhang & Skolnick formula)
    if d0 is None:
        if L > 15:
            d0 = 1.24 * ((L - 15) ** (1.0/3.0)) - 1.8
        else:
            d0 = 0.5

    # Align to principal axes for canonical orientation (optional)
    if use_principal_axes:
        # Create full coord arrays with valid coords
        full_coords1 = np.full((len(coords1), 3), np.nan)
        full_coords2 = np.full((len(coords2), 3), np.nan)
        full_coords1[valid_mask] = valid_coords1
        full_coords2[valid_mask] = valid_coords2

        # Align to principal axes
        aligned1 = align_to_principal_axes(full_coords1)
        aligned2 = align_to_principal_axes(full_coords2)

        # Extract valid coords
        centered1 = aligned1[valid_mask]
        centered2 = aligned2[valid_mask]
    else:
        # Center coordinates at origin before alignment
        com1 = valid_coords1.mean(axis=0)
        com2 = valid_coords2.mean(axis=0)
        centered1 = valid_coords1 - com1
        centered2 = valid_coords2 - com2

    # Align structures
    try:
        sup = SVDSuperimposer()
        sup.set(centered1, centered2)
        sup.run()
        rot, tran = sup.get_rotran()
        aligned_coords1 = np.dot(centered1, rot) + tran
    except Exception as e:
        print(f"Warning: SVD alignment failed: {e}, using centered coords")
        aligned_coords1 = centered1

    # Calculate distances (compare to centered coords)
    distances = np.sqrt(np.sum((aligned_coords1 - centered2)**2, axis=1))

    # TM-score calculation
    tm_terms = 1.0 / (1.0 + (distances / d0)**2)
    tm_score = np.sum(tm_terms) / L  # Normalized by target length
    tm_score_aligned = np.sum(tm_terms) / len(valid_coords1)  # Normalized by aligned length

    return {
        'tm_score': tm_score,
        'tm_score_normalized': tm_score_aligned,
        'd0': d0,
        'num_aligned': len(valid_coords1),
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'max_distance': np.max(distances)
    }


def evaluate_structure_similarity(pred_coords: np.ndarray,
                                  true_coords: np.ndarray,
                                  use_principal_axes: bool = False) -> Dict[str, any]:
    """
    Comprehensive structural similarity evaluation.

    Args:
        pred_coords: Predicted coordinates (n, 3)
        true_coords: True coordinates (n, 3)
        use_principal_axes: Whether to align to principal axes before comparison

    Returns:
        Dictionary with multiple similarity metrics
    """
    # Ensure same length (pad if needed)
    max_len = max(len(pred_coords), len(true_coords))

    if len(pred_coords) < max_len:
        padded_pred = np.full((max_len, 3), np.nan)
        padded_pred[:len(pred_coords)] = pred_coords
        pred_coords = padded_pred

    if len(true_coords) < max_len:
        padded_true = np.full((max_len, 3), np.nan)
        padded_true[:len(true_coords)] = true_coords
        true_coords = padded_true

    # Calculate all metrics
    rmsd_metrics = calculate_rmsd(pred_coords, true_coords, align=True,
                                   use_principal_axes=use_principal_axes)
    tm_metrics = calculate_tm_score_approx(pred_coords, true_coords,
                                           use_principal_axes=use_principal_axes)

    # Combine results
    results = {
        **rmsd_metrics,
        **tm_metrics,
        'length_pred': len(pred_coords),
        'length_true': len(true_coords)
    }

    return results


if __name__ == "__main__":
    print("RMSD Calculator Module")
    print("\nExample usage:")
    print("""
    from src.evaluation.rmsd_calculator import evaluate_structure_similarity
    import numpy as np

    # Load coordinates
    pred_coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    true_coords = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]])

    # Evaluate
    metrics = evaluate_structure_similarity(pred_coords, true_coords)
    print(f"RMSD: {metrics['aligned_rmsd']:.3f}")
    print(f"TM-score: {metrics['tm_score']:.3f}")
    """)
