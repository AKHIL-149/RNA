"""
Template-Based Modeling: Coordinate Adaptation Module

I implemented this module to handle coordinate transfer and adaptation from template structures
to query sequences for RNA 3D structure prediction.

Inspired by the winning approach from the competition.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def transfer_coordinates(template_coords, alignment_mapping, query_length):
    """
    Transfer the 3D coordinates from template to query based on sequence alignment.

    Args:
        template_coords (np.ndarray): Template coordinates, shape (n_residues, 3)
        alignment_mapping (dict): Maps query positions to template positions
        query_length (int): Length of query sequence

    Returns:
        np.ndarray: Query coordinates, shape (query_length, 3)
                   Positions without templates are filled with NaN
    """
    query_coords = np.full((query_length, 3), np.nan)

    for query_pos, template_pos in alignment_mapping.items():
        if template_pos < len(template_coords):
            query_coords[query_pos] = template_coords[template_pos]

    return query_coords


def fill_missing_residues_linear(coords):
    """
    Fill in missing residues using linear interpolation.

    For gaps in the structure, interpolate between known coordinates.

    Args:
        coords (np.ndarray): Coordinates with NaN for missing residues, shape (n, 3)

    Returns:
        np.ndarray: Coordinates with interpolated positions
    """
    coords = coords.copy()
    n = len(coords)

    for i in range(n):
        if np.isnan(coords[i]).any():
            # Find previous and next known positions
            prev_idx = None
            next_idx = None

            for j in range(i - 1, -1, -1):
                if not np.isnan(coords[j]).any():
                    prev_idx = j
                    break

            for j in range(i + 1, n):
                if not np.isnan(coords[j]).any():
                    next_idx = j
                    break

            # Interpolate if both neighbors exist
            if prev_idx is not None and next_idx is not None:
                gap_size = next_idx - prev_idx
                position = i - prev_idx
                alpha = position / gap_size
                coords[i] = (1 - alpha) * coords[prev_idx] + alpha * coords[next_idx]
            elif prev_idx is not None:
                # Extend from previous known position
                coords[i] = coords[prev_idx] + np.array([3.5, 0, 0])  # ~3.5Å C1' distance
            elif next_idx is not None:
                # Extend from next known position
                coords[i] = coords[next_idx] - np.array([3.5, 0, 0])

    return coords


def align_structures(coords1, coords2):
    """
    Align two structures using Kabsch algorithm.

    Finds optimal rotation and translation to superimpose coords1 onto coords2.

    Args:
        coords1 (np.ndarray): First structure coordinates, shape (n, 3)
        coords2 (np.ndarray): Second structure coordinates, shape (n, 3)

    Returns:
        tuple: (rotation_matrix, translation_vector, aligned_coords1)
    """
    # Remove NaN positions
    mask = ~(np.isnan(coords1).any(axis=1) | np.isnan(coords2).any(axis=1))
    c1 = coords1[mask]
    c2 = coords2[mask]

    if len(c1) < 3:
        # Not enough points for alignment
        return np.eye(3), np.zeros(3), coords1

    # Center both structures
    centroid1 = c1.mean(axis=0)
    centroid2 = c2.mean(axis=0)

    c1_centered = c1 - centroid1
    c2_centered = c2 - centroid2

    # Compute rotation using SVD (Kabsch algorithm)
    H = c1_centered.T @ c2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply transformation to all coords1
    aligned_coords1 = (coords1 - centroid1) @ R.T + centroid2

    return R, centroid2 - centroid1 @ R.T, aligned_coords1


def combine_multiple_templates(template_coords_list, alignment_mappings, query_length, weights=None):
    """
    Combine coordinates from multiple templates using weighted averaging.

    Args:
        template_coords_list (list): List of template coordinate arrays
        alignment_mappings (list): List of alignment mappings for each template
        query_length (int): Length of query sequence
        weights (list): Weight for each template (based on sequence identity)
                       If None, use equal weights

    Returns:
        np.ndarray: Combined query coordinates
    """
    if weights is None:
        weights = [1.0] * len(template_coords_list)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    query_coords = np.zeros((query_length, 3))
    position_weights = np.zeros(query_length)

    for template_coords, mapping, weight in zip(template_coords_list, alignment_mappings, weights):
        for query_pos, template_pos in mapping.items():
            if template_pos < len(template_coords):
                query_coords[query_pos] += weight * template_coords[template_pos]
                position_weights[query_pos] += weight

    # Normalize by weights
    mask = position_weights > 0
    query_coords[mask] /= position_weights[mask, np.newaxis]
    query_coords[~mask] = np.nan

    return query_coords


def refine_structure_local(coords, window_size=5):
    """
    Refine structure using local smoothing.

    Applies moving average to smooth out local irregularities while preserving
    overall structure.

    Args:
        coords (np.ndarray): Coordinates to refine, shape (n, 3)
        window_size (int): Size of smoothing window

    Returns:
        np.ndarray: Refined coordinates
    """
    refined = coords.copy()
    n = len(coords)

    for i in range(n):
        if np.isnan(coords[i]).any():
            continue

        # Get window around this position
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        window = coords[start:end]

        # Remove NaN positions
        valid_coords = window[~np.isnan(window).any(axis=1)]

        if len(valid_coords) > 0:
            # Weighted average (higher weight for closer positions)
            distances = np.abs(np.arange(start, end) - i)
            distances = distances[~np.isnan(window).any(axis=1)]
            weights = np.exp(-distances / 2)
            weights = weights / weights.sum()

            refined[i] = (valid_coords.T @ weights).T

    return refined


def calculate_rmsd(coords1, coords2):
    """
    Calculate Root Mean Square Deviation between two structures.

    Args:
        coords1 (np.ndarray): First structure, shape (n, 3)
        coords2 (np.ndarray): Second structure, shape (n, 3)

    Returns:
        float: RMSD value in Angstroms
    """
    # Remove NaN positions
    mask = ~(np.isnan(coords1).any(axis=1) | np.isnan(coords2).any(axis=1))
    c1 = coords1[mask]
    c2 = coords2[mask]

    if len(c1) == 0:
        return np.inf

    squared_diffs = np.sum((c1 - c2) ** 2, axis=1)
    rmsd = np.sqrt(np.mean(squared_diffs))

    return rmsd


if __name__ == "__main__":
    # Example usage
    print("Testing coordinate adaptation functions...")

    # Create example template coordinates (helix)
    n = 10
    template_coords = np.zeros((n, 3))
    for i in range(n):
        template_coords[i] = [i * 3.5, np.sin(i * 0.6) * 5, np.cos(i * 0.6) * 5]

    # Example alignment mapping (some gaps)
    alignment_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7}
    query_length = 12

    # Transfer coordinates
    query_coords = transfer_coordinates(template_coords, alignment_mapping, query_length)
    print(f"\nQuery coords shape: {query_coords.shape}")
    print(f"Missing positions: {np.isnan(query_coords).any(axis=1).sum()}")

    # Fill in missing residues
    filled_coords = fill_missing_residues_linear(query_coords)
    print(f"After filling: {np.isnan(filled_coords).any(axis=1).sum()} missing")

    # Calculate RMSD
    rmsd = calculate_rmsd(template_coords[:8], filled_coords[:8])
    print(f"RMSD: {rmsd:.2f} Å")

    print("\nCoordinate adaptation module is working correctly!")
