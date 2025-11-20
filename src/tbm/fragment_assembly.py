"""
Fragment-Based Assembly for Long RNA Sequences

Handles prediction of very long RNA sequences (>500nt) by breaking them into
overlapping fragments, predicting each fragment independently, and stitching
them together.
"""

import numpy as np
from typing import Dict, List, Tuple
from .similarity import align_sequences, get_alignment_mapping
from .adaptation import transfer_coordinates


def predict_fragment(
    fragment_seq: str,
    templates: List[Dict],
    train_coords: Dict,
    train_sequences: Dict,
    predict_func
) -> np.ndarray:
    """
    Predict structure for a single fragment.
    
    Args:
        fragment_seq: Fragment RNA sequence
        templates: List of available templates
        train_coords: Dictionary of template coordinates
        train_sequences: Dictionary of template sequences
        predict_func: Function to predict from template
    
    Returns:
        Fragment coordinates (n, 3)
    """
    if len(templates) == 0:
        # No templates, return NaN
        return np.full((len(fragment_seq), 3), np.nan)
    
    # Use best template
    best_template_id = templates[0]['template_id']
    
    try:
        coords = predict_func(fragment_seq, best_template_id)
        return coords
    except:
        return np.full((len(fragment_seq), 3), np.nan)


def create_overlapping_fragments(
    sequence: str,
    fragment_size: int = 100,
    overlap: int = 20
) -> List[Tuple[int, int, str]]:
    """
    Break sequence into overlapping fragments.
    
    Args:
        sequence: Full RNA sequence
        fragment_size: Size of each fragment
        overlap: Overlap between adjacent fragments
    
    Returns:
        List of (start_pos, end_pos, fragment_seq) tuples
    """
    fragments = []
    seq_len = len(sequence)
    
    if seq_len <= fragment_size:
        # Sequence is short enough, no fragmentation needed
        return [(0, seq_len, sequence)]
    
    step = fragment_size - overlap
    pos = 0
    
    while pos < seq_len:
        end_pos = min(pos + fragment_size, seq_len)
        fragment_seq = sequence[pos:end_pos]
        fragments.append((pos, end_pos, fragment_seq))
        
        if end_pos >= seq_len:
            break
        
        pos += step
    
    return fragments


def stitch_fragments(
    fragments: List[Tuple[int, int, np.ndarray]],
    full_length: int,
    overlap: int = 20
) -> np.ndarray:
    """
    Stitch fragment predictions into full structure.
    
    Uses weighted averaging in overlap regions based on distance from fragment edges.
    
    Args:
        fragments: List of (start_pos, end_pos, coords) tuples
        full_length: Length of full sequence
        overlap: Overlap size between fragments
    
    Returns:
        Full structure coordinates (n, 3)
    """
    result = np.full((full_length, 3), np.nan)
    weights = np.zeros(full_length)
    
    for start_pos, end_pos, frag_coords in fragments:
        # Skip if prediction failed
        if frag_coords is None or len(frag_coords) == 0:
            continue

        frag_len = end_pos - start_pos

        # Calculate position-dependent weights (highest in center, taper at edges)
        frag_weights = np.ones(frag_len)
        
        if overlap > 0:
            # Taper at start (if not first fragment)
            if start_pos > 0:
                taper_len = min(overlap, frag_len)
                taper = np.linspace(0, 1, taper_len)
                frag_weights[:taper_len] = taper
            
            # Taper at end (if not last fragment)
            if end_pos < full_length:
                taper_len = min(overlap, frag_len)
                taper = np.linspace(1, 0, taper_len)
                frag_weights[-taper_len:] = taper
        
        # Add weighted coords to result
        for i, global_pos in enumerate(range(start_pos, end_pos)):
            if not np.isnan(frag_coords[i]).any():
                if np.isnan(result[global_pos]).all():
                    # First assignment
                    result[global_pos] = frag_coords[i] * frag_weights[i]
                    weights[global_pos] = frag_weights[i]
                else:
                    # Accumulate weighted average
                    result[global_pos] += frag_coords[i] * frag_weights[i]
                    weights[global_pos] += frag_weights[i]
    
    # Normalize by weights
    for i in range(full_length):
        if weights[i] > 0:
            result[i] /= weights[i]
    
    return result


def predict_long_sequence(
    query_seq: str,
    find_templates_func,
    train_coords: Dict,
    train_sequences: Dict,
    predict_func,
    fragment_size: int = 100,
    overlap: int = 20
) -> np.ndarray:
    """
    Predict structure for long RNA sequence using fragment assembly.
    
    Args:
        query_seq: Query RNA sequence
        find_templates_func: Function to find templates for a sequence
        train_coords: Dictionary of template coordinates
        train_sequences: Dictionary of template sequences
        predict_func: Function to predict from template
        fragment_size: Size of each fragment (default 100nt)
        overlap: Overlap between fragments (default 20nt)
    
    Returns:
        Predicted coordinates (n, 3)
    """
    # Create fragments
    fragments = create_overlapping_fragments(query_seq, fragment_size, overlap)
    
    print(f"  Fragment assembly: {len(fragments)} fragments")
    
    # Predict each fragment
    predicted_fragments = []
    
    for i, (start_pos, end_pos, frag_seq) in enumerate(fragments):
        # Find templates for this fragment
        templates = find_templates_func(frag_seq, top_n=5, min_identity=0.5)
        
        # Predict fragment
        frag_coords = predict_fragment(
            frag_seq, templates, train_coords, train_sequences, predict_func
        )
        
        predicted_fragments.append((start_pos, end_pos, frag_coords))
        
        if (i + 1) % 5 == 0 or i == len(fragments) - 1:
            print(f"    Predicted {i+1}/{len(fragments)} fragments")
    
    # Stitch fragments together
    result = stitch_fragments(predicted_fragments, len(query_seq), overlap)
    
    return result


if __name__ == "__main__":
    print("Fragment Assembly Module")
    print("\nExample usage:")
    print("""
    from src.tbm.fragment_assembly import predict_long_sequence
    
    # Predict long sequence (e.g., 720nt)
    coords = predict_long_sequence(
        query_seq,
        pipeline.find_templates,
        train_coords,
        train_sequences,
        lambda seq, tid: pipeline.predict_single_template(seq, tid),
        fragment_size=100,
        overlap=20
    )
    """)
