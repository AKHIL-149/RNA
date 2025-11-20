"""
Template-Based Modeling: Sequence Similarity Module

I implemented this module to handle sequence similarity search and alignment functions
for RNA 3D structure prediction using template-based modeling.

Inspired by the competition winner's methodology, adapted for my implementation.
"""

import numpy as np
from Bio import pairwise2
from Bio.pairwise2 import format_alignment


def calculate_sequence_similarity(query_seq, template_seq):
    """
    Calculate how similar two sequences are between query and template sequences.

    Uses global alignment (Needleman-Wunsch algorithm) to find the best alignment
    and calculate identity percentage.

    Args:
        query_seq (str): Query RNA sequence (A, U, G, C)
        template_seq (str): Template RNA sequence (A, U, G, C)

    Returns:
        dict: {
            'identity': float (0-1),
            'alignment': alignment object,
            'score': alignment score
        }
    """
    # Global alignment with match=1, mismatch=-1, gap_open=-1, gap_extend=-0.5
    alignments = pairwise2.align.globalms(
        query_seq,
        template_seq,
        match=1,      # Match score
        mismatch=-1,  # Mismatch penalty
        open=-1,      # Gap open penalty
        extend=-0.5   # Gap extend penalty
    )

    if not alignments:
        return {
            'identity': 0.0,
            'alignment': None,
            'score': 0
        }

    # Take best alignment
    best_alignment = alignments[0]

    # Calculate identity
    aligned_query = best_alignment.seqA
    aligned_template = best_alignment.seqB

    matches = sum(1 for a, b in zip(aligned_query, aligned_template) if a == b and a != '-')
    total_positions = max(len(query_seq), len(template_seq))
    identity = matches / total_positions if total_positions > 0 else 0.0

    return {
        'identity': identity,
        'alignment': best_alignment,
        'score': best_alignment.score,
        'aligned_query': aligned_query,
        'aligned_template': aligned_template
    }


def find_best_templates(query_seq, template_coords_dict, top_n=5, min_identity=0.3):
    """
    Find the best matching templates from training set based on sequence similarity.

    Args:
        query_seq (str): Query RNA sequence
        template_coords_dict (dict): Dictionary mapping template IDs to coordinate arrays
        top_n (int): Number of top templates to return
        min_identity (float): Minimum sequence identity threshold (0-1)

    Returns:
        list: List of tuples (template_id, similarity_info) sorted by identity
    """
    similarities = []

    for template_id in template_coords_dict.keys():
        # Get template sequence (for now, we'll need to load this separately)
        # In practice, you'd have a separate dict mapping IDs to sequences
        # For now, we'll skip templates where we can't compute similarity
        # This will be enhanced when we integrate with the full training data
        pass

    # This is a placeholder - will be implemented when we have sequence data
    # For now, return empty list
    return []


def find_best_templates_from_msa(query_seq, msa_dir, top_n=5):
    """
    Find the best matching templates using pre-computed MSA (Multiple Sequence Alignments).

    This is the approach used by the winning team - leverage pre-computed MSAs
    from the competition dataset.

    Args:
        query_seq (str): Query RNA sequence
        msa_dir (str): Path to directory containing MSA files
        top_n (int): Number of top templates to return

    Returns:
        list: List of template IDs sorted by similarity
    """
    # TODO: Implement MSA-based template search
    # The competition provides MSA files which already contain sequence alignments
    # We can use these to quickly find similar sequences
    raise NotImplementedError("MSA-based search will be implemented next")


def align_sequences(seq1, seq2):
    """
    Perform pairwise sequence alignment.

    Args:
        seq1 (str): First sequence
        seq2 (str): Second sequence

    Returns:
        tuple: (aligned_seq1, aligned_seq2, identity_score)
    """
    result = calculate_sequence_similarity(seq1, seq2)

    return (
        result['aligned_query'],
        result['aligned_template'],
        result['identity']
    )


def get_alignment_mapping(aligned_query, aligned_template):
    """
    Create a mapping from query positions to template positions based on alignment.

    This is crucial for transferring coordinates from template to query.

    Args:
        aligned_query (str): Aligned query sequence (with gaps as '-')
        aligned_template (str): Aligned template sequence (with gaps as '-')

    Returns:
        dict: Mapping from query position (0-indexed) to template position (0-indexed)
              Only includes positions where both sequences have residues (no gaps)
    """
    mapping = {}
    query_pos = 0
    template_pos = 0

    for q_char, t_char in zip(aligned_query, aligned_template):
        # If both have residues (not gaps), create mapping
        if q_char != '-' and t_char != '-':
            mapping[query_pos] = template_pos

        # Increment positions
        if q_char != '-':
            query_pos += 1
        if t_char != '-':
            template_pos += 1

    return mapping


if __name__ == "__main__":
    # Example usage
    query = "GGGGAAAAAACCCC"
    template = "GGGGUUUUUUCCCC"

    result = calculate_sequence_similarity(query, template)
    print(f"Sequence Identity: {result['identity']:.2%}")
    print(f"Alignment Score: {result['score']}")

    if result['alignment']:
        print("\nAlignment:")
        print(f"Query:    {result['aligned_query']}")
        print(f"Template: {result['aligned_template']}")

    mapping = get_alignment_mapping(result['aligned_query'], result['aligned_template'])
    print(f"\nPosition mapping: {mapping}")
