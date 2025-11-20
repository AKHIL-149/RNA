"""
MSA-Based Template Search

The competition provides pre-computed Multiple Sequence Alignments (MSA) which
dramatically speeds up template search. This module leverages those MSAs.

This is likely how the winning team achieved fast template discovery.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def parse_a3m_file(a3m_path):
    """
    Parse A3M format MSA file.

    A3M format:
    >sequence_id
    SEQUENCE
    >sequence_id
    SEQUENCE
    ...

    Args:
        a3m_path (str): Path to A3M file

    Returns:
        list: List of (sequence_id, sequence) tuples
    """
    sequences = []
    current_id = None
    current_seq = []

    with open(a3m_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                # Start new sequence
                current_id = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_id is not None:
            sequences.append((current_id, ''.join(current_seq)))

    return sequences


def calculate_msa_identity(query_aligned, template_aligned):
    """
    Calculate sequence identity from aligned sequences in MSA.

    Args:
        query_aligned (str): Query sequence (with gaps as '-' or lowercase)
        template_aligned (str): Template sequence (with gaps)

    Returns:
        float: Sequence identity (0-1)
    """
    if len(query_aligned) != len(template_aligned):
        raise ValueError("Aligned sequences must have same length")

    matches = 0
    valid_positions = 0

    for q, t in zip(query_aligned, template_aligned):
        # Skip gaps
        if q == '-' or t == '-' or q.islower() or t.islower():
            continue

        valid_positions += 1
        if q == t:
            matches += 1

    if valid_positions == 0:
        return 0.0

    return matches / valid_positions


def find_templates_from_msa(msa_path, top_n=10, min_identity=0.3):
    """
    Find best templates from MSA file.

    Args:
        msa_path (str): Path to MSA file (.a3m format)
        top_n (int): Number of top templates to return
        min_identity (float): Minimum identity threshold

    Returns:
        list: List of (template_id, identity) tuples
    """
    # Parse MSA
    sequences = parse_a3m_file(msa_path)

    if len(sequences) == 0:
        return []

    # First sequence is the query
    query_id, query_seq = sequences[0]
    query_aligned = query_seq

    # Calculate identity for all templates
    templates = []
    for template_id, template_seq in sequences[1:]:
        identity = calculate_msa_identity(query_aligned, template_seq)

        if identity >= min_identity:
            templates.append({
                'template_id': template_id,
                'identity': identity,
                'aligned_query': query_aligned,
                'aligned_template': template_seq
            })

    # Sort by identity
    templates.sort(key=lambda x: x['identity'], reverse=True)

    return templates[:top_n]


def extract_alignment_from_msa(query_aligned, template_aligned):
    """
    Extract position mapping from MSA alignment.

    Args:
        query_aligned (str): Aligned query sequence
        template_aligned (str): Aligned template sequence

    Returns:
        dict: Mapping from query position to template position
    """
    mapping = {}
    query_pos = 0
    template_pos = 0

    for q_char, t_char in zip(query_aligned, template_aligned):
        # Uppercase = match position, lowercase/gap = insertion/deletion
        q_is_residue = q_char.isupper() and q_char != '-'
        t_is_residue = t_char.isupper() and t_char != '-'

        if q_is_residue and t_is_residue:
            mapping[query_pos] = template_pos

        if q_is_residue:
            query_pos += 1
        if t_is_residue:
            template_pos += 1

    return mapping


def get_msa_path_for_target(target_id, msa_dir):
    """
    Get MSA file path for a given target ID.

    Args:
        target_id (str): Target identifier (e.g., 'R1107')
        msa_dir (str or Path): Directory containing MSA files

    Returns:
        Path: Path to MSA file, or None if not found
    """
    msa_dir = Path(msa_dir)

    # Try different possible filenames
    possible_names = [
        f"{target_id}.a3m",
        f"{target_id}.msa",
        f"{target_id}.sto",
    ]

    for name in possible_names:
        msa_path = msa_dir / name
        if msa_path.exists():
            return msa_path

    return None


class MSATemplateSearch:
    """
    Fast template search using pre-computed MSAs.
    """

    def __init__(self, msa_dir, train_coords_dict, train_sequences_dict):
        """
        Initialize MSA-based search.

        Args:
            msa_dir (str): Directory containing MSA files
            train_coords_dict (dict): Training coordinates
            train_sequences_dict (dict): Training sequences
        """
        self.msa_dir = Path(msa_dir)
        self.train_coords = train_coords_dict
        self.train_sequences = train_sequences_dict

        print(f"Initialized MSA search with {len(self.train_coords)} templates")
        print(f"MSA directory: {self.msa_dir}")

    def find_templates(self, target_id, top_n=10, min_identity=0.3):
        """
        Find templates using MSA for given target.

        Args:
            target_id (str): Target identifier
            top_n (int): Number of templates to return
            min_identity (float): Minimum identity threshold

        Returns:
            list: List of template match dictionaries
        """
        # Find MSA file
        msa_path = get_msa_path_for_target(target_id, self.msa_dir)

        if msa_path is None:
            print(f"Warning: No MSA found for {target_id}")
            return []

        # Parse MSA and find templates
        templates = find_templates_from_msa(msa_path, top_n=top_n, min_identity=min_identity)

        # Filter to only templates we have coordinates for
        valid_templates = []
        for template in templates:
            template_id = template['template_id']

            # MSA might use different ID format, try to match
            if template_id in self.train_coords:
                valid_templates.append(template)
            else:
                # Try alternative formats
                alt_id = template_id.split('_')[0] if '_' in template_id else template_id
                if alt_id in self.train_coords:
                    template['template_id'] = alt_id
                    valid_templates.append(template)

        return valid_templates[:top_n]


if __name__ == "__main__":
    print("MSA-Based Template Search Module")
    print("\nUsage:")
    print("""
    # Initialize search
    searcher = MSATemplateSearch(msa_dir, train_coords, train_sequences)

    # Find templates for a target
    templates = searcher.find_templates('R1107', top_n=10)

    # Access results
    for match in templates:
        print(f"{match['template_id']}: {match['identity']:.1%}")
    """)
