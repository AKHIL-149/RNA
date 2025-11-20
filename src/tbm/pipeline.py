"""
Template-Based Modeling: Main Pipeline

This module orchestrates the complete TBM workflow for RNA 3D structure prediction.

Workflow:
1. Find similar templates from training set
2. Align query to templates
3. Transfer and adapt coordinates
4. Generate multiple predictions
5. Select best prediction

Inspired by the winning approach from the competition.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .similarity import (
    calculate_sequence_similarity,
    get_alignment_mapping
)
from .adaptation import (
    transfer_coordinates,
    fill_missing_residues_linear,
    combine_multiple_templates,
    refine_structure_local
)
from .fast_search import FastTemplateSearch


class TBMPipeline:
    """
    Template-Based Modeling pipeline for RNA structure prediction.
    """

    def __init__(self, train_coords_dict, train_sequences_dict=None):
        """
        Initialize TBM pipeline.

        Args:
            train_coords_dict (dict): Dictionary mapping template IDs to coordinates
            train_sequences_dict (dict): Dictionary mapping template IDs to sequences
        """
        self.train_coords = train_coords_dict
        self.train_sequences = train_sequences_dict or {}

        print(f"Initialized TBM pipeline with {len(self.train_coords)} templates")

        # Initialize fast search if sequences available
        if self.train_sequences:
            self.fast_search = FastTemplateSearch(self.train_sequences, k=6)
        else:
            self.fast_search = None

    def find_templates(self, query_seq, top_n=10, min_identity=0.3):
        """
        Find the best matching templates for query sequence.

        Uses fast k-mer search first, then refines with alignment.

        Args:
            query_seq (str): Query RNA sequence
            top_n (int): Number of templates to return
            min_identity (float): Minimum sequence identity threshold

        Returns:
            list: List of template match dictionaries
        """
        if not self.train_sequences:
            raise ValueError("Training sequences not provided. Cannot compute similarity.")

        # Step 1: Fast k-mer based screening (get top 50 candidates)
        if self.fast_search:
            candidates = self.fast_search.find_similar_templates(
                query_seq,
                top_n=min(50, len(self.train_sequences)),
                min_similarity=0.1  # Low threshold for initial screening
            )
            candidate_ids = [template_id for template_id, _ in candidates]
        else:
            candidate_ids = list(self.train_sequences.keys())

        # Step 2: Refine with full alignment on candidates
        similarities = []

        for template_id in candidate_ids:
            if template_id not in self.train_coords:
                continue

            template_seq = self.train_sequences[template_id]
            result = calculate_sequence_similarity(query_seq, template_seq)

            if result['identity'] >= min_identity:
                similarities.append({
                    'template_id': template_id,
                    'identity': result['identity'],
                    'score': result['score'],
                    'aligned_query': result['aligned_query'],
                    'aligned_template': result['aligned_template']
                })

        # Sort by identity
        similarities.sort(key=lambda x: x['identity'], reverse=True)

        return similarities[:top_n]

    def predict_single_template(self, query_seq, template_id, template_seq=None):
        """
        Predict structure using a single template.

        Args:
            query_seq (str): Query RNA sequence
            template_id (str): Template ID to use
            template_seq (str): Template sequence (if not in self.train_sequences)

        Returns:
            np.ndarray: Predicted coordinates, shape (query_length, 3)
        """
        # Get template sequence
        if template_seq is None:
            if template_id not in self.train_sequences:
                raise ValueError(f"Template sequence not found for {template_id}")
            template_seq = self.train_sequences[template_id]

        # Get template coordinates
        if template_id not in self.train_coords:
            raise ValueError(f"Template coordinates not found for {template_id}")
        template_coords = self.train_coords[template_id]

        # Align sequences
        alignment = calculate_sequence_similarity(query_seq, template_seq)
        mapping = get_alignment_mapping(
            alignment['aligned_query'],
            alignment['aligned_template']
        )

        # Transfer coordinates
        query_coords = transfer_coordinates(
            template_coords,
            mapping,
            len(query_seq)
        )

        # Fill in missing residues
        query_coords = fill_missing_residues_linear(query_coords)

        return query_coords

    def predict_multi_template(self, query_seq, template_matches, use_top_n=5):
        """
        Predict structure using multiple templates with weighted averaging.

        Args:
            query_seq (str): Query RNA sequence
            template_matches (list): List of template match dictionaries
            use_top_n (int): Number of top templates to use

        Returns:
            np.ndarray: Predicted coordinates
        """
        # Use top N templates
        top_templates = template_matches[:use_top_n]

        if len(top_templates) == 0:
            raise ValueError("No templates provided")

        if len(top_templates) == 1:
            # Single template case
            return self.predict_single_template(
                query_seq,
                top_templates[0]['template_id']
            )

        # Multiple templates - combine with weighted averaging
        template_coords_list = []
        alignment_mappings = []
        weights = []

        for match in top_templates:
            template_id = match['template_id']
            template_coords = self.train_coords[template_id]

            mapping = get_alignment_mapping(
                match['aligned_query'],
                match['aligned_template']
            )

            template_coords_list.append(template_coords)
            alignment_mappings.append(mapping)
            weights.append(match['identity'])

        # Combine coordinates
        query_coords = combine_multiple_templates(
            template_coords_list,
            alignment_mappings,
            len(query_seq),
            weights
        )

        # Fill in missing residues
        query_coords = fill_missing_residues_linear(query_coords)

        # Refine structure
        query_coords = refine_structure_local(query_coords)

        return query_coords

    def predict(self, query_seq, n_predictions=5, strategy='multi_template'):
        """
        Generate multiple predictions for a query sequence.

        The competition requires best-of-5 predictions, so this generates
        multiple diverse predictions.

        Args:
            query_seq (str): Query RNA sequence
            n_predictions (int): Number of predictions to generate (default 5)
            strategy (str): Prediction strategy ('single_template' or 'multi_template')

        Returns:
            list: List of coordinate arrays, one per prediction
        """
        # Find templates
        template_matches = self.find_templates(query_seq, top_n=20)

        if len(template_matches) == 0:
            raise ValueError(f"No templates found for query sequence")

        predictions = []

        if strategy == 'single_template':
            # Generate predictions using single best templates
            for i, match in enumerate(template_matches[:n_predictions]):
                coords = self.predict_single_template(
                    query_seq,
                    match['template_id']
                )
                predictions.append(coords)

        elif strategy == 'multi_template':
            # Generate predictions using different combinations of templates
            # Strategy: Use top-k templates with varying k
            ks = [3, 5, 7, 10, 15]
            for k in ks[:n_predictions]:
                coords = self.predict_multi_template(
                    query_seq,
                    template_matches,
                    use_top_n=min(k, len(template_matches))
                )
                predictions.append(coords)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return predictions

    @classmethod
    def load_from_files(cls, coords_pickle_path, sequences_dict=None):
        """
        Load TBM pipeline from saved training data files.

        Args:
            coords_pickle_path (str): Path to pickled coordinates dictionary
            sequences_dict (dict): Optional sequences dictionary

        Returns:
            TBMPipeline: Initialized pipeline
        """
        print(f"Loading coordinates from {coords_pickle_path}...")
        with open(coords_pickle_path, 'rb') as f:
            train_coords = pickle.load(f)

        return cls(train_coords, sequences_dict)


def save_prediction_to_pdb(coords, sequence, output_path, target_id="PRED"):
    """
    Save predicted coordinates to PDB format compatible with US-align.

    Args:
        coords (np.ndarray): Coordinates array, shape (n, 3)
        sequence (str): RNA sequence
        output_path (str): Path to output PDB file
        target_id (str): Target identifier
    """
    # Map nucleotides to standard PDB residue names
    base_map = {'A': '  A', 'U': '  U', 'G': '  G', 'C': '  C'}

    with open(output_path, 'w') as f:
        f.write(f"HEADER    RNA STRUCTURE PREDICTION    {target_id}\n")

        atom_num = 1
        for i, (coord, base) in enumerate(zip(coords, sequence), start=1):
            if np.isnan(coord).any():
                # Skip missing residues
                continue

            # Get proper residue name
            res_name = base_map.get(base, '  N')

            # PDB format: ATOM line
            # C1' atom (representative for nucleotide)
            atom_line = (
                f"ATOM  {atom_num:5d}  C1' {res_name} A{i:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           C\n"
            )
            f.write(atom_line)
            atom_num += 1

        f.write("END\n")


if __name__ == "__main__":
    print("TBM Pipeline module")
    print("Usage:")
    print("""
    # Load pipeline
    pipeline = TBMPipeline.load_from_files('data/train_coords_dict.pkl', sequences_dict)

    # Generate predictions
    predictions = pipeline.predict(query_sequence, n_predictions=5)

    # Save to PDB
    save_prediction_to_pdb(predictions[0], query_sequence, 'output.pdb')
    """)
