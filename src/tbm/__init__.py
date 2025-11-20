"""
Template-Based Modeling (TBM) Module

Implementation of the 1st place winning approach for RNA 3D structure prediction.

Main components:
- similarity: Sequence similarity and template search
- adaptation: Coordinate transfer and adaptation
- pipeline: Complete TBM workflow
"""

from .similarity import (
    calculate_sequence_similarity,
    get_alignment_mapping,
    align_sequences
)

from .adaptation import (
    transfer_coordinates,
    fill_missing_residues_linear,
    combine_multiple_templates,
    refine_structure_local,
    calculate_rmsd
)

from .pipeline import (
    TBMPipeline,
    save_prediction_to_pdb
)

from .ensemble import (
    predict_multi_template_weighted,
    select_diverse_templates,
    consensus_structure,
    quality_weighted_ensemble
)

from .fragment_assembly import (
    predict_long_sequence,
    create_overlapping_fragments,
    stitch_fragments
)

__all__ = [
    # Similarity
    'calculate_sequence_similarity',
    'get_alignment_mapping',
    'align_sequences',
    # Adaptation
    'transfer_coordinates',
    'fill_missing_residues_linear',
    'combine_multiple_templates',
    'refine_structure_local',
    'calculate_rmsd',
    # Pipeline
    'TBMPipeline',
    'save_prediction_to_pdb',
    # Ensemble
    'predict_multi_template_weighted',
    'select_diverse_templates',
    'consensus_structure',
    'quality_weighted_ensemble',
    # Fragment Assembly
    'predict_long_sequence',
    'create_overlapping_fragments',
    'stitch_fragments',
]
