"""
Multi-Template Ensemble Methods

Combines multiple template predictions using weighted averaging and
consensus structure building.
"""

import numpy as np
from typing import List, Dict, Optional
from .adaptation import align_structures


def predict_multi_template_weighted(
    query_seq: str,
    templates: List[Dict],
    train_coords: Dict,
    predict_func,
    top_n: int = 3,
    weighting: str = 'identity',
    min_identity_for_ensemble: float = 0.99
) -> np.ndarray:
    """
    Predict structure using weighted ensemble of multiple templates.

    Smart ensemble: Only uses multiple templates when the best template
    is not near-perfect. If best template has >95% identity, uses it alone.

    Args:
        query_seq: Query RNA sequence
        templates: List of template dictionaries with 'template_id' and 'identity'
        train_coords: Dictionary of template coordinates
        predict_func: Function to predict from single template
        top_n: Number of top templates to use
        weighting: Weighting scheme ('identity', 'squared', 'uniform')
        min_identity_for_ensemble: If best template >= this, skip ensemble

    Returns:
        Ensemble prediction coordinates (n, 3)
    """
    if len(templates) == 0:
        return None

    # Select top N templates
    selected_templates = templates[:min(top_n, len(templates))]

    if len(selected_templates) == 1:
        # Single template, no ensemble needed
        return predict_func(query_seq, selected_templates[0]['template_id'])

    # Smart ensemble: If best template is near-perfect, use it alone
    best_identity = selected_templates[0]['identity']
    if best_identity >= min_identity_for_ensemble:
        # Best template is excellent, don't dilute with lower quality
        return predict_func(query_seq, selected_templates[0]['template_id'])

    # Generate predictions from each template
    predictions = []
    weights = []

    for template in selected_templates:
        template_id = template['template_id']

        # Skip if template not in training data
        if template_id not in train_coords:
            continue

        # Get prediction
        pred = predict_func(query_seq, template_id)

        if pred is not None and len(pred) == len(query_seq):
            predictions.append(pred)

            # Calculate weight based on template quality
            if weighting == 'identity':
                weight = template['identity']
            elif weighting == 'squared':
                weight = template['identity'] ** 2  # Emphasize high-quality templates
            elif weighting == 'uniform':
                weight = 1.0
            else:
                weight = template['identity']

            weights.append(weight)

    if len(predictions) == 0:
        return None

    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    # Align all predictions to first prediction (reference)
    reference = predictions[0]
    aligned_predictions = [reference]

    for pred in predictions[1:]:
        # Align to reference using Kabsch
        try:
            _, _, aligned = align_structures(pred, reference)
            aligned_predictions.append(aligned)
        except:
            # If alignment fails, use original
            aligned_predictions.append(pred)

    # Weighted average
    ensemble = np.zeros_like(reference)

    for pred, weight in zip(aligned_predictions, weights):
        ensemble += weight * pred

    return ensemble


def select_diverse_templates(
    templates: List[Dict],
    train_sequences: Dict,
    max_templates: int = 5,
    min_diversity: float = 0.1
) -> List[Dict]:
    """
    Select diverse set of templates to avoid redundancy.

    Args:
        templates: List of template dictionaries
        train_sequences: Dictionary of template sequences
        max_templates: Maximum number of templates to select
        min_diversity: Minimum sequence diversity (1 - identity)

    Returns:
        List of diverse templates
    """
    if len(templates) <= max_templates:
        return templates

    selected = [templates[0]]  # Start with best template

    for template in templates[1:]:
        if len(selected) >= max_templates:
            break

        template_id = template['template_id']
        if template_id not in train_sequences:
            continue

        template_seq = train_sequences[template_id]

        # Check diversity against already selected
        is_diverse = True
        for sel_template in selected:
            sel_id = sel_template['template_id']
            if sel_id not in train_sequences:
                continue

            sel_seq = train_sequences[sel_id]

            # Calculate sequence identity
            if len(template_seq) == len(sel_seq):
                matches = sum(1 for a, b in zip(template_seq, sel_seq) if a == b)
                identity = matches / len(template_seq)

                if (1 - identity) < min_diversity:
                    is_diverse = False
                    break

        if is_diverse:
            selected.append(template)

    return selected


def consensus_structure(
    predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
    threshold: float = 0.7
) -> np.ndarray:
    """
    Build consensus structure from multiple predictions.

    Uses weighted voting to determine final coordinates for each residue.

    Args:
        predictions: List of coordinate predictions
        weights: Optional weights for each prediction
        threshold: Confidence threshold for accepting coordinates

    Returns:
        Consensus coordinates (n, 3)
    """
    if len(predictions) == 0:
        return None

    if len(predictions) == 1:
        return predictions[0]

    # Default uniform weights
    if weights is None:
        weights = [1.0] * len(predictions)

    weights = np.array(weights) / np.sum(weights)

    # Initialize consensus
    n_residues = len(predictions[0])
    consensus = np.full((n_residues, 3), np.nan)

    for i in range(n_residues):
        # Collect coordinates for this residue from all predictions
        coords = []
        valid_weights = []

        for pred, weight in zip(predictions, weights):
            if i < len(pred) and not np.isnan(pred[i]).any():
                coords.append(pred[i])
                valid_weights.append(weight)

        if len(coords) == 0:
            continue

        # Normalize weights for valid predictions
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / np.sum(valid_weights)

        # Check if consensus is confident enough
        consensus_confidence = np.sum(valid_weights)

        if consensus_confidence >= threshold:
            # Weighted average of coordinates
            consensus[i] = np.average(coords, axis=0, weights=valid_weights)

    return consensus


def quality_weighted_ensemble(
    query_seq: str,
    templates: List[Dict],
    train_coords: Dict,
    train_sequences: Dict,
    predict_func,
    top_n: int = 5,
    min_identity_for_ensemble: float = 0.99
) -> np.ndarray:
    """
    Advanced ensemble using quality scoring and diversity selection.

    Smart ensemble: Only uses multiple templates when beneficial.

    Args:
        query_seq: Query RNA sequence
        templates: List of template dictionaries
        train_coords: Dictionary of template coordinates
        train_sequences: Dictionary of template sequences
        predict_func: Function to predict from single template
        top_n: Number of templates to use
        min_identity_for_ensemble: If best template >= this, skip ensemble

    Returns:
        Quality-weighted ensemble prediction
    """
    if len(templates) == 0:
        return None

    # Smart ensemble: If best template is near-perfect, use it alone
    if templates[0]['identity'] >= min_identity_for_ensemble:
        return predict_func(query_seq, templates[0]['template_id'])

    # Select diverse templates
    diverse_templates = select_diverse_templates(
        templates,
        train_sequences,
        max_templates=top_n,
        min_diversity=0.1
    )

    # Score template quality
    scored_templates = []
    for template in diverse_templates:
        template_id = template['template_id']

        if template_id not in train_coords:
            continue

        coords = train_coords[template_id]

        # Quality score based on coordinate completeness
        valid_ratio = (~np.isnan(coords).any(axis=1)).sum() / len(coords)

        # Combined score: identity × quality
        combined_score = template['identity'] * valid_ratio

        scored_templates.append({
            **template,
            'quality_score': combined_score
        })

    # Sort by quality score
    scored_templates.sort(key=lambda x: x['quality_score'], reverse=True)

    # Generate weighted ensemble
    predictions = []
    weights = []

    for template in scored_templates[:top_n]:
        template_id = template['template_id']
        pred = predict_func(query_seq, template_id)

        if pred is not None:
            predictions.append(pred)
            weights.append(template['quality_score'])

    if len(predictions) == 0:
        return None

    # Align predictions
    reference = predictions[0]
    aligned = [reference]

    for pred in predictions[1:]:
        try:
            _, _, aligned_pred = align_structures(pred, reference)
            aligned.append(aligned_pred)
        except:
            aligned.append(pred)

    # Build consensus
    consensus = consensus_structure(aligned, weights, threshold=0.5)

    return consensus


if __name__ == "__main__":
    print("Multi-Template Ensemble Module")
    print("\nExample usage:")
    print("""
    from src.tbm.ensemble import predict_multi_template_weighted

    # Get templates
    templates = pipeline.find_templates(query_seq, top_n=10)

    # Ensemble prediction
    ensemble_pred = predict_multi_template_weighted(
        query_seq,
        templates,
        train_coords,
        lambda seq, tid: pipeline.predict_single_template(seq, tid),
        top_n=3,
        weighting='squared'
    )
    """)
