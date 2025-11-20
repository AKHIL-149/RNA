# 1st Place Winner - Detailed Code Analysis

**Notebook**: `rna-3d-folds-tbm-only-approach.ipynb`
**Team**: john
**Final Score**: Private LB 0.57773 | Public LB 0.60388

---

## Overview

This is a **pure Template-Based Modeling (TBM) approach** with NO deep learning models. The winner used algorithmic improvements and domain knowledge to beat fine-tuned transformer models.

---

## Code Architecture

### 1. Data Loading & Processing

```python
# Base datasets
train_seqs = pd.read_csv('train_sequences.csv')  # 844 sequences
train_labels = pd.read_csv('train_labels.csv')   # 137,095 coordinates

# Extended datasets (custom-generated from PDB CIF files)
train_seqs_v2 = pd.read_csv('rna_sequences.csv')      # 18,881 sequences
train_labels_v2 = pd.read_csv('rna_coordinates.csv')  # 10,135,546 coordinates (!!)
```

**Key Finding**: Winner created **custom dataset** from CIF files:
- 93 nucleotide variant mapping (modified bases)
- Disorder-aware coordinate extraction
- **10M+ coordinate records** (73× more than base dataset!)

### 2. Dataset Extension Strategy

```python
def extend_dataset(original_df, v2_df, key_columns, dataset_name):
    # Identifies unique records in v2 not in original
    # Combines: original + new_records_from_v2
    # Result: 18,946 training sequences (22× original!)
```

**Final Training Set**:
- Sequences: 844 → 18,946 (+2,146%)
- Coordinates: 137,095 → 10,147,442 (+7,302%)

**Critical Insight**: Majority of extra data has NO temporal_cutoff metadata!
- Only 844 original sequences have temporal info
- 18,102 additional sequences lack timestamp → potential leakage risk
- Winner likely filtered by PDB release dates manually

---

## Core Algorithm Components

### Component 1: Enhanced Sequence Similarity Search

**Function**: `find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5)`

#### Multi-Tier Filtering

```python
# Length-based filtering (adaptive)
len_ratio = abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq))

if len(query_seq) < 50 or len(train_seq) < 50:  # Short sequences
    threshold = 0.6  # More permissive
elif len(query_seq) > 1000 or len(train_seq) > 1000:  # Long sequences
    threshold = 0.2  # Stricter
else:  # Medium sequences
    threshold = 0.4  # Standard
```

**Strategy**: Adaptive tolerance based on sequence length

#### Composite Similarity Scoring

```python
composite_score = (
    0.4 * global_alignment_score +    # Global structure conservation
    0.3 * local_alignment_score +     # Local motif similarity
    0.2 * feature_similarity +        # Structural features
    0.1 * kmer_similarity             # Sequence motifs
)
```

**Alignment Parameters**:
- Match: +2.9
- Mismatch: -1
- Gap open: -10
- Gap extend: -0.5

**Note**: These are **highly optimized** for RNA (not default BioPython values)

#### Enhanced RNA Features (21 features total)

```python
features = [
    # 1. Nucleotide frequencies (4 features)
    freq_A, freq_U, freq_G, freq_C,

    # 2. Important dinucleotide frequencies (10 features)
    freq_AU, freq_UA, freq_GC, freq_CG, freq_GU, freq_UG,
    freq_AA, freq_UU, freq_GG, freq_CC,

    # 3. Structural indicators (4 features)
    gc_content, au_content, purine_content, pyrimidine_content,

    # 4. Complexity measures (2 features)
    length_normalized, entropy_normalized,

    # 5. Repetitive patterns (1 feature)
    repeat_content
]
```

**Innovation**: K-mer similarity for motif detection
```python
def _calculate_kmer_similarity(seq1, seq2, k=3):
    kmers1 = set(seq[i:i+k] for i in range(len(seq) - k + 1))
    kmers2 = set(seq[j:j+k] for j in range(len(seq) - k + 1))
    return jaccard_similarity(kmers1, kmers2)
```

#### Diversity-Based Clustering

```python
# For candidates ≥ 15: KMeans clustering
# For candidates < 15: Diversity-based selection
# Goal: Maximize diversity among top 5 templates
```

**Why This Matters**:
- TM-score metric uses best-of-5 predictions
- Diverse templates → better coverage of conformational space
- Reduces risk of all predictions being similar (and wrong)

---

### Component 2: Template Adaptation

**Function**: `adapt_template_to_query(query_seq, template_seq, template_coords, alignment)`

#### Step 1: Sequence Alignment

```python
alignments = pairwise2.align.globalms(
    query_seq, template_seq,
    match=2.9, mismatch=-1, gap_open=-10, gap_extend=-0.5
)
```

#### Step 2: Coordinate Transfer

```python
# Direct copy for aligned positions
if query_char != '-' and template_char != '-':
    query_coords[query_idx] = template_coords[template_idx]
```

#### Step 3: Improved Gap Filling

**Key Innovation**: Geometric backbone reconstruction

```python
backbone_distance = 5.9  # Typical C1'-C1' distance

if total_distance < expected_distance * 0.7:  # Compressed gap
    # Extend along realistic backbone with curvature
    direction = (next_coord - prev_coord) / distance
    for idx in gap:
        progress = (idx - prev_idx) / gap_size
        base_pos = prev_coord + direction * expected_distance * progress

        # Add realistic curvature
        perpendicular = cross(direction, [0, 0, 1])
        curve_amplitude = 2.0 * sin(progress * π)
        coords[idx] = base_pos + perpendicular * curve_amplitude

else:  # Normal gap
    # Linear interpolation
    for idx in gap:
        weight = (idx - prev_idx) / gap_size
        coords[idx] = (1 - weight) * prev_coord + weight * next_coord
```

**Geometric Principles**:
- Maintains C1'-C1' distance (~5.9Å)
- Adds sinusoidal perturbations for realism
- Extends terminal regions along backbone direction

---

### Component 3: Adaptive Refinement

**Function**: `adaptive_rna_constraints(coordinates, sequence, confidence)`

**Core Idea**: Constraint strength inversely proportional to template confidence

```python
constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))

# High confidence (0.8+): minimal constraints (preserve template)
# Medium confidence (0.5-0.8): moderate constraints
# Low confidence (<0.5): strong constraints
```

#### Constraint 1: Sequential Distance

```python
seq_min_dist = 5.5  # Minimum C1'-C1' distance
seq_max_dist = 6.5  # Maximum C1'-C1' distance

# Only adjust if outside range
if current_dist < seq_min_dist or current_dist > seq_max_dist:
    target_dist = (seq_min_dist + seq_max_dist) / 2
    adjustment = (target_dist - current_dist) * constraint_strength
    coords[i+1] = coords[i] + direction * (current_dist + adjustment)
```

#### Constraint 2: Steric Clash Prevention

```python
min_allowed_distance = 3.8  # Minimum distance between non-consecutive atoms

# Find severe clashes
dist_matrix = distance_matrix(coords, coords)
severe_clashes = where((dist_matrix < min_allowed_distance) & (dist_matrix > 0))

# Fix clashes
for i, j in severe_clashes:
    if abs(i - j) <= 1:  # Skip consecutive residues
        continue

    adjustment = (min_allowed_distance - current_dist) * constraint_strength
    coords[i] -= direction * (adjustment / 2)
    coords[j] += direction * (adjustment / 2)
```

#### Constraint 3: Light Base-Pairing (Low Confidence Only)

```python
if constraint_strength > 0.3:  # Only for low-confidence templates
    pairs = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

    for i in range(n_residues):
        complement = pairs.get(sequence[i])

        # Look for complementary bases within 3-20 residue window
        for j in range(i + 3, min(i + 20, n_residues)):
            if sequence[j] == complement:
                current_dist = norm(coords[i] - coords[j])

                # Only consider if distance suggests potential pairing (8-14Å)
                if 8.0 < current_dist < 14.0:
                    target_dist = 10.5  # Generic C1'-C1' base-pair distance
                    adjustment = (target_dist - current_dist) * constraint_strength * 0.3

                    # Gentle adjustment
                    coords[i] -= direction * (adjustment / 2)
                    coords[j] += direction * (adjustment / 2)
                    break  # One pair per base
```

**Key Design Decisions**:
1. **Sequential constraints always applied** (fundamental RNA geometry)
2. **Steric clash prevention always active** (physical validity)
3. **Base-pairing constraints only for low confidence** (avoid over-constraining good templates)

---

### Component 4: De Novo Structure Generation (Fallback)

**Function**: `generate_improved_rna_structure(sequence)`

Used when no good templates found or as additional predictions.

#### Step 1: Identify Potential Stems

```python
def identify_potential_stems(sequence):
    # Look for self-complementary segments
    # Minimum stem length: 3 base pairs

    for i in range(len(sequence) - min_stem_length):
        for j in range(i + min_stem_length + 3, len(sequence)):
            # Check if regions could form a stem
            if all([
                sequence[i+k] == complement[sequence[j+stem_len-k-1]]
                for k in range(stem_len)
            ]):
                stems.append((i, i+stem_len-1, j, j+stem_len-1))
```

#### Step 2: Assign Structure Types

```python
structure_types = ['single'] * len(sequence)

# Mark stem regions
for start1, end1, start2, end2 in stems:
    structure_types[start1:end1+1] = ['stem'] * (end1-start1+1)
    structure_types[start2:end2+1] = ['stem'] * (end2-start2+1)

# Mark loop regions (between paired regions)
# ... (regions connecting stems)
```

#### Step 3: Generate Coordinates Based on Structure Type

```python
for i in range(n_residues):
    if structure_types[i] == 'stem':
        # Helical geometry
        angle += angle_per_residue_helix (0.6 rad)
        coords[i] = [
            radius_helix * cos(angle),  # 10.0 Å
            radius_helix * sin(angle),
            z + rise_per_residue_helix  # 2.5 Å
        ]

    elif structure_types[i] == 'loop':
        # Loop geometry
        angle += angle_per_residue_loop (0.3 rad)
        z_shift = rise_per_residue_loop * sin(angle * 0.5)  # 1.5 Å
        coords[i] = [
            radius_loop * cos(angle),  # 15.0 Å
            radius_loop * sin(angle),
            z + z_shift
        ]

    else:
        # Single-stranded (random walk)
        jitter = np.random.normal(0, 1, 3) * 2.0
        coords[i] = coords[i-1] + jitter
```

**Parameters** (RNA-specific):
- Helix radius: 10.0 Å (A-form helix)
- Helix rise: 2.5 Å/residue
- Helix rotation: 0.6 rad/residue (~34°)
- Loop radius: 15.0 Å (larger, more flexible)
- Loop rise: 1.5 Å/residue

---

### Component 5: Main Prediction Pipeline

**Function**: `predict_rna_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_predictions=5)`

```python
def predict_rna_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_predictions=5):
    predictions = []

    # 1. Find top 5 similar sequences (with clustering)
    similar_seqs = find_similar_sequences(
        sequence, train_seqs_df, train_coords_dict, top_n=5
    )

    # 2. Adapt each template
    for template_id, template_seq, similarity_score, template_coords in similar_seqs:
        # Transfer coordinates
        adapted_coords = adapt_template_to_query(
            sequence, template_seq, template_coords
        )

        # Apply adaptive constraints (stronger for lower similarity)
        refined_coords = adaptive_rna_constraints(
            adapted_coords, sequence, confidence=similarity_score
        )

        # Add small randomness (less for better templates)
        random_scale = max(0.05, 0.8 - similarity_score)
        randomized_coords = refined_coords + np.random.normal(0, random_scale, shape)

        predictions.append(randomized_coords)

    # 3. Fill remaining predictions with de novo structures
    while len(predictions) < n_predictions:
        seed = hash(target_id) % 10000 + len(predictions) * 1000
        de_novo_coords = generate_rna_structure(sequence, seed=seed)

        # Apply strong constraints (low confidence)
        refined_de_novo = adaptive_rna_constraints(
            de_novo_coords, sequence, confidence=0.2
        )

        predictions.append(refined_de_novo)

    return predictions[:n_predictions]
```

**Strategy Breakdown**:
1. **Template-based predictions**: Use top 5 diverse templates
2. **Confidence-weighted randomization**: Better templates get less noise
3. **De novo fallback**: Generate structures when templates insufficient
4. **Diversity maximization**: Different seeds for de novo structures

---

## Performance Analysis

### Runtime

```
Processing 12 test sequences:
- Total runtime: 421.8 seconds (~35 seconds/sequence)
- Includes: similarity search, alignment, adaptation, refinement
```

**Note**: This is on Kaggle notebook (likely 2 CPU cores, no GPU)

### Computational Complexity

**Per Sequence**:
1. Similarity search: O(N × M) where N = train sequences (18,946), M = seq length
2. Alignment: O(L1 × L2) for each candidate (50 candidates × alignment cost)
3. Clustering: O(K × I × N) for K-means (50 candidates, 5 clusters)
4. Refinement: O(L²) for distance matrix (L = sequence length)

**Total**: Approximately O(N × M × K) per test sequence

---

## Key Innovations

### 1. Custom Dataset Generation

- Processed ALL PDB CIF files (not just competition data)
- 93 nucleotide variant mapping (modified bases: pseudouridine, methylated bases, etc.)
- Disorder-aware coordinate extraction (skips disordered residues)
- **Result**: 18,946 training templates (vs. 844 baseline)

### 2. Multi-Criteria Similarity

Not just sequence alignment! Combines:
- Global alignment (40%)
- Local alignment (30%)
- Feature similarity (20%)
- K-mer similarity (10%)

**Why**: Captures both sequence AND structural conservation

### 3. Diversity-Based Template Selection

- K-means clustering on 21-dimensional feature space
- Selects best representative from each cluster
- **Result**: 5 diverse templates instead of 5 similar ones

### 4. Adaptive Constraints

- High-confidence templates: minimal intervention (preserve structure)
- Low-confidence templates: strong constraints (enforce physics)
- **Philosophy**: "Trust good templates, guide bad ones"

### 5. Geometric Gap Filling

Not simple linear interpolation! Features:
- Maintains C1'-C1' distance (5.9Å)
- Adds realistic backbone curvature (sinusoidal)
- Extends terminals along backbone direction
- **Result**: Physically plausible structures

### 6. Confidence-Weighted Randomization

```python
random_scale = max(0.05, 0.8 - similarity_score)
```

- Best template (score 0.8): random_scale = 0.05 (minimal noise)
- Worst template (score 0.0): random_scale = 0.8 (high diversity)
- **Result**: Diverse ensemble without destroying good predictions

---

## Limitations & Failure Modes

### 1. Template Dependency

**Problem**: Fails for sequences with no homologs

**Evidence**:
- All test sequences found templates in 18,946-sequence database
- But new RNA folds (e.g., synthetic designs) would fail

**Mitigation**: De novo fallback (but likely low quality)

### 2. Sequence Length Sensitivity

**Problem**: Long sequences (>400nt) slow

**Evidence**:
- R1128 (238nt): 122.8 seconds elapsed for 6 sequences
- Complexity: O(N × M²) where M = sequence length

**Mitigation**: None in this implementation

### 3. No Ligand Awareness

**Problem**: Ignores ligands/ions

**Evidence**:
- PreQ1 riboswitch (R1117v2) treated same as apo RNA
- Apta-FRET (R1136) with ligands ignored

**Mitigation**: None

### 4. No Uncertainty Quantification

**Problem**: Which of 5 predictions is most reliable?

**Evidence**: All predictions treated equally

**Mitigation**: None (leaves confidence on table)

### 5. Potential Temporal Leakage

**Problem**: 18,102 additional sequences lack temporal_cutoff

**Risk**: May include structures released after test sequences

**Mitigation**: Likely manual PDB release date filtering (not shown in notebook)

---

## Why This Beat Deep Learning

### 1. Generalization

**TBM Advantage**: Templates are experimental structures
- **Public LB**: 0.60388 → **Private LB**: 0.57773 (-4.3% drop)
- Compare to 3rd place DL: 0.61253 → 0.54312 (-11.3% drop!)

**Reason**: Physical structures more robust than learned patterns

### 2. Data Efficiency

**TBM Advantage**: Uses structural database directly
- 18,946 templates from PDB (decades of experiments)
- No training required
- No overfitting to competition data

### 3. Interpretability

**TBM Advantage**: Every prediction traceable to template
- Can inspect which template was used
- Understand why prediction looks a certain way
- Debug failures by examining templates

### 4. Computational Efficiency

**TBM Advantage**: Runs on CPU
- No GPU required
- No model loading overhead
- 35 seconds/sequence acceptable for competition

### 5. Domain Knowledge Integration

**TBM Advantage**: Directly encodes RNA physics
- C1'-C1' distances (5.9Å)
- Base-pairing geometry (10.5Å)
- Helical parameters (10Å radius, 2.5Å rise)
- Steric clash prevention (3.8Å minimum)

**Deep Learning**: Must learn these from data (harder!)

---

## Lessons for Your Research

### What to Reproduce

****Must Reproduce**:
1. Dataset extension strategy (custom CIF processing)
2. Multi-criteria similarity scoring
3. Diversity-based template selection
4. Adaptive constraint system
5. Geometric gap filling

****Good to Have**:
1. De novo structure generation (fallback)
2. K-mer similarity metric
3. Enhanced RNA features

### What to Improve (Research Novelties)

🔬 **High Impact**:
1. **Uncertainty quantification**: Predict which of 5 is best
2. **Ligand awareness**: Condition on ligand presence
3. **Ensemble ranking**: Learn better selection function
4. **Physics-informed constraints**: Quantum mechanics force fields

🔬 **Medium Impact**:
1. **Long sequence optimization**: Chunking/hierarchical methods
2. **Conformational ensembles**: Multiple states from MD
3. **Transfer learning**: Protein→RNA geometric priors

🔬 **Lower Priority**:
1. **Runtime optimization**: Parallel processing, approximate NN
2. **Feature engineering**: Better structural descriptors

### Benchmarking Strategy

**Phase 1** (Week 1-2):
1. Reproduce winner's pipeline on test set
2. Measure TM-scores vs. validation labels
3. Analyze which sequences benefit from TBM

**Phase 2** (Week 3-4):
1. Ablation studies:
   - Remove diversity clustering → measure impact
   - Remove adaptive constraints → measure impact
   - Remove geometric gap filling → measure impact
2. Identify bottlenecks

**Phase 3** (Month 2):
1. Implement physics-informed selector
2. Compare to winner's approach
3. Measure improvement on held-out data

---

## Code Quality Assessment

### Strengths

**Well-structured functions (separation of concerns)
**Clear variable names
**Progress tracking (time estimates)
**Vectorized operations (NumPy)
**Modular design (easy to modify components)

### Weaknesses

**No docstrings (function documentation)
**No type hints
**No unit tests
**Hard-coded magic numbers (10.0, 5.9, 0.8, etc.)
**No logging (hard to debug)
**No configuration file (parameters scattered)

### Suggestions for Improvement

```python
# Instead of:
radius_helix = 10.0
rise_per_residue_helix = 2.5

# Use configuration:
RNA_PARAMS = {
    'helix_radius': 10.0,  # Å, A-form RNA helix
    'helix_rise': 2.5,     # Å per residue
    'helix_rotation': 0.6,  # radians per residue
    'c1_c1_distance': 5.9,  # Å, sequential backbone
    'base_pair_distance': 10.5,  # Å, Watson-Crick
    'min_clash_distance': 3.8,  # Å, van der Waals
}
```

---

## Reproducibility Checklist

### Required Files

- [ ] `rna_sequences.csv` (18,881 sequences) - **NOT PROVIDED**
- [ ] `rna_coordinates.csv` (10M+ coords) - **NOT PROVIDED**
- [ ] Train sequences (competition data) - **Available
- [ ] Train labels (competition data) - **Available
- [ ] Test sequences - **Available

### Missing Information

❓ **How were CIF files processed?**
- 93 nucleotide variant mapping (which variants?)
- Disorder-aware extraction (threshold?)
- PDB release date filtering (how?)

❓ **Hyperparameter tuning?**
- Alignment scoring (2.9, -1, -10, -0.5) - how chosen?
- Composite weights (0.4, 0.3, 0.2, 0.1) - optimized or intuition?
- Constraint strengths - grid search or expert knowledge?

❓ **Validation strategy?**
- How were parameters validated?
- Cross-validation on training set?
- Or just public leaderboard feedback?

---

## Next Steps

### Immediate (This Week)

1. **Extract RNA parameters** to configuration file
2. **Document functions** with docstrings
3. **Test on your local data** (18,946 training sequences if you can generate)
4. **Measure baseline performance**

### Short-term (2 Weeks)

1. **Reproduce CIF processing** pipeline
2. **Implement ablation studies**:
   - Disable diversity clustering
   - Disable adaptive constraints
   - Disable geometric gap filling
   - Measure impact of each component
3. **Profile code** for bottlenecks

### Medium-term (Month 2)

1. **Implement physics-informed ranking**
2. **Add uncertainty quantification**
3. **Test ligand-aware modifications**
4. **Compare to DRfold2 baseline**

---

## Summary

**Winner's Secret Sauce**:
1. **Massive custom dataset** (18,946 templates from PDB)
2. **Multi-criteria similarity** (not just sequence alignment)
3. **Diversity-based selection** (K-means clustering)
4. **Adaptive constraints** (confidence-weighted physics)
5. **Geometric gap filling** (realistic backbone reconstruction)
6. **Ensemble diversity** (randomization scaled by confidence)

**Why It Worked**:
- Leverages decades of experimental data (PDB)
- Encodes domain knowledge directly (RNA geometry)
- Robust to distribution shift (physical structures generalize)
- Computationally efficient (no GPU needed)

**Your Opportunity**:
- Improve physics constraints (QM force fields)
- Add uncertainty quantification (which prediction to trust?)
- Incorporate ligand information (holo vs. apo)
- Learn optimal ranking function (ML on physical features)

**Bottom Line**: This is a masterclass in **domain-specific algorithm design**. The winner didn't use the fanciest deep learning - they used **deep understanding of RNA biology** combined with **solid software engineering**.

---

**Next Update**: After reproducing this implementation and measuring baseline TM-scores
