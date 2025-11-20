#!/usr/bin/env python3
"""
Data Exploration Script - RNA 3D Folding Competition
Converted from notebooks/01_data_exploration.ipynb for direct execution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Set up paths
PROJECT_DIR = '/Users/akhil/Documents/GitHub/RNA'
DATA_DIR = f'{PROJECT_DIR}/stanford-rna-3d-folding'
FIGURES_DIR = f'{PROJECT_DIR}/figures'
RESULTS_DIR = f'{PROJECT_DIR}/results'

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("RNA 3D FOLDING - DATA EXPLORATION")
print("=" * 60)
print()

# ============================================================================
# 1. Load Test Sequences
# ============================================================================
print("[1/6] Loading test sequences...")

test_seqs = pd.read_csv(f'{DATA_DIR}/test_sequences.csv')
valid_labels = pd.read_csv(f'{DATA_DIR}/validation_labels.csv')

print(f"  Number of test sequences: {len(test_seqs)}")
print(f"  Number of validation label rows: {len(valid_labels)}")
print()

# ============================================================================
# 2. Sequence Length Analysis
# ============================================================================
print("[2/6] Analyzing sequence lengths...")

test_seqs['length'] = test_seqs['sequence'].str.len()

print("  Sequence length distribution:")
print(f"    Min: {test_seqs['length'].min()} nt")
print(f"    Max: {test_seqs['length'].max()} nt")
print(f"    Mean: {test_seqs['length'].mean():.1f} nt")
print(f"    Median: {test_seqs['length'].median():.1f} nt")
print()

# Categorize by length
test_seqs['length_category'] = pd.cut(
    test_seqs['length'],
    bins=[0, 50, 100, 200, 400, 1000],
    labels=['Very Short (<50)', 'Short (50-100)', 'Medium (100-200)',
            'Long (200-400)', 'Very Long (>400)']
)

print("  Length categories:")
for cat, count in test_seqs['length_category'].value_counts().sort_index().items():
    print(f"    {cat}: {count}")
print()

# Visualize length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(test_seqs['length'], bins=20, edgecolor='black')
axes[0].set_xlabel('Sequence Length (nt)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Sequence Lengths')
axes[0].axvline(test_seqs['length'].mean(), color='r', linestyle='--',
                label=f'Mean: {test_seqs["length"].mean():.0f}')
axes[0].legend()

test_seqs['length_category'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_xlabel('Length Category')
axes[1].set_ylabel('Count')
axes[1].set_title('Test Sequences by Length Category')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/test_sequence_lengths.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/test_sequence_lengths.png")
plt.close()

# ============================================================================
# 3. RNA Type Classification
# ============================================================================
print("[3/6] Classifying RNA types...")

def classify_rna_type(description):
    """Classify RNA based on description"""
    if pd.isna(description):
        return 'Unknown'

    desc_lower = description.lower()

    if 'riboswitch' in desc_lower:
        return 'Riboswitch'
    elif 'ribozyme' in desc_lower:
        return 'Ribozyme'
    elif 'trna' in desc_lower or 't-rna' in desc_lower:
        return 'tRNA'
    elif 'rrna' in desc_lower or 'ribosom' in desc_lower:
        return 'rRNA'
    elif 'aptamer' in desc_lower:
        return 'Aptamer'
    elif 'loop' in desc_lower or 'hairpin' in desc_lower:
        return 'Loop/Hairpin'
    elif 'helix' in desc_lower or 'stem' in desc_lower:
        return 'Helix/Stem'
    elif 'viral' in desc_lower or 'virus' in desc_lower or 'cov' in desc_lower:
        return 'Viral RNA'
    else:
        return 'Other'

test_seqs['rna_type'] = test_seqs['description'].apply(classify_rna_type)

print("  RNA Type Distribution:")
for rna_type, count in test_seqs['rna_type'].value_counts().items():
    print(f"    {rna_type}: {count} ({count/len(test_seqs)*100:.1f}%)")
print()

# Visualize RNA types
fig, ax = plt.subplots(figsize=(10, 6))
test_seqs['rna_type'].value_counts().plot(kind='barh', color='coral', ax=ax)
ax.set_xlabel('Count')
ax.set_ylabel('RNA Type')
ax.set_title('Test Sequences by RNA Type')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/test_sequence_types.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/test_sequence_types.png")
plt.close()

# ============================================================================
# 4. Sequence Composition Analysis
# ============================================================================
print("[4/6] Analyzing sequence composition...")

def calculate_composition(sequence):
    """Calculate nucleotide composition"""
    if pd.isna(sequence) or len(sequence) == 0:
        return {'A': 0, 'U': 0, 'G': 0, 'C': 0, 'GC': 0}

    length = len(sequence)
    return {
        'A': sequence.count('A') / length,
        'U': sequence.count('U') / length,
        'G': sequence.count('G') / length,
        'C': sequence.count('C') / length,
        'GC': (sequence.count('G') + sequence.count('C')) / length
    }

compositions = test_seqs['sequence'].apply(calculate_composition)
composition_df = pd.DataFrame(compositions.tolist())
test_seqs = pd.concat([test_seqs, composition_df], axis=1)

print("  Average nucleotide composition:")
print(f"    A: {composition_df['A'].mean():.3f}")
print(f"    U: {composition_df['U'].mean():.3f}")
print(f"    G: {composition_df['G'].mean():.3f}")
print(f"    C: {composition_df['C'].mean():.3f}")
print(f"    GC: {composition_df['GC'].mean():.3f}")
print()

# Visualize GC content distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(test_seqs['GC'], bins=15, edgecolor='black', color='lightgreen')
ax.set_xlabel('GC Content')
ax.set_ylabel('Count')
ax.set_title('Distribution of GC Content in Test Sequences')
ax.axvline(test_seqs['GC'].mean(), color='r', linestyle='--',
           label=f'Mean: {test_seqs["GC"].mean():.2f}')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/gc_content_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/gc_content_distribution.png")
plt.close()

# ============================================================================
# 5. Create Comprehensive Catalog
# ============================================================================
print("[5/6] Creating test sequences catalog...")

def extract_organism(description):
    """Extract organism from description"""
    if pd.isna(description):
        return 'Unknown'

    if 'human' in description.lower() or 'homo sapiens' in description.lower():
        return 'Human'
    elif 'virus' in description.lower() or 'viral' in description.lower():
        return 'Virus'
    elif 'bacteria' in description.lower() or 'e. coli' in description.lower():
        return 'Bacteria'
    elif 'synthetic' in description.lower():
        return 'Synthetic'
    else:
        return 'Other'

catalog = test_seqs[[
    'target_id', 'length', 'length_category', 'rna_type',
    'A', 'U', 'G', 'C', 'GC', 'temporal_cutoff', 'description'
]].copy()

catalog['organism'] = test_seqs['description'].apply(extract_organism)
catalog['has_ligand'] = test_seqs['description'].str.contains('ligand', case=False, na=False)

catalog.to_csv(f'{RESULTS_DIR}/test_sequences_catalog.csv', index=False)
print(f"  ✓ Saved: results/test_sequences_catalog.csv ({catalog.shape[0]} sequences)")
print()

# ============================================================================
# 6. Summary Statistics
# ============================================================================
print("[6/6] Generating summary...")
print()
print("=" * 60)
print("TEST SET SUMMARY")
print("=" * 60)
print(f"Total sequences: {len(test_seqs)}")
print()
print("Length statistics:")
print(f"  Min: {test_seqs['length'].min()} nt")
print(f"  Max: {test_seqs['length'].max()} nt")
print(f"  Mean: {test_seqs['length'].mean():.1f} nt")
print(f"  Median: {test_seqs['length'].median():.1f} nt")
print()
print("RNA Type breakdown:")
for rna_type, count in test_seqs['rna_type'].value_counts().items():
    print(f"  {rna_type}: {count} ({count/len(test_seqs)*100:.1f}%)")
print()
print("Composition:")
print(f"  GC content: {test_seqs['GC'].mean():.2f} ± {test_seqs['GC'].std():.2f}")
print()
print(f"Sequences with ligands: {catalog['has_ligand'].sum()}")
print()
print(f"Organism breakdown:")
for org, count in catalog['organism'].value_counts().items():
    print(f"  {org}: {count}")
print("=" * 60)
print()
print("✅ DATA EXPLORATION COMPLETE!")
print()
print("Next steps:")
print("  1. Extract TBM functions from winner's notebook")
print("  2. Implement TM-score calculator")
print("  3. Run baseline benchmark")
print()
