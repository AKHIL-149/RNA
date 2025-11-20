#!/usr/bin/env python3
"""
Visualize Baseline Performance

Generate comprehensive visualizations of TBM baseline prediction quality.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_DIR = Path(__file__).parent.parent
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
RESULTS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'
FIGURES_DIR = PROJECT_DIR / 'figures'

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("VISUALIZING BASELINE PERFORMANCE")
print("=" * 80)
print()

# Load data
print("Loading data...")
eval_results = pd.read_csv(RESULTS_DIR / 'rmsd_evaluation.csv')
test_catalog = pd.read_csv(PROJECT_DIR / 'results' / 'test_sequences_catalog.csv')
prediction_summary = pd.read_csv(RESULTS_DIR / 'prediction_summary.csv')

# Merge datasets
merged = eval_results.merge(test_catalog, on='target_id', how='left')
merged = merged.merge(
    prediction_summary[['target_id', 'num_templates', 'best_template',
                        'best_identity', 'time_seconds']].rename(
        columns={'best_identity': 'best_template_identity'}
    ),
    on='target_id',
    how='left',
    suffixes=('', '_pred')
)

# Filter successful predictions (from eval_results)
successful = merged[merged['status'] == 'success'].copy()

# Clean outliers (RMSD > 100 Å indicates alignment failure)
successful['best_rmsd_clean'] = successful['best_rmsd'].apply(
    lambda x: x if x < 100 else np.nan
)

print(f"✓ Loaded {len(merged)} targets ({len(successful)} successful)")
print()

# Figure 1: TM-score vs Sequence Length
print("[1/6] Creating TM-score vs Sequence Length plot...")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    successful['sequence_length'],
    successful['best_tm_score'],
    c=successful['best_rmsd_clean'],
    s=100,
    alpha=0.6,
    cmap='viridis_r'
)
ax.set_xlabel('Sequence Length (nucleotides)', fontsize=12)
ax.set_ylabel('Best TM-score', fontsize=12)
ax.set_title('TM-score vs Sequence Length', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('RMSD (Å)', fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'baseline_tm_vs_length.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {FIGURES_DIR / 'baseline_tm_vs_length.png'}")
plt.close()

# Figure 2: RMSD vs Sequence Length
print("[2/6] Creating RMSD vs Sequence Length plot...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    successful['sequence_length'],
    successful['best_rmsd_clean'],
    s=100,
    alpha=0.6,
    c='coral'
)
ax.set_xlabel('Sequence Length (nucleotides)', fontsize=12)
ax.set_ylabel('Best RMSD (Å)', fontsize=12)
ax.set_title('RMSD vs Sequence Length', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'baseline_rmsd_vs_length.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {FIGURES_DIR / 'baseline_rmsd_vs_length.png'}")
plt.close()

# Figure 3: TM-score Distribution
print("[3/6] Creating TM-score distribution plot...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(successful['best_tm_score'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(successful['best_tm_score'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean = {successful["best_tm_score"].mean():.3f}')
ax.axvline(successful['best_tm_score'].median(), color='orange', linestyle='--',
           linewidth=2, label=f'Median = {successful["best_tm_score"].median():.3f}')
ax.set_xlabel('TM-score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('TM-score Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'baseline_tm_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {FIGURES_DIR / 'baseline_tm_distribution.png'}")
plt.close()

# Figure 4: Template Identity vs TM-score
print("[4/6] Creating Template Identity vs TM-score plot...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    successful['best_template_identity'],
    successful['best_tm_score'],
    s=100,
    alpha=0.6,
    c='mediumseagreen'
)
ax.set_xlabel('Best Template Sequence Identity (%)', fontsize=12)
ax.set_ylabel('Best TM-score', fontsize=12)
ax.set_title('Template Identity vs Structure Similarity', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add correlation
valid_mask = ~(pd.isna(successful['best_template_identity']) | pd.isna(successful['best_tm_score']))
if valid_mask.sum() > 1:
    correlation = np.corrcoef(
        successful.loc[valid_mask, 'best_template_identity'],
        successful.loc[valid_mask, 'best_tm_score']
    )[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'baseline_identity_vs_tm.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {FIGURES_DIR / 'baseline_identity_vs_tm.png'}")
plt.close()

# Figure 5: Performance by RNA Type
print("[5/6] Creating Performance by RNA Type plot...")
if 'rna_type' in successful.columns and successful['rna_type'].notna().sum() > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # TM-score by type
    rna_types = successful.groupby('rna_type')['best_tm_score'].agg(['mean', 'count'])
    rna_types = rna_types[rna_types['count'] >= 1]  # Only types with data
    ax1.bar(range(len(rna_types)), rna_types['mean'], color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(rna_types)))
    ax1.set_xticklabels(rna_types.index, rotation=45, ha='right')
    ax1.set_ylabel('Mean TM-score', fontsize=12)
    ax1.set_title('Mean TM-score by RNA Type', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # RMSD by type
    rna_types_rmsd = successful.groupby('rna_type')['best_rmsd_clean'].agg(['mean', 'count'])
    rna_types_rmsd = rna_types_rmsd[rna_types_rmsd['count'] >= 1]
    ax2.bar(range(len(rna_types_rmsd)), rna_types_rmsd['mean'], color='coral', alpha=0.7)
    ax2.set_xticks(range(len(rna_types_rmsd)))
    ax2.set_xticklabels(rna_types_rmsd.index, rotation=45, ha='right')
    ax2.set_ylabel('Mean RMSD (Å)', fontsize=12)
    ax2.set_title('Mean RMSD by RNA Type', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'baseline_by_rna_type.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {FIGURES_DIR / 'baseline_by_rna_type.png'}")
    plt.close()
else:
    print("  ⊘ Skipped: No RNA type information available")

# Figure 6: Summary Dashboard
print("[6/6] Creating summary dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Overall statistics
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
stats_text = f"""
BASELINE TBM PERFORMANCE SUMMARY
{'='*80}

Total Targets: {len(merged)}   |   Successful Predictions: {len(successful)}   |   Success Rate: {len(successful)/len(merged)*100:.1f}%

TM-SCORE STATISTICS                    RMSD STATISTICS (Å)
  Mean:   {successful['best_tm_score'].mean():.3f}                             Mean:   {successful['best_rmsd_clean'].mean():.2f}
  Median: {successful['best_tm_score'].median():.3f}                             Median: {successful['best_rmsd_clean'].median():.2f}
  Std:    {successful['best_tm_score'].std():.3f}                             Std:    {successful['best_rmsd_clean'].std():.2f}
  Min:    {successful['best_tm_score'].min():.3f}                             Min:    {successful['best_rmsd_clean'].min():.2f}
  Max:    {successful['best_tm_score'].max():.3f}                             Max:    {successful['best_rmsd_clean'].max():.2f}

TEMPLATE QUALITY
  Mean Sequence Identity: {successful['best_template_identity'].mean():.1f}%
  Mean Templates Found:   {successful['num_templates'].mean():.1f}
"""
ax1.text(0.05, 0.5, stats_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

# TM-score histogram
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(successful['best_tm_score'], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
ax2.set_xlabel('TM-score', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_title('TM-score Distribution', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# RMSD histogram
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(successful['best_rmsd_clean'].dropna(), bins=15, alpha=0.7, color='coral', edgecolor='black')
ax3.set_xlabel('RMSD (Å)', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.set_title('RMSD Distribution', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Template Identity histogram
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(successful['best_template_identity'].dropna(), bins=15, alpha=0.7,
         color='mediumseagreen', edgecolor='black')
ax4.set_xlabel('Identity (%)', fontsize=10)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title('Template Identity', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# TM vs Length
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(successful['sequence_length'], successful['best_tm_score'],
            alpha=0.6, s=60, c='steelblue')
ax5.set_xlabel('Length (nt)', fontsize=10)
ax5.set_ylabel('TM-score', fontsize=10)
ax5.set_title('TM-score vs Length', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# RMSD vs Length
ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(successful['sequence_length'], successful['best_rmsd_clean'],
            alpha=0.6, s=60, c='coral')
ax6.set_xlabel('Length (nt)', fontsize=10)
ax6.set_ylabel('RMSD (Å)', fontsize=10)
ax6.set_title('RMSD vs Length', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Identity vs TM
ax7 = fig.add_subplot(gs[2, 2])
ax7.scatter(successful['best_template_identity'], successful['best_tm_score'],
            alpha=0.6, s=60, c='mediumseagreen')
ax7.set_xlabel('Identity (%)', fontsize=10)
ax7.set_ylabel('TM-score', fontsize=10)
ax7.set_title('Identity vs TM-score', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

plt.savefig(FIGURES_DIR / 'baseline_summary_dashboard.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {FIGURES_DIR / 'baseline_summary_dashboard.png'}")
plt.close()

print()
print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print(f"\nAll figures saved to: {FIGURES_DIR}")
print()

# Print key insights
print("KEY INSIGHTS")
print("=" * 80)

# High TM-score predictions
high_tm = successful[successful['best_tm_score'] > 0.9]
print(f"\n1. EXCELLENT PREDICTIONS (TM-score > 0.9): {len(high_tm)}/{len(successful)} ({len(high_tm)/len(successful)*100:.1f}%)")
for _, row in high_tm.iterrows():
    print(f"   • {row['target_id']}: TM={row['best_tm_score']:.3f}, RMSD={row['best_rmsd_clean']:.3f}Å, Identity={row['best_template_identity']:.1f}%")

# Low TM-score predictions
low_tm = successful[successful['best_tm_score'] < 0.3]
print(f"\n2. POOR PREDICTIONS (TM-score < 0.3): {len(low_tm)}/{len(successful)} ({len(low_tm)/len(successful)*100:.1f}%)")
for _, row in low_tm.iterrows():
    rmsd_str = f"{row['best_rmsd_clean']:.3f}" if pd.notna(row['best_rmsd_clean']) else "N/A"
    print(f"   • {row['target_id']}: TM={row['best_tm_score']:.3f}, RMSD={rmsd_str}Å, Identity={row['best_template_identity']:.1f}%")

# Correlation analysis
valid_mask = ~(pd.isna(successful['best_template_identity']) | pd.isna(successful['best_tm_score']))
if valid_mask.sum() > 1:
    correlation = np.corrcoef(
        successful.loc[valid_mask, 'best_template_identity'],
        successful.loc[valid_mask, 'best_tm_score']
    )[0, 1]
    print(f"\n3. SEQUENCE IDENTITY vs TM-SCORE CORRELATION: {correlation:.3f}")
    if correlation > 0.7:
        print("   → Strong positive correlation: Higher sequence identity → Better structure prediction")
    elif correlation > 0.4:
        print("   → Moderate positive correlation: Sequence identity partially predicts structure quality")
    else:
        print("   → Weak correlation: Sequence identity alone insufficient for structure prediction")

print()
print("=" * 80)
