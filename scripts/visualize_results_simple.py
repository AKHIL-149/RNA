#!/usr/bin/env python3
"""
Simple Performance Visualization

Generate key visualizations from evaluation results.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'
FIGURES_DIR = PROJECT_DIR / 'figures'

# Set style
sns.set_style('whitegrid')

print("=" * 80)
print("GENERATING PERFORMANCE VISUALIZATIONS")
print("=" * 80)
print()

# Load evaluation results
print("Loading data...")
eval_df = pd.read_csv(RESULTS_DIR / 'rmsd_evaluation.csv')
catalog_df = pd.read_csv(PROJECT_DIR / 'results' / 'test_sequences_catalog.csv')
pred_df = pd.read_csv(RESULTS_DIR / 'prediction_summary.csv')

# Merge
data = eval_df.merge(catalog_df, on='target_id', how='left')
data = data.merge(
    pred_df[['target_id', 'best_identity']].rename(columns={'best_identity': 'template_identity'}),
    on='target_id',
    how='left'
)

# Filter successful only
successful = data[data['status'] == 'success'].copy()

# Clean RMSD outliers
successful['rmsd_clean'] = successful['best_rmsd'].apply(lambda x: x if x < 100 else np.nan)

print(f"✓ Loaded {len(data)} targets, {len(successful)} successful")
print()

# Figure 1: TM-score vs Sequence Length
print("[1/5] TM-score vs Sequence Length...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(successful['length'], successful['best_tm_score'],
           s=100, alpha=0.6, c='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Sequence Length (nucleotides)', fontsize=12)
ax.set_ylabel('Best TM-score', fontsize=12)
ax.set_title('TM-score vs Sequence Length', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
for _, row in successful.iterrows():
    ax.annotate(row['target_id'], (row['length'], row['best_tm_score']),
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'tm_vs_length.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: tm_vs_length.png")
plt.close()

# Figure 2: RMSD Distribution
print("[2/5] RMSD Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
rmsd_clean = successful['rmsd_clean'].dropna()
ax.hist(rmsd_clean, bins=20, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(rmsd_clean.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {rmsd_clean.mean():.2f} Å')
ax.axvline(rmsd_clean.median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median = {rmsd_clean.median():.2f} Å')
ax.set_xlabel('RMSD (Å)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('RMSD Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'rmsd_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: rmsd_distribution.png")
plt.close()

# Figure 3: TM-score Distribution
print("[3/5] TM-score Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(successful['best_tm_score'], bins=20, alpha=0.7, color='mediumseagreen', edgecolor='black')
ax.axvline(successful['best_tm_score'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {successful["best_tm_score"].mean():.3f}')
ax.axvline(successful['best_tm_score'].median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median = {successful["best_tm_score"].median():.3f}')
ax.set_xlabel('TM-score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('TM-score Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'tm_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: tm_distribution.png")
plt.close()

# Figure 4: Template Identity vs TM-score
print("[4/5] Template Identity vs TM-score...")
fig, ax = plt.subplots(figsize=(10, 6))
valid = successful.dropna(subset=['template_identity', 'best_tm_score'])
ax.scatter(valid['template_identity'] * 100, valid['best_tm_score'],
           s=100, alpha=0.6, c='purple', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Template Sequence Identity (%)', fontsize=12)
ax.set_ylabel('Best TM-score', fontsize=12)
ax.set_title('Template Identity vs Structure Similarity', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add correlation if enough points
if len(valid) > 1:
    corr = np.corrcoef(valid['template_identity'], valid['best_tm_score'])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'identity_vs_tm.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: identity_vs_tm.png")
plt.close()

# Figure 5: Summary Bar Chart
print("[5/5] Performance Summary...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Top targets by TM-score
top_5 = successful.nlargest(5, 'best_tm_score')
ax1.barh(range(len(top_5)), top_5['best_tm_score'], color='steelblue', alpha=0.7)
ax1.set_yticks(range(len(top_5)))
ax1.set_yticklabels(top_5['target_id'])
ax1.set_xlabel('TM-score', fontsize=11)
ax1.set_title('Top 5 Predictions (by TM-score)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Top targets by RMSD (lowest)
top_rmsd = successful[successful['rmsd_clean'].notna()].nsmallest(5, 'rmsd_clean')
ax2.barh(range(len(top_rmsd)), top_rmsd['rmsd_clean'], color='coral', alpha=0.7)
ax2.set_yticks(range(len(top_rmsd)))
ax2.set_yticklabels(top_rmsd['target_id'])
ax2.set_xlabel('RMSD (Å)', fontsize=11)
ax2.set_title('Top 5 Predictions (by RMSD)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Performance categories
tm_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
tm_labels = ['Poor\n(<0.3)', 'Fair\n(0.3-0.5)', 'Good\n(0.5-0.7)', 'V.Good\n(0.7-0.9)', 'Excellent\n(>0.9)']
tm_counts = []
for i in range(len(tm_bins) - 1):
    low, high = tm_bins[i], tm_bins[i+1]
    if i == len(tm_bins) - 2:
        count = ((successful['best_tm_score'] >= low) & (successful['best_tm_score'] <= high)).sum()
    else:
        count = ((successful['best_tm_score'] >= low) & (successful['best_tm_score'] < high)).sum()
    tm_counts.append(count)

ax3.bar(range(len(tm_labels)), tm_counts, color='mediumseagreen', alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(tm_labels)))
ax3.set_xticklabels(tm_labels, fontsize=10)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('TM-score Categories', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Statistics table
stats_text = f"""
PERFORMANCE STATISTICS

TM-score:
  Mean:   {successful['best_tm_score'].mean():.3f}
  Median: {successful['best_tm_score'].median():.3f}
  Std:    {successful['best_tm_score'].std():.3f}

RMSD (Å):
  Mean:   {rmsd_clean.mean():.2f}
  Median: {rmsd_clean.median():.2f}
  Std:    {rmsd_clean.std():.2f}

Coverage:
  Success: {len(successful)}/{len(data)}
  Rate:    {len(successful)/len(data)*100:.1f}%
"""

ax4.axis('off')
ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.suptitle('Baseline TBM Performance Summary', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'performance_summary.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: performance_summary.png")
plt.close()

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
print(f"\nAll figures saved to: {FIGURES_DIR}")
print()

# Print insights
print("KEY INSIGHTS")
print("=" * 80)

# Excellent predictions
excellent = successful[successful['best_tm_score'] > 0.9]
print(f"\n1. Excellent Predictions (TM > 0.9): {len(excellent)}/{len(successful)}")
for _, row in excellent.iterrows():
    rmsd = row['rmsd_clean'] if pd.notna(row['rmsd_clean']) else row['best_rmsd']
    print(f"   • {row['target_id']}: TM={row['best_tm_score']:.3f}, RMSD={rmsd:.3f}Å")

# Poor predictions
poor = successful[successful['best_tm_score'] < 0.3]
print(f"\n2. Poor Predictions (TM < 0.3): {len(poor)}/{len(successful)}")
for _, row in poor.iterrows():
    rmsd = row['rmsd_clean'] if pd.notna(row['rmsd_clean']) else row['best_rmsd']
    print(f"   • {row['target_id']}: TM={row['best_tm_score']:.3f}, RMSD={rmsd:.3f}Å")

print()
print("=" * 80)
