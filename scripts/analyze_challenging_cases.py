#!/usr/bin/env python3
"""
Analyze Challenging Cases

Deep dive into sequences with TM < 0.7 to identify improvement opportunities.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline

PROJECT_DIR = Path(__file__).parent.parent
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
RESULTS_DIR = PROJECT_DIR / 'results' / 'tbm_baseline'

print("=" * 80)
print("ANALYZING CHALLENGING CASES")
print("=" * 80)
print()

# Load data
print("[1/3] Loading data...")
with open(PROJECT_DIR / 'data' / 'train_coords_dict.pkl', 'rb') as f:
    train_coords = pickle.load(f)
with open(PROJECT_DIR / 'data' / 'train_sequences_dict.pkl', 'rb') as f:
    train_sequences = pickle.load(f)

eval_df = pd.read_csv(RESULTS_DIR / 'rmsd_evaluation.csv')
test_seqs = pd.read_csv(STANFORD_DIR / 'test_sequences.csv')
pred_summary = pd.read_csv(RESULTS_DIR / 'prediction_summary.csv')

pipeline = TBMPipeline(train_coords, train_sequences)

print(f"  ✓ Loaded {len(train_coords)} templates")
print()

# Identify challenging cases (TM < 0.7)
print("[2/3] Identifying challenging cases...")
challenging = eval_df[
    (eval_df['status'] == 'success') &
    (eval_df['best_tm_score'] < 0.7) &
    (eval_df['target_id'] != 'R1116')  # Exclude corrupted
].copy()

print(f"  Found {len(challenging)} sequences with TM < 0.7")
print()

# Analyze each case
print("[3/3] Detailed analysis...")
print("=" * 80)
print()

for idx, row in challenging.iterrows():
    target_id = row['target_id']
    tm_score = row['best_tm_score']

    print(f"\n{'='*80}")
    print(f"TARGET: {target_id} (TM = {tm_score:.3f})")
    print(f"{'='*80}")

    # Get test sequence info
    test_info = test_seqs[test_seqs['target_id'] == target_id].iloc[0]
    pred_info = pred_summary[pred_summary['target_id'] == target_id].iloc[0]

    print(f"\nSequence Information:")
    print(f"  Length: {test_info['length']} nt")
    print(f"  Type: {test_info.get('rna_type', 'Unknown')}")
    print(f"  GC content: {test_info.get('GC', 0)*100:.1f}%")

    print(f"\nTemplate Matching:")
    print(f"  Best template: {pred_info['best_template']}")
    print(f"  Sequence identity: {pred_info['best_identity']*100:.1f}%")
    print(f"  Templates found: {pred_info['num_templates']}")

    # Get all templates for this sequence
    query_seq = test_info['sequence']
    templates = pipeline.find_templates(query_seq, top_n=10, min_identity=0.5)

    print(f"\nAvailable Templates (top 10):")
    if len(templates) > 0:
        for i, t in enumerate(templates[:5], 1):
            print(f"  {i}. {t['template_id']}: {t['identity']*100:.1f}% identity, "
                  f"{t['coverage']*100:.1f}% coverage")
        if len(templates) > 5:
            print(f"  ... and {len(templates)-5} more templates")
    else:
        print("  No templates found!")

    print(f"\nPerformance Metrics:")
    print(f"  Current TM-score: {tm_score:.3f}")
    print(f"  Target TM-score: 0.70+ (improvement needed: {0.70-tm_score:.3f})")
    print(f"  RMSD: {row['best_rmsd']:.3f} Å")

    # Improvement recommendations
    print(f"\n{'─'*80}")
    print("IMPROVEMENT RECOMMENDATIONS:")
    print(f"{'─'*80}")

    if test_info['length'] > 500:
        print("  ⚠️  LONG SEQUENCE (>500nt)")
        print("     → Try fragment-based assembly")
        print("     → Use multiple templates for different regions")
        print("     → Add long-range contact constraints")

    if pred_info['num_templates'] < 3:
        print("  ⚠️  LIMITED TEMPLATES")
        print("     → Lower similarity threshold")
        print("     → Use MSA-based search")
        print("     → Try more diverse template selection")

    if pred_info['best_identity'] < 0.9:
        print("  ⚠️  MODERATE SEQUENCE IDENTITY")
        print("     → Multi-template consensus")
        print("     → Add secondary structure constraints")
        print("     → Use covariation information")

    if len(templates) >= 3:
        print("  ✓  MULTIPLE TEMPLATES AVAILABLE")
        print("     → Try weighted ensemble of top 3-5 templates")
        print("     → Use diversity-based selection")
        print("     → Implement consensus structure")

    if row['best_rmsd'] > 10:
        print("  ⚠️  HIGH RMSD")
        print("     → Check template quality")
        print("     → Try different alignment parameters")
        print("     → Add refinement step")

    print()

print("=" * 80)
print("SUMMARY OF CHALLENGES")
print("=" * 80)
print()

# Overall statistics
print(f"Total challenging cases: {len(challenging)}")
print()

# Categorize by issue
long_sequences = challenging.merge(test_seqs, on='target_id')
long_seq_count = (long_sequences['length'] > 500).sum()
low_identity = challenging.merge(pred_summary, on='target_id')
low_id_count = (low_identity['best_identity'] < 0.9).sum()

print("Issue Categories:")
print(f"  Long sequences (>500nt):        {long_seq_count}/{len(challenging)}")
print(f"  Low template identity (<90%):   {low_id_count}/{len(challenging)}")
print(f"  High RMSD (>10Å):               {(challenging['best_rmsd'] > 10).sum()}/{len(challenging)}")
print()

print("Recommended Optimization Priority:")
print("  1. Multi-template weighted ensemble (helps all cases)")
print("  2. Fragment-based assembly (for long sequences)")
print("  3. Template quality scoring (filter bad templates)")
print("  4. Secondary structure constraints (low identity cases)")
print()

print("Expected Impact:")
print("  Multi-template: +0.05-0.10 TM-score")
print("  Fragment assembly: +0.10-0.15 for long sequences")
print("  Quality scoring: +0.03-0.05 across board")
print("  Structure constraints: +0.05-0.08 for low identity")
print()

print("=" * 80)
