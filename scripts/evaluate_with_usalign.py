#!/usr/bin/env python3
"""
Evaluate with Official US-align

Use the official US-align tool for accurate TM-score calculation.
"""

import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbm import TBMPipeline

PROJECT_DIR = Path(__file__).parent.parent
STANFORD_DIR = PROJECT_DIR / 'stanford-rna-3d-folding'
PRED_DIR = PROJECT_DIR / 'results' / 'tbm_baseline' / 'predictions'
VALID_DIR = PROJECT_DIR / 'results' / 'tbm_baseline' / 'validation_pdbs'
RESULTS_DIR = PROJECT_DIR / 'results' / 'usalign_evaluation'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("EVALUATION WITH OFFICIAL US-ALIGN")
print("=" * 80)
print()

# Check if US-align is installed
print("[1/4] Checking US-align installation...")
try:
    result = subprocess.run(['USalign', '-h'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0 or 'USalign' in result.stdout or 'USalign' in result.stderr:
        print("  ✓ US-align found")
        usalign_cmd = 'USalign'
    else:
        raise FileNotFoundError
except (FileNotFoundError, subprocess.TimeoutExpired):
    # Try alternative name
    try:
        result = subprocess.run(['us-align', '-h'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 or 'align' in result.stdout.lower():
            print("  ✓ US-align found (as 'us-align')")
            usalign_cmd = 'us-align'
        else:
            raise FileNotFoundError
    except:
        print("  ✗ US-align not found")
        print()
        print("To install US-align:")
        print("  1. Download from: https://zhanggroup.org/US-align/")
        print("  2. Compile: g++ -static -O3 -ffast-math -o USalign USalign.cpp")
        print("  3. Add to PATH or place in project directory")
        print()
        print("Falling back to downloading...")
        
        # Try to download and compile
        try:
            print("  Downloading US-align...")
            subprocess.run([
                'curl', '-o', '/tmp/USalign.cpp',
                'https://zhanggroup.org/US-align/bin/module/USalign.cpp'
            ], check=True, capture_output=True)
            
            print("  Compiling US-align...")
            subprocess.run([
                'g++', '-O3', '-ffast-math', 
                '-o', str(PROJECT_DIR / 'USalign'),
                '/tmp/USalign.cpp'
            ], check=True, capture_output=True)
            
            usalign_cmd = str(PROJECT_DIR / 'USalign')
            print("  ✓ US-align compiled successfully")
        except Exception as e:
            print(f"  ✗ Failed to download/compile: {e}")
            print()
            print("Please install US-align manually and re-run this script.")
            sys.exit(1)

print()

# Find prediction and validation files
print("[2/4] Finding PDB files...")
pred_files = sorted(PRED_DIR.glob('*.pdb'))
valid_files = sorted(VALID_DIR.glob('*.pdb'))

print(f"  Predictions: {len(pred_files)}")
print(f"  Validation: {len(valid_files)}")
print()

# Create mapping
pred_map = {f.stem: f for f in pred_files}
valid_map = {f.stem.replace('_validation', ''): f for f in valid_files}

# Find matching pairs
matches = []
for target_id in pred_map.keys():
    if target_id in valid_map:
        matches.append((target_id, pred_map[target_id], valid_map[target_id]))

print(f"  Matching pairs: {len(matches)}")
print()

# Run US-align on each pair
print("[3/4] Running US-align evaluations...")
results = []

for i, (target_id, pred_file, valid_file) in enumerate(matches, 1):
    print(f"  [{i}/{len(matches)}] {target_id}...", end=' ')
    
    try:
        # Run US-align
        cmd = [usalign_cmd, str(pred_file), str(valid_file), '-ter', '0']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"FAILED (exit code {result.returncode})")
            results.append({
                'target_id': target_id,
                'status': 'failed',
                'tm_score': None,
                'rmsd': None
            })
            continue
        
        # Parse output
        output = result.stdout
        tm_score = None
        rmsd = None
        
        for line in output.split('\n'):
            if 'TM-score=' in line and 'Chain_1' in line:
                # Extract TM-score for chain 1 (predicted)
                parts = line.split('TM-score=')
                if len(parts) > 1:
                    tm_str = parts[1].split()[0]
                    try:
                        tm_score = float(tm_str)
                    except:
                        pass
            
            if 'RMSD=' in line and 'aligned length=' in line:
                # Extract RMSD
                parts = line.split('RMSD=')
                if len(parts) > 1:
                    rmsd_str = parts[1].split(',')[0].strip()
                    try:
                        rmsd = float(rmsd_str)
                    except:
                        pass
        
        if tm_score is not None:
            print(f"TM={tm_score:.3f}")
            results.append({
                'target_id': target_id,
                'status': 'success',
                'tm_score': tm_score,
                'rmsd': rmsd
            })
        else:
            print("FAILED (parsing)")
            results.append({
                'target_id': target_id,
                'status': 'parse_error',
                'tm_score': None,
                'rmsd': None
            })
    
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        results.append({
            'target_id': target_id,
            'status': 'timeout',
            'tm_score': None,
            'rmsd': None
        })
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            'target_id': target_id,
            'status': 'error',
            'tm_score': None,
            'rmsd': None,
            'error': str(e)
        })

print()

# Save results
print("[4/4] Saving results...")
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / 'usalign_evaluation.csv', index=False)
print(f"  ✓ Saved to {RESULTS_DIR / 'usalign_evaluation.csv'}")
print()

# Calculate metrics
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

successful = results_df[results_df['status'] == 'success']

print("Overall Performance:")
print(f"  Total sequences: {len(results_df)}")
print(f"  Successful: {len(successful)}/{len(results_df)} ({len(successful)/len(results_df)*100:.1f}%)")
print()

if len(successful) > 0:
    # Clean metrics (exclude R1116 if present)
    clean = successful[successful['target_id'] != 'R1116']
    
    if len(clean) > 0:
        print("Clean Performance (excluding R1116):")
        print(f"  Mean TM-score: {clean['tm_score'].mean():.3f}")
        print(f"  Median TM-score: {clean['tm_score'].median():.3f}")
        print(f"  Min TM-score: {clean['tm_score'].min():.3f}")
        print(f"  Max TM-score: {clean['tm_score'].max():.3f}")
        print()
        
        # Quality breakdown
        excellent = (clean['tm_score'] >= 0.9).sum()
        high = (clean['tm_score'] >= 0.7).sum()
        acceptable = (clean['tm_score'] >= 0.5).sum()
        
        print("Quality Distribution:")
        print(f"  Excellent (≥0.9): {excellent}/{len(clean)} ({excellent/len(clean)*100:.1f}%)")
        print(f"  High (≥0.7): {high}/{len(clean)} ({high/len(clean)*100:.1f}%)")
        print(f"  Acceptable (≥0.5): {acceptable}/{len(clean)} ({acceptable/len(clean)*100:.1f}%)")
        print()
        
        # Compare to BioPython baseline
        biopython_mean = 0.834
        usalign_mean = clean['tm_score'].mean()
        difference = usalign_mean - biopython_mean
        
        print("Comparison to BioPython Approximation:")
        print(f"  BioPython mean: {biopython_mean:.3f}")
        print(f"  US-align mean: {usalign_mean:.3f}")
        print(f"  Difference: {difference:+.3f} ({difference/biopython_mean*100:+.1f}%)")
        print()
        
        if usalign_mean >= 0.87:
            print("✓ GOAL ACHIEVED! US-align TM-score ≥ 0.87")
        elif usalign_mean > biopython_mean:
            print(f"✓ US-align scores are higher than BioPython approximation")
            gap = 0.87 - usalign_mean
            print(f"  Still {gap:.3f} below 0.87 target")
        else:
            print(f"⚠️  US-align scores similar to or lower than BioPython")

print()
print("=" * 80)
