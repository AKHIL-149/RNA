# V3 Quick Start Guide

## For Immediate Kaggle Submission

### Step 1: Upload to Kaggle
1. Go to your Kaggle notebook interface
2. Upload `akhil.ipynb` (this is the V3 improved version)
3. Make sure your dataset `rna-predictions` is attached

### Step 2: Verify Settings
- Internet: OFF
- Accelerator: None (CPU only)
- Environment: Python 3

### Step 3: Run
Click "Run All" and wait about 2 minutes for completion.

### Step 4: Check Output
Look for these indicators of success:
```
VALIDATION COMPLETE: ALL CHECKS PASSED
The submission file is ready for upload.
```

### Step 5: Download Submission
Get `submission.csv` from `/kaggle/working/` directory.

---

## Expected Results

| Sequence | V2 TM-Score | V3 TM-Score | Improvement |
|----------|-------------|-------------|-------------|
| R1107 | 0.95-1.00 | 0.95-1.00 | Same |
| R1108 | 0.95-1.00 | 0.95-1.00 | Same |
| R1116 | 0.65-0.75 | 0.65-0.75 | Same |
| R1117v2 | 0.00 | 0.25 | +0.25 |
| R1126 | 0.95-1.00 | 0.95-1.00 | Same |
| R1128 | 0.95-1.00 | 0.95-1.00 | Same |
| R1136 | 0.95-1.00 | 0.95-1.00 | Same |
| R1138 | 0.95-1.00 | 0.95-1.00 | Same |
| R1149 | 0.95-1.00 | 0.95-1.00 | Same |
| R1156 | 0.95-1.00 | 0.95-1.00 | Same |
| R1189 | 0.95-1.00 | 0.95-1.00 | Same |
| R1190 | 0.95-1.00 | 0.95-1.00 | Same |
| Mean | 0.834 | ~0.855 | +0.02 |

The main improvement is fixing R1117v2, which previously returned all zeros. Now it gets a reasonable extended chain structure.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'Bio'"
This means biopython installation failed. The notebook should install it from the local wheel automatically. Check that your rna-predictions dataset includes the wheels directory with biopython wheels for both Python 3.10 and 3.11.

### Issue: Notebook times out
This shouldn't happen - runtime is about 2 minutes. Check that you're using CPU (Accelerator = None) not GPU.

### Issue: Validation fails
Look at which specific validation check failed. The most common is if R1117v2 still has zeros, which means the extended chain fallback isn't working.

### Issue: "Exception after deadline"
This is just Kaggle saying the competition deadline has passed. It's not an actual error - your code ran successfully. Look at the output to verify all validations passed.

---

## What's Different from V2

### Main Improvement: Extended Chain Fallback

V2 Output for R1117v2:
```
R1117v2    30       None              0.00%     No templates found
```

V3 Output for R1117v2:
```
R1117v2    30       None              0.00%     Using extended chain
```

### Coordinate Changes

V2 R1117v2 coordinates (all zeros):
```csv
R1117v2_1,G,1,0.0,0.0,0.0,0.0,0.0,0.0,...
R1117v2_2,G,2,0.0,0.0,0.0,0.0,0.0,0.0,...
```

V3 R1117v2 coordinates (proper backbone):
```csv
R1117v2_1,G,1,0.0,0.0,0.0,0.0,0.0,0.0,...
R1117v2_2,G,2,5.9,0.147,0.0,5.9,0.147,0.0,...
```

The extended chain uses proper C1'-C1' spacing (5.9 Angstroms) with a slight sinusoidal curve, which gives a TM-score of about 0.25 instead of 0.0.

---

## Next Steps After V3

Once you have V3 working, here are potential research directions:

1. Complete MSA Integration (1-2 weeks)
   - See NOVEL_APPROACHES_RESEARCH.md
   - Expected improvement: +0.02 TM-score

2. Physics-Informed Gap Filling (3-4 weeks)
   - Replace linear interpolation with distance constraints
   - Expected improvement: +0.05 TM-score

3. PINN for Novel Sequences (6-8 weeks)
   - Proper solution for sequences without templates
   - Expected improvement for R1117v2: 0.25 to 0.45 TM-score

See GETTING_STARTED_RESEARCH.md for detailed implementation guides.

---

## Files You Need

For Kaggle submission:
- `akhil.ipynb` - Main notebook with V3 improvements
- `rna.zip` - Dataset with templates and biopython wheels

For reference:
- `V3_IMPROVEMENTS.md` - Detailed changelog
- `V3_QUICK_START.md` - This file
- `NOVEL_APPROACHES_RESEARCH.md` - Future research directions

---

## Common Issues

The most common issue is seeing "Notebook Threw Exception (after deadline)" at the top of the Kaggle output. This is misleading - it just means the competition deadline passed. Your code actually works fine if you see "VALIDATION COMPLETE: ALL CHECKS PASSED" in the output.

Another common confusion is the zero coordinates percentage. V3 reduces zero coordinates from about 1.2% to 0.4%, which is expected - only the first nucleotide of each sequence starts at the origin.

---

## Summary Checklist

- Upload akhil.ipynb to Kaggle
- Attach rna-predictions dataset
- Set Internet OFF, Accelerator None
- Run all cells
- Check validation passes
- Download submission.csv

Expected improvement: +0.02 mean TM-score (0.834 to ~0.855)
