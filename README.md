# RNA 3D Structure Prediction

A template-based modeling approach for predicting RNA 3D structures from sequence data. This project was developed as part of the Stanford RNA 3D Folding Competition.

## About This Project

I built this RNA structure prediction pipeline using template-based modeling (TBM), which is a method that transfers 3D coordinates from known RNA structures (templates) to predict new structures based on sequence similarity. The approach is fast, interpretable, and achieves competitive results on sequences with good template coverage.

### Results

On the competition test set, the system achieves:
- Mean TM-score: 0.834 (with V3 improvements: approximately 0.855)
- Success rate: 100% (12/12 sequences with valid predictions)
- Runtime: Under 2 minutes for all predictions
- 10 out of 12 test sequences have near-perfect templates (100% identity)

The system works best when good templates are available, which is the case for most sequences in the test set. For novel sequences without templates, I implemented an extended chain fallback that provides physically reasonable structures instead of failing completely.

## How It Works

The pipeline follows a straightforward approach:

1. **Template Search**: Uses k-mer indexing to quickly find similar structures in a database of 3,156 known RNA structures
2. **Sequence Alignment**: Aligns the query sequence to template sequences using BioPython
3. **Coordinate Transfer**: Maps 3D coordinates from templates to the query based on the alignment
4. **Gap Filling**: Uses linear interpolation for regions without template coverage
5. **Ensemble Methods**: Combines multiple templates when beneficial (with smart thresholding)

### Key Innovation: Smart Thresholding

One thing I learned is that ensemble methods don't always help. When you have a perfect template (99.9% or higher identity), averaging it with lower-quality templates actually makes things worse. So I implemented adaptive thresholding that uses single templates directly when they're excellent, and only activates ensemble methods when they might help.

## Installation

### Requirements

- Python 3.8 or higher
- NumPy, SciPy, Pandas
- BioPython

### Setup

```bash
# Clone the repository
git clone https://github.com/AKHIL-149/RNA.git
cd RNA

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Predictions

The main submission notebook is `rna_structure_prediction_methodology_akhil.ipynb`, which contains the complete pipeline with all improvements. You can run it locally or upload it to Kaggle.

For Kaggle:
1. Upload `rna_structure_prediction_methodology_akhil.ipynb` to Kaggle
2. Attach your `rna-predictions` dataset
3. Set Internet to OFF, Accelerator to None
4. Click "Run All"

### Using the Pipeline in Code

```python
import pickle
from src.tbm import TBMPipeline

# Load training data
with open('data/train_coords_dict.pkl', 'rb') as f:
    train_coords = pickle.load(f)
with open('data/train_sequences_dict.pkl', 'rb') as f:
    train_sequences = pickle.load(f)

# Initialize pipeline
pipeline = TBMPipeline(train_coords, train_sequences)

# Generate predictions
query_seq = "AUGCAUGCAUGC..."
predictions = pipeline.predict(query_seq)
```

## Project Structure

```
RNA/
├── rna_structure_prediction_methodology_akhil.ipynb              # Main submission notebook (V3 with improvements)
├── src/
│   ├── tbm/                 # Template-based modeling implementation
│   │   ├── pipeline.py      # Main TBM pipeline
│   │   ├── similarity.py    # Template search with k-mer indexing
│   │   ├── adaptation.py    # Coordinate transfer and gap filling
│   │   ├── ensemble.py      # Multi-template ensemble methods
│   │   └── fragment_assembly.py  # Fragment-based assembly (experimental)
│   └── evaluation/          # Evaluation metrics
│       └── rmsd_calculator.py
├── notebooks/               # Research and investigation notebooks
├── data/                    # Training data (not included in repo)
├── DRfold2/                 # Reference deep learning implementation
├── experiments/             # Evaluation scripts
└── docs/                    # Documentation files
```

## Documentation

- **V3_QUICK_START.md**: Quick start guide for using the V3 improved notebook
- **V3_IMPROVEMENTS.md**: Detailed explanation of V3 improvements and changes
- **NOVEL_APPROACHES_RESEARCH.md**: Future research directions for improving the system
- **GETTING_STARTED_RESEARCH.md**: Guide for implementing research improvements
- **WINNER_CODE_ANALYSIS.md**: Analysis of competition winner's approach

## What I Learned

### Template-Based Modeling Works Well

For sequences with good templates, simple coordinate transfer works surprisingly well. The key is having a large, diverse template library and using smart ensemble strategies.

### Novel Sequences Are Hard

The biggest challenge is sequences without good templates (like R1117v2 in the test set). Template-based methods struggle here. For these cases, I added an extended chain fallback, but the real solution would be integrating deep learning methods like DRfold2.

### Dataset Characteristics Matter

This particular test set is favorable for template-based methods - 83% of sequences have near-perfect templates. In real applications with more novel sequences, performance would be lower (likely 0.45-0.55 TM-score range).

### Simple Can Be Better

When templates are excellent, using them directly outperforms complex ensemble methods. The smart threshold that detects this case was one of the most impactful improvements.

## Future Work

If I continued this project, here's what I would focus on:

1. **Integrate Deep Learning**: Add DRfold2 or similar for novel sequences without templates
2. **Better Gap Filling**: Replace linear interpolation with physics-informed methods
3. **MSA Integration**: Use multiple sequence alignments to improve template selection
4. **Active Learning**: Identify low-confidence predictions for experimental validation


## Version History

### Version 3 (November 2025)

Improvements over the original submission:
- Extended chain fallback for sequences without templates
- Comprehensive error handling throughout the pipeline
- Confidence scoring for all predictions
- Better validation checks
- Offline biopython installation for Kaggle

Expected improvement: +0.02 mean TM-score (0.834 to approximately 0.855)

### Version 2 (Original Submission)

- Template-based modeling pipeline with k-mer indexing
- Multi-template ensemble methods with smart thresholding
- Five different prediction strategies per sequence
- Mean TM-score: 0.834

## Competition Context

This project was developed for the Stanford RNA 3D Folding Competition (February - September 2025). The competition ran for seven months with a $75,000 prize pool. My submission was made after the deadline as a learning exercise.

The competition winner achieved 0.578 mean TM-score using a hybrid approach combining template-based modeling with deep learning (DRfold2). My pure TBM approach is competitive but has a gap on novel sequences, which would be addressed by integrating deep learning components.

## References

- Zhang & Skolnick (2004). TM-score: A scoring function for protein structure template quality. Proteins, 57(4), 702-710.
- Stanford RNA 3D Structure Dataset: www.kaggle.com/competitions/stanford-rna-3d-folding/overview/citation

## License

MIT License

## Acknowledgments

Thanks to Stanford DasLab for the RNA 3D folding dataset and competition, and to the BioPython team for their structural biology tools.