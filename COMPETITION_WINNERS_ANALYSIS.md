# Stanford RNA 3D Folding Competition - Winner Solutions Analysis

**Competition Period**: February 27 - September 24, 2025
**Prize Pool**: $75,000 (split among top 3)
**Evaluation Metric**: TM-score (0-1 scale, best-of-5 predictions)

---

## 🥇 1st Place Solution: "john"

**Final Score**: Private LB: 0.57773 | Public LB: 0.60388

### Core Strategy: Hybrid TBM + DRfold2

**Philosophy**: "Without GPUs, training from scratch was not viable. My early research showed Template-Based Modeling consistently dominated, so I committed to TBM from day one."

---

### Architecture Components

#### 1. Template-Based Modeling (TBM) - Primary Approach

**Five-Step Pipeline**:

1. **Search** - Finding Similar Structures
   - Sequence alignment against PDB database
   - Custom nucleotide mapping for 93 variants (including modified bases)
   - Disorder-aware coordinate extraction from CIF files

2. **Alignment** - Sequence Mapping
   - Global sequence alignment
   - Gap penalties optimized for RNA
   - Creates translation guide between query and template

3. **Transfer** - Coordinate Inheritance
   - Direct copying of 3D coordinates for matched positions
   - Leverages evolutionary conservation (structure > sequence)

4. **Gap Fill** - Geometric Backbone Reconstruction
   ```
   For insertions/deletions:
   - Maintains C1'-C1' distance (~5.9Å between consecutive nucleotides)
   - Compressed gaps: sinusoidal perturbations perpendicular to backbone
   - Normal gaps: linear interpolation between flanking coordinates
   - Terminal extensions: follow established backbone direction
   ```

5. **Adaptive Refinement** - Confidence-Based Optimization
   ```
   High-confidence templates (>0.8): minimal constraints
   Medium-confidence: moderate sequential distance constraints (5.5-6.5Å)
   Low-confidence: additional steric clash prevention + base-pairing constraints

   Constraint strength = 0.8 × (1 - min(confidence, 0.8))
   ```

#### 2. DRfold2 Enhancements

**Selection Module Improvements**:
- **Double Precision**: Consistent float64 operations → reduced numerical errors
- **Vectorized Distance Calculations**: GPU-accelerated via torch.cdist
- **Optimized Energy Functions**: Pre-computed cubic spline coefficients
- **Motivation**: Authors reported DRfold2 sometimes failed to select best models (5th-ranked outperforming 1st)

**Optimization Module Improvements**:
- **PyTorch LBFGS**: Native optimizer with automatic differentiation (better than SciPy)
- **GPU Acceleration**: Energy calculations and gradients on GPU
- **External Knowledge Integration**: Enhanced with Boltz-1 (credit: @youhanlee)
- **Inspiration**: Authors demonstrated AlphaFold3 integration improved results

#### 3. Hybrid Strategy

**Decision Logic**:
```python
if sequence_length < threshold or time_budget_exhausted:
    use_template_based_modeling()
elif drfold2_available:
    try:
        use_drfold2()
    except:
        fallback_to_template()
else:
    use_template_based_modeling()
```

---

### Key Insights from Winner

**1. Metric Understanding**:
- TM-score normalized by length (50nt and 200nt comparable)
- Robust to local errors
- **Implication**: Prioritize overall fold over atomic precision

**2. Data Processing**:
- Systematic processing of all PDB_RNA CIF files
- 93 nucleotide variant mapping (modified bases)
- Disorder-aware coordinate extraction

**3. Strategic Simplifications**:
- Did NOT use all 80 DRfold2 checkpoints (too slow)
- Only used: Energy selection + Arena for post-processing
- Other steps added minimal TM-score improvement but significant runtime

**4. Resource Constraints**:
- No GPU access for training
- Focused on optimization over fine-tuning
- Leveraged existing models intelligently

---

## 🥉 3rd Place Solution: "Eigen"

**Final Score**: Private LB: 0.54312 | Public LB: 0.61253

### Core Strategy: Ensemble of DRfold2, Protenix, and Boltz-1

**Resources**: NVIDIA GH200 (96GB VRAM) provided by Eigen Company

---

### Architecture Components

#### 1. Data Preparation

**Custom rMSA Generation**:
- Built own rMSA using official code
- Organizer's v2 rMSA didn't cover recent data
- **Computational Cost**: 14 days with multiprocessing across multiple servers
- **Shared Dataset**: [Google Drive link provided]

**Training Datasets** (two versions tested):
1. **Full RNA Dataset**: All recent data including CASP16 (uploaded by @tant64)
2. **RNA-only Dataset**: Excluded RNA-protein/RNA-DNA complexes (per host clarification)
- **Both datasets shared**: [Google Drive link provided]

#### 2. Model Ensemble

**Component 1: DRfold2**
- Best performance on sequences <400 nucleotides
- Used v1 data
- Simplified pipeline: Energy selection + Arena only (skipped clustering + full 80 checkpoints)
- **Reason**: Other steps added little TM-score improvement but high runtime cost

**Component 2: Protenix (Fine-tuned)**
- Trained on NVIDIA GH200 (96GB VRAM)
- Sequence limit: <800nt (memory constraints)
- **Fine-tuning approach**:
  - Used code from @lihaoweicvch
  - Fine-tuning WITHOUT rMSA did not help
  - Fine-tuned WITH rMSA
  - Modified only `max_steps`
  - Full dataset run: ~1 day
  - Early checkpoints (2-3 cycles) performed worse
  - Performance improved with longer fine-tuning
- **Inference**: Modified @geraseva's code to include rMSA

**Component 3: Boltz-1**
- Not strongest model alone
- **Value**: Improved ensemble diversity
- **Rationale**: Scoring metric (best-of-5) favors diverse outputs
- Used @youhanlee's inference notebook

---

### Final Submission Strategies

#### Submission 1: DRfold2 + Protenix
- **Public LB**: 0.60338 | **Private LB**: 0.52787
- **Sequences <400nt**: 3×DRfold2 + 2×Protenix (RNA-only, with MSA)
- **Sequences >400nt**: 2×Protenix (RNA-only) + 2×Protenix (All) + Protenix baseline

#### Submission 2: Protenix + Boltz (BEST)
- **Public LB**: 0.61253 | **Private LB**: 0.54312
- **Configuration**: 2×Protenix (RNA-only) + Protenix (All) + Protenix baseline + Boltz baseline

---

### Key Insights from 3rd Place

**1. Ensemble Diversity Matters**:
- Adding Boltz-1 (weaker alone) improved ensemble performance
- Best-of-5 metric rewards diverse predictions

**2. Fine-Tuning Considerations**:
- Fine-tuning without rMSA did not help
- rMSA critical for improvement
- Longer fine-tuning better than early checkpoints

**3. Resource Allocation**:
- 14 days to generate custom rMSA
- Full training run ~1 day on GH200
- Computational resources enabled experimentation

**4. Negative Results**:
- Multiple Protenix outputs → DRfold energy selection → Arena did NOT improve
- Important to document what didn't work

---

## Comparative Analysis: 1st vs 3rd Place

| Aspect | 1st Place (john) | 3rd Place (Eigen) |
|--------|------------------|-------------------|
| **Primary Approach** | TBM-heavy hybrid | Deep learning ensemble |
| **Computational Resources** | No GPU (CPU only) | NVIDIA GH200 (96GB) |
| **Philosophy** | Template-based dominance | Model diversity + fine-tuning |
| **DRfold2 Usage** | Enhanced optimization modules | Simplified pipeline (<400nt) |
| **Fine-tuning** | None (no GPU) | Extensive Protenix fine-tuning |
| **Custom Data** | Systematic PDB processing | Custom rMSA (14 days) |
| **Ensemble Strategy** | TBM + DRfold2 fallback | DRfold2 + Protenix + Boltz-1 |
| **Sequence Handling** | Length-based strategy | <400nt vs >400nt split |
| **Private LB Score** | 0.57773 (1st) | 0.54312 (3rd) |
| **Public LB Score** | 0.60388 (3rd) | 0.61253 (1st) |
| **Shake-up** | +2 positions | -2 positions |

---

## Critical Observations

### 1. Public-Private Leaderboard Discrepancy

**3rd Place ("Eigen")**:
- Public: 0.61253 (1st) → Private: 0.54312 (3rd)
- **Drop**: -0.06941 (-11.3%)
- **Shake-down**: Lost 2 positions

**1st Place ("john")**:
- Public: 0.60388 (3rd) → Private: 0.57773 (1st)
- **Drop**: -0.02615 (-4.3%)
- **Shake-up**: Gained 2 positions

**Interpretation**:
- Template-based approach (1st) generalized better to unseen data
- Deep learning ensemble (3rd) overfit to public leaderboard
- **Lesson**: TBM more robust, DL requires careful validation

---

### 2. Model Selection Matters More Than You Think

Both winners emphasized DRfold2's selection module weakness:
- **1st Place**: Enhanced selection with double precision, vectorized calculations
- **3rd Place**: Simplified to energy selection + Arena only
- **Original DRfold2 authors**: Acknowledged 5th-ranked models often outperform 1st

**Implication**: Post-processing and model ranking are critical research areas

---

### 3. The Computational Divide

**With GPU (3rd place)**:
- Fine-tune state-of-the-art models (Protenix)
- Generate custom rMSA (14 days)
- Train ensemble of 3 models
- **Result**: 0.54312

**Without GPU (1st place)**:
- Focus on algorithmic improvements
- Optimize existing tools
- Leverage PDB database intelligently
- **Result**: 0.57773 (BETTER!)

**Lesson**: Smart algorithms > brute-force compute (sometimes)

---

### 4. Community Collaboration

Both winners extensively acknowledged community contributions:

**Shared by 3rd Place**:
- @hengck23: discussions and guidance
- @geraseva: early Protenix inference code
- @lihaoweicvch: Protenix fine-tuning code
- @youhanlee: updated Boltz inference code
- @tant64: CASP16 dataset upload

**Shared by 1st Place**:
- @youhanlee: Boltz-1 integration
- @hengck23: research papers and models

**Implication**: Open science and collaboration accelerate progress

---

## Gaps and Limitations (Research Opportunities)

### Identified by Winners

**1st Place Limitations**:
1. ** DRfold2 model selection unreliable
2. ** No GPU access limited exploration
3. ** Runtime constraints forced simplifications
4. ** Template dependency (fails for novel folds)

**3rd Place Limitations**:
1. ** Memory constraints (<800nt for Protenix)
2. ** rMSA generation extremely time-consuming (14 days)
3. ** Fine-tuning without rMSA ineffective
4. ** Multi-stage refinement (Protenix→DRfold→Arena) didn't help
5. ** Overfitting to public leaderboard (-11.3% drop)

---

### Unexplored Areas (Novel Research Directions)

**1. Uncertainty Quantification**
- Winners provide 5 predictions but no confidence scores
- Which prediction is most reliable?
- Per-residue uncertainty maps?

**2. Active Learning for Template Selection**
- How to automatically decide: TBM vs. DL?
- Can we predict which approach will work better?

**3. Physics-Informed Constraints**
- Both approaches sometimes violate stereochemistry
- Incorporate quantum mechanics force fields?
- Enforce RNA-specific geometric rules?

**4. Conformational Ensembles**
- All methods predict single static structure
- RNA is dynamic (riboswitches, flexible loops)
- Diffusion models for ensemble generation?

**5. Ligand-Aware Prediction**
- Competition includes ligand-bound RNAs (PreQ1 riboswitch, apta-FRET)
- Current methods ignore ligands
- Predict apo vs. holo structures?

**6. Transfer Learning from Proteins**
- Limited RNA structural data
- Can AlphaFold/ESMFold priors help?
- Cross-modal transfer?

**7. Efficient rMSA Generation**
- 3rd place spent 14 days on rMSA
- Can we speed this up with:
  - Better algorithms?
  - Pretrained embeddings?
  - Approximation methods?

**8. Long Sequence Handling**
- 3rd place limited to <800nt (memory)
- Competition has sequences up to 720nt
- Sparse transformers? Chunking strategies?

**9. Interpretable Model Selection**
- Why does model ranking fail?
- Can we learn better scoring functions?
- Attention mechanisms for interpretability?

**10. Systematic Benchmarking**
- Which RNA types benefit from TBM?
- Which benefit from DL?
- Build decision tree for method selection?

---

## Benchmarking Strategy

### Phase 1: Reproduce 1st Place Solution

**Why start with 1st place?**
- Higher final score (0.57773 vs 0.54312)
- Better generalization (smaller LB drop)
- More accessible (no GPU required)
- Clearer methodology (TBM + DRfold2)

**Implementation Steps**:
1. Set up DRfold2 with enhanced modules
2. Process PDB_RNA database (93 nucleotide variants)
3. Implement TBM 5-step pipeline
4. Build hybrid decision logic
5. Evaluate on validation set

**Expected Timeline**: 2-3 weeks

---

### Phase 2: Reproduce 3rd Place Solution

**Why also reproduce 3rd place?**
- Access to shared datasets (rMSA, training labels)
- Multiple model ensemble approach
- Fine-tuning methodology
- Understanding overfitting

**Implementation Steps**:
1. Download shared rMSA and training data
2. Set up Protenix (requires GPU)
3. Fine-tune on RNA datasets
4. Integrate Boltz-1
5. Build ensemble pipeline

**Expected Timeline**: 3-4 weeks (GPU access required)

---

### Phase 3: Comparative Analysis

**Benchmarking Matrix**:
```
Method                  | TM-score | Runtime | GPU | Novel Folds | Conserved | Long Seq
------------------------|----------|---------|-----|-------------|-----------|----------
TBM (1st place)        |   ?      |   ?     | No  |     ?       |     ?     |    ?
DRfold2 (1st)          |   ?      |   ?     | ?   |     ?       |     ?     |    ?
Hybrid (1st)           | 0.57773  |   ?     | No  |     ?       |     ?     |    ?
DRfold2 (3rd)          |   ?      |   ?     | Yes |     ?       |     ?     |    ?
Protenix (3rd)         |   ?      |   ?     | Yes |     ?       |     ?     |    ?
Boltz-1 (3rd)          |   ?      |   ?     | Yes |     ?       |     ?     |    ?
Ensemble (3rd)         | 0.54312  |   ?     | Yes |     ?       |     ?     |    ?
```

**Analysis Questions**:
1. Error patterns: which sequences does each method fail on?
2. Sequence length impact: performance vs. length curve
3. RNA type dependency: ribozyme vs. riboswitch vs. tRNA
4. Template availability: performance with/without homologs
5. Computational cost: TM-score per GPU-hour
6. Diversity metrics: RMSD between 5 predictions

---

## Novel Research Contribution Plan

### Selected Research Direction: **Physics-Informed Model Selection**

**Motivation**:
- Both winners struggled with model selection/ranking
- DRfold2 authors acknowledged this weakness
- Current scoring functions (energy, geometry) insufficient

**Hypothesis**:
Physics-based validity metrics can improve model selection better than current energy functions

**Approach**:
```python
class PhysicsInformedSelector:
    def score_structure(self, pdb):
        # Geometry validation
        bond_score = self.validate_bond_lengths(pdb)       # C1'-C1' distances
        angle_score = self.validate_bond_angles(pdb)       # Backbone torsions
        clash_score = self.detect_steric_clashes(pdb)      # Van der Waals

        # RNA-specific constraints
        sugar_score = self.validate_sugar_pucker(pdb)      # C2'/C3'-endo
        base_score = self.validate_base_pairing(pdb)       # Watson-Crick geometry
        stack_score = self.validate_base_stacking(pdb)     # Aromatic stacking

        # Physical plausibility
        energy_score = self.compute_force_field_energy(pdb)  # AMBER/CHARMM
        solvation_score = self.estimate_solvation(pdb)       # Implicit solvent

        # Weighted combination
        total_score = (w1*bond_score + w2*angle_score + w3*clash_score +
                      w4*sugar_score + w5*base_score + w6*stack_score +
                      w7*energy_score + w8*solvation_score)

        return total_score

    def select_best_models(self, candidates, n=5):
        # Score all candidates
        scored = [(self.score_structure(pdb), pdb) for pdb in candidates]

        # Select diverse top-k
        selected = self.diverse_top_k(scored, n)

        return selected
```

**Training Data**:
- Experimental structures (PDB_RNA) as "correct" examples
- DRfold2/Protenix failures as "incorrect" examples
- Learn weights w1-w8 via supervised learning

**Validation**:
- Replace DRfold2's selection module with physics-informed version
- Evaluate on competition test set
- Measure TM-score improvement

**Expected Impact**:
- 5-10% improvement in DRfold2 performance
- Better generalization (reduce LB shake-up)
- Interpretable selection (understand why models are chosen)
- Publishable contribution (method + analysis)

---

## Next Steps

### Immediate (This Week)
1. ** Download shared datasets from 3rd place (rMSA, training labels)
2. ** Clone DRfold2 repository
3. ** Set up Python environment
4. ** Test DRfold2 installation
5. ** Convert test sequences to FASTA

### Short-term (Next 2 Weeks)
1. Implement TBM 5-step pipeline (1st place approach)
2. Process PDB_RNA database
3. Run DRfold2 baseline on test set
4. Evaluate TM-scores
5. Analyze error patterns

### Medium-term (Month 2-3)
1. Reproduce hybrid approach (1st place)
2. Set up Protenix + Boltz-1 (3rd place)
3. Build benchmarking dashboard
4. Identify systematic failure modes
5. Prototype physics-informed selector

### Long-term (Month 4-6)
1. Implement novel research contribution
2. Run ablation studies
3. Write paper
4. Release code + models
5. Submit to venue (Bioinformatics, Nature Methods, or NeurIPS)

---

## Resources & Acknowledgments

### Shared Datasets (3rd Place)
- **Custom rMSA**: [Google Drive](https://drive.google.com/drive/folders/example)
- **Training Data & Labels**: [Google Drive](https://drive.google.com/drive/folders/example)

### Code Repositories
- **DRfold2**: https://github.com/leeyang/DRfold2
- **Protenix**: https://github.com/protenix/protenix
- **Boltz**: https://github.com/boltz/boltz
- **US-align (TM-score)**: https://github.com/pylelab/USalign

### Community Contributors
- @hengck23: Research papers, models, discussions
- @geraseva: Protenix inference code
- @lihaoweicvch: Protenix fine-tuning code
- @youhanlee: Boltz-1 inference + DRfold2 integration
- @tant64: CASP16 dataset

### Papers
1. Li et al. (2025) - DRfold2: bioRxiv 2025.03.05.641632
2. He et al. (2024) - RibonanzaNet: bioRxiv 2024.02.24.581671

---

## Conclusion

**Key Takeaways**:

1. **Template-based methods still competitive** (1st place beat fine-tuned DL)
2. **Model selection is critical** (both winners emphasized this)
3. **Ensemble diversity matters** (best-of-5 metric)
4. **Generalization > Leaderboard climbing** (11% LB drop for 3rd place)
5. **Community collaboration accelerates progress** (extensive code sharing)

**Research Opportunities**:
- Improve model selection/ranking
- Quantify uncertainty
- Handle long sequences efficiently
- Incorporate physics constraints
- Predict conformational ensembles
- Develop ligand-aware methods

**Personal Strategy**:
1. Start with 1st place reproduction (more accessible)
2. Benchmark systematically
3. Focus on physics-informed selection (novel + feasible)
4. Publish results + code
5. Build portfolio for PhD applications

---

**Status**: Ready to begin implementation
**Next Update**: After DRfold2 baseline benchmark (Week 2)
