# Phase 1: Feature Engineering and Data Scale

Phase 1 of the Rank 1 winning strategy focuses on creating a foundation of multi-modal, large-scale, and hierarchy-aware inputs. This phase includes crucial setup, exhaustive data collection, and state-of-the-art feature generation necessary to maximize the Information Accretion (IA) weighted F1 score.

## Step 1: Environment and Core Data Setup

Before any processing can begin, the appropriate computational environments must be created and the foundational data loaded.

### Environment Creation
Create two specific Python environments:
- **pytorch-env**: Handles all Deep Learning (DL) models and sequence embeddings
- **rapids-env**: For preprocessing, Py-Boost (GBDT), and Logistic Regression training

### Configuration and Input Data
Load the configuration file (`config.yaml`) and ensure all necessary input data are present:
- CAFA training and test data
- Sample submission file
- **Critical**: Information Accretion weights (`IA.tsv`)

---

## Step 2: Input Data Structuring and Prior Calculation

The raw FASTA and target files are converted into machine-readable formats, and priors are calculated to prepare for IA weighting.

### Parse and Feather Sequences
Parse FASTA files containing training and test sequences and save in `.feather` format:
- `train_seq.feather`
- `test_seq.feather`

### Convert Targets and Calculate Priors
1. Convert ground truth targets into Parquet format (`real_targets`)
2. Calculate helper data:
   - Prior means for each Gene Ontology (GO) term (excluding NaN cells)
   - NaN rate for each term

### IA Weight Utilization
The IA weights (`IA.tsv`) determine the competition's scoring metric:
- Heavily rewards rare, specific GO terms
- Must be loaded and utilized as class weights during base model training
- Prioritizes learning difficult, high-value terms in DNNs using `class_weight` argument

---

## Step 3: External Data Acquisition and Evidence Code Separation

Enhance the model's knowledge beyond the provided CAFA training set by integrating external annotations.

### External Data Download
Download external annotation datasets from:
```
http://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/
```

Example: `goa_uniprot_all.gaf.216.gz`

### Evidence Code Split (Kaggle/No-Kaggle)
Parse datasets and separate based on evidence codes:

| Category | Description | Usage |
|----------|-------------|-------|
| **Kaggle** | Experimental Codes | Standard training labels |
| **No-Kaggle** | Electronic Labeling | Powerful features for stacker models (`prop_train_no_kaggle.tsv`, `prop_test_no_kaggle.tsv`) |

---

## Step 4: Hierarchy Propagation in Preprocessing

Apply GO hierarchy rules to maximize the reach and accuracy of external labels.

### Propagate External Labels
Apply propagation rules using the Gene Ontology hierarchy file (`go-basic.obo`).

### Purpose
Ensure every specific term (child) is associated with all its corresponding general parent terms (ancestors) in the hierarchy graph. This preparation is critical because the evaluation metric inherently relies on the hierarchical structure of GO.

---

## Step 5: Multi-Modal Feature Generation (Embeddings)

The model captures complex sequence information using high-dimensional vectors from large, pre-trained language models, alongside ancillary biological data.

### Feature Modalities

| Embedding Type | Source | Dimension | Purpose |
|---|---|---|---|
| **T5 Embeddings** | T5 Language Model | 1024 | Core sequence representation |
| **ESM2 Small** | ESM2 Family | 1280 | Additional protein language signal |
| **ESM2 Large (3B)** | ESM2 3 Billion Parameter Model | 2560 | Enhanced brute-force feature scaling |
| **Ankh Embeddings** | Ankh Model | 1536 | Alternative protein representation |
| **Taxonomic Data** | Taxon ID Matrix | 70 | Evolutionary context (e.g., 9606 for *Homo sapiens*) |
| **Text Abstract Embeddings** | Literature Embeddings | 10279 | Multimodal input from research knowledge |

### Generation Process
1. T5 embeddings generated as core sequence features
2. ESM2 embeddings (both small and large versions) for brute-force scaling
3. Ankh embeddings for additional vector representation
4. Taxonomic IDs loaded to provide evolutionary context
5. Text abstract embeddings integrated for truly multimodal input

---

## End of Phase 1 Output

By the end of Phase 1, the system has:
- Multiple high-dimensional embedding vectors for every protein (T5, ESM, Ankh, Taxa, Text)
- Structured and IA-weighted training labels
- External annotations with hierarchy propagation
- Ready-to-use feature matrices for Level 1 model training

This comprehensive foundation sets the stage for the Level 1 and Level 2 ensemble models.

---

⚠️ **Note**: NotebookLM can be inaccurate; please double-check all responses.






---

# Phase 2: Diverse Base Model Training (Level 1)

Phase 2 involves training the foundational Level 1 models, which convert massive, high-dimensional feature inputs from Phase 1 into initial probability predictions. This phase is characterized by:
- **Diversity**: GBDT, Linear, and Deep Learning models
- **Resource Intensity**: Requires 32 GB GPU RAM per model

## Primary Goal

Generate robust **Out-Of-Fold (OOF) predictions**, which will serve as structural features for the final stacking layer (Phase 3).

---

## Step 1: Training Gradient Boosted Decision Tree (GBDT) Models (Py-Boost)

GBDTs, using the Py-Boost framework, excel at capturing interactions between T5, ESM, and Taxon features.

### GBDT Model Sets

| Model Set | Features Used | Target Type | Output Term Count | Training Duration |
|-----------|---|---|---|---|
| **GBDT Set 1** | T5 + Taxon | Multilabel (Raw) | 4,500 (BP 3000, MF 1000, CC 500) | ~15 hours per 5-fold CV |
| **GBDT Set 2** | T5 + ESM + Taxon | Conditional | 4,500 (BP 3000, MF 1000, CC 500) | ~15 hours per 5-fold CV |
| **GBDT Set 3** | T5 + ESM + Taxon | Multilabel (Raw) | 4,500 (BP 3000, MF 1000, CC 500) | ~15 hours per 5-fold CV |
| **GBDT Set 4** | T5 + Taxon | Conditional | 4,500 (BP 3000, MF 1000, CC 500) | ~15 hours per 5-fold CV |

### Key Points

- **Computational Requirement**: 32 GB GPU RAM (e.g., V100 card) to prevent Out-of-Memory errors
- **Parallel Execution**: All four GBDT models can run in parallel
- **Output Storage**: For each model:
  - Final fitted models: `models_0.pkl` to `models_4.pkl`
  - OOF predictions: `oof_pred.pkl`
  - Test predictions: `test_pred.pkl`

---

## Step 2: Training Logistic Regression (LogReg) Models

LogReg models introduce additional ensemble diversity, targeting larger GO term sets than GBDTs.

### LogReg Model Sets

| Model Set | Features Used | Target Type | Output Term Count | Training Duration |
|-----------|---|---|---|---|
| **LogReg Set 1** | T5 + Taxon | Multilabel (Raw) | 13,500 (BP 10000, MF 2000, CC 1500) | ~10 hours per 5-fold CV |
| **LogReg Set 2** | T5 + Taxon | Conditional | 4,500 (BP 3000, MF 1000, CC 500) | ~2 hours per 5-fold CV |

### Key Points

- **Computational Requirement**: 32 GB GPU RAM (rapids-env)
- **Purpose**: Essential Level 1 predictions that combine with GBDT outputs for Level 2 GCN stacker
- **Output Format**: Same as GBDT (models, OOF, and test predictions)

---

## Step 3: Training Deep Neural Network (DNN) Models (Brute Force Blending)

DNN strategy leverages all available embeddings (seven separate feature inputs) with complex architecture for maximum blending and variance reduction.

### Feature Ingestion: Seven Input Layers

| Input Layer | Source | Dimension |
|---|---|---|
| **Input 1** | T5 Embeddings | 1024 |
| **Input 2** | ESM2 Small Embeddings | 1280 |
| **Input 3** | ESM2 Large (3B) Embeddings | 2560 |
| **Input 4** | PB Embeddings | 1024 |
| **Input 5** | Ankh Embeddings | 1536 |
| **Input 6** | Taxonomic Data | 70 |
| **Input 7** | Text Abstract Embeddings | 10279 |

### Network Architecture

1. **Separate Input Processing**: Each input processed through:
   - Dense layers
   - Batch Normalization
   - LeakyReLU activation
   - Dropout

2. **Concatenation**: Outputs from seven processing blocks combined via `Concatenate()`

3. **Final Layers**: Concatenated features passed through:
   - Dense layers
   - Output layer with sigmoid activation

### IA-Weighted Training

- **Loss Function**: Binary crossentropy
- **Class Weights**: Information Accretion (IA) weights
- **Purpose**: Prioritizes learning rare, high-value GO terms aligned with competition's F1 metric

### Extreme Ensembling (Variance Reduction)

Employ K-Fold Cross-Validation (5-fold) across multiple random states (5 states):

$$\text{Total Models} = 5 \text{ CV folds} \times 5 \text{ random states} = 25 \text{ models}$$

**Benefits**:
- Stabilizes predictions
- Minimizes noise
- Boosts generalization
- Final predictions averaged across all 25 models

---

## Summary: Phase 2 Outputs

Upon completion of Phase 2, the system produces comprehensive OOF and test predictions:

### Output Matrices

| Model Type | Prediction Matrices | Term Coverage |
|---|---|---|
| **Py-Boost GBDTs** | 4 model sets (multilabel + conditional) | 4,500 terms |
| **Logistic Regression** | 2 model sets (multilabel + conditional) | Up to 13,500 terms |
| **DNN Ensemble** | 25 averaged models | Up to 1,500 terms per ontology |

### Feature Replacement

These numerical predictions **replace raw embeddings** as input features for Phase 3 (Hierarchy-Aware Stacking).

---

**Next Phase**: These diverse Level 1 predictions feed directly into the GCN stacker for hierarchy-aware refinement.


---

# Phase 3: Hierarchy-Aware Stacking (Level 2 - GCNs)

Phase 3 is the **Hierarchy-Aware Stacking** phase—the core of the winning strategy. It moves beyond generating raw statistical probabilities (Phase 2) and uses Graph Convolutional Networks (GCNs) to impose structure and biological logic onto those predictions before final constraint enforcement.

## Overview

This phase requires:
- Integrating outputs from Level 1 base models (GBDTs, LogRegs, DNNs)
- Leveraging massive computational resources (32 GB GPU RAM)
- Training three ontology-specific graph-based models

---

## Step 1: Feature Transformation and Input Preparation

The inputs for Phase 3 are **no longer raw sequence embeddings** (T5, ESM, Ankh, Text), but the Out-Of-Fold (OOF) predictions generated in Phase 2.

### Consolidate Level 1 Predictions

Aggregate various Level 1 predictions into a unified feature set:

| Source | Coverage | Description |
|--------|----------|-------------|
| **Py-Boost GBDT OOFs** | 4,500 terms | Raw and conditional targets |
| **Logistic Regression OOFs** | Up to 13,500 terms | Large-scale predictions |
| **DNN Averaged OOFs** | Variable | 25 model average (if saved) |

### Define GCN Input

These OOF predictions serve as numerical features for the GCN to process:
- **GBDTs/LogRegs**: Perform statistical feature extraction
- **GCNs**: Structure those statistical features according to GO hierarchy

**Key Design**: Feature transformation rather than raw embedding input.

---

## Step 2: GCN Training and Ontology Specialization

Train **three independent GCN models**, one for each GO sub-ontology.

### Ontology Separation

| Ontology | Abbreviation | Hierarchy Complexity | Training Time |
|----------|---|---|---|
| **Biological Process** | BP | High | ~13 hours |
| **Molecular Function** | MF | Medium | ~4 hours |
| **Cellular Component** | CC | Low | ~2 hours |

### Graph Structure Integration

GCNs are adept at understanding GO hierarchy structure:
- Incorporate topological relationships directly (parent-child, is_a, part_of)
- Learn graph structure through network architecture
- Enable hierarchy-aware predictions from statistical features

### Resource Allocation

- **GPU RAM Required**: 32 GB per model
- **Parallel Execution**: Train BP on GPU 1, MF/CC on GPU 2 simultaneously
- **Total Time (Parallel)**: ~13 hours (bottleneck is BP)

---

## Step 3: Hierarchy Constraint Learning (Comparative Insight)

### Alternative Approach: 4th Place Strategy

The lower-ranked team attempted a different philosophy:

| Aspect | Approach |
|--------|----------|
| **Constraint Integration** | Baked directly into training via custom loss function |
| **Loss Function** | Custom MCM (Maximum Common Ancestor Loss) |
| **Implementation** | R matrix (parent-child relationship mapper) penalizes hierarchy violations |
| **Outcome** | Less effective than post-processing approach |

### Winning Strategy: Phase 3 vs Phase 4 Separation

**R1 Plan Philosophy**:
1. **Phase 3 (GCN)**: Statistically structure predictions based on graph relationships
2. **Phase 4 (Post-Processing)**: Guarantee hierarchical adherence via strict rules

**Why This Wins**:
- Separation of concerns (statistics vs. rules)
- Allows Level 1 & 2 models to focus on prediction quality
- Strict enforcement in post-processing proved more robust than constraint-aware training
- Top-performing teams adopted this modular approach

---

## Step 4: GCN Inference and Test-Time Augmentation (TTA)

After GCN training, generate final predictions on test set with stability-maximizing methods.

### Inference Phase
Run inference on test data using the three trained GCN models (one per ontology).

### Test-Time Augmentation (TTA)

Apply TTA during inference to generate multiple distinct prediction outputs:

| Augmentation | Output File | Purpose |
|---|---|---|
| **TTA Set 0** | `pred_tta_0.tsv` | Augmented inference run 1 |
| **TTA Set 1** | `pred_tta_1.tsv` | Augmented inference run 2 |
| **TTA Set 2** | `pred_tta_2.tsv` | Augmented inference run 3 |
| **TTA Set 3** | `pred_tta_3.tsv` | Augmented inference run 4 |

**Benefits**:
- Dampens noise in individual predictions
- Boosts generalization through averaging
- Stabilizes predictions across random variations

### TTA Aggregation

Multiple TTA predictions aggregated by averaging:

$$\text{pred.tsv} = \frac{1}{4}(\text{pred\_tta\_0} + \text{pred\_tta\_1} + \text{pred\_tta\_2} + \text{pred\_tta\_3})$$

**Output**: `pred.tsv` — final statistical probabilities before strict biological constraints (Phase 4).

---

## Summary: Phase 3 Outputs

- Three trained GCN models (BP, MF, CC)
- Multiple TTA prediction matrices (pred_tta_0 to pred_tta_3)
- Averaged GCN predictions (pred.tsv)
- Ready for Phase 4 constraint enforcement

**Next Phase**: Strict post-processing with Max/Min propagation to guarantee biological validity.


---

# Phase 4: Strict Post-Processing and Submission

Phase 4 is **critical**—all complex feature engineering (Phase 1) and ensemble modeling (Phases 2–3) are meaningless if the final output violates the basic biological rules of the Gene Ontology (GO) hierarchy or competition formatting requirements.

## Overview

The winning strategy enforces biological and formatting constraints **strictly after prediction**, using explicit post-hoc rules rather than attempting to encode constraints during training.

---

## Step 1: Hierarchy Enforcement (Max Propagation / Parent Rule)
Apply the **first critical constraint**: upward consistency in the GO hierarchy.

### Rule Definition

**Max Propagation** enforces upward constraint enforcement:

> If a specific, detailed child term is predicted, all parent terms (ancestors) must be assigned a score **at least as high** as the child's score.

### Example

```
Child Term:   "DNA ligase activity" (score: 0.85)
  ↓ Must trigger
Parent Terms:
  - "catalytic activity" (score: ≥ 0.85)
  - "ligase activity" (score: ≥ 0.85)
  - "molecular function" (score: ≥ 0.85)
```

### Goal

Ensure **upward consistency**—if a protein performs a specific function, it must also possess the more general functions encompassing it.

### Output

**`pred_max.tsv`** — upward-enforced predictions

---

## Step 2: Hierarchy Enforcement (Min Propagation / Child Rule)
Apply the **second critical constraint**: downward consistency, preventing overly confident specific predictions.

### Rule Definition

**Min Propagation** enforces downward constraint enforcement:

> If a general parent term has a low prediction score, its specific child terms cannot have a confidence score **higher than that parent**.

### Example

```
Parent Term:  "protein binding" (score: 0.15)
  ↓ Constrains
Child Terms:
  - "ATP binding" (score: ≤ 0.15)
  - "zinc ion binding" (score: ≤ 0.15)
  - "DNA binding" (score: ≤ 0.15)
```

### Goal

Prevent highly specific predictions when the broader category is uncertain. A protein cannot be confidently predicted to have a specific function if the general class prediction is weak.

### Output

**`pred_min.tsv`** — downward-enforced predictions

---

## Step 3: Final Submission Generation
Enforced predictions are aggregated and validated against competition rules.

### 3.1 Averaging Enforced Outputs

Combine Max and Min propagation results:

$$\text{final\_scores} = \frac{1}{2}(\text{pred\_max} + \text{pred\_min})$$

### 3.2 Score Formatting

Validate strict adherence to required format:

| Requirement | Rule | Example |
|---|---|---|
| **Score Range** | (0, 1.000] | ✅ 0.543, ✅ 1.000 |
| **Zero Scores** | NOT listed | ❌ 0.000 |
| **Precision** | Up to 3 decimals | ✅ 0.123, ❌ 0.1234 |

### 3.3 Prediction Limit Enforcement

Limit total predicted GO terms **per protein** across all three ontologies (BP, MF, CC):

| Constraint | Limit |
|---|---|
| **Maximum GO terms per protein** | 1,500 |
| **Implementation** | Sort by score, take top-1,500 terms |

### Output

**`submission.tsv`** — final, validated submission file

---

## Step 4: Final Constraint Verification Checklist
Before submission, validate all requirements:

### Format Validation
- ✅ All scores in (0, 1.000]
- ✅ No zero-valued predictions listed
- ✅ Correct TSV format with protein ID and GO term columns
- ✅ Proper header row

### Biological Validation
- ✅ Max Propagation applied (child ≤ parent upward)
- ✅ Min Propagation applied (child ≤ parent downward)
- ✅ No hierarchy violations detected

### Size Validation
- ✅ Maximum 1,500 terms per protein
- ✅ No duplicate protein-term pairs
- ✅ File size reasonable (<1 GB typically)

---

## Key Insight: Why Post-Processing Wins

### The Philosophy

This two-stage enforcement approach (Max Prop → Min Prop) proved more effective than attempting to "bake in" hierarchy constraints during training:

| Strategy | When | How | Result |
|----------|------|-----|--------|
| **During Training** | 4th Place | Custom MCM loss + R matrix | ❌ Less effective |
| **After Prediction** | 1st Place (Ours) | Explicit Min/Max propagation | ✅ More robust |

### Why Separation of Concerns Wins

1. **Prediction Phase**: Level 1 & 2 models focus purely on generating strongest statistical probabilities
2. **Enforcement Phase**: Strict rules guarantee biological validity
3. **Flexibility**: Constraints can be refined post-hoc without retraining
4. **Robustness**: Clear, interpretable rules vs. learned soft constraints

### CAFA 5 Lesson

Strict enforcement in post-processing proved **more robust and effective** than constraint-aware training. Top-performing teams adopted this modular philosophy.

---

## Summary: Phase 4 Outputs

| Output | Description |
|--------|-------------|
| **pred_max.tsv** | Upward-enforced predictions |
| **pred_min.tsv** | Downward-enforced predictions |
| **submission.tsv** | Final, validated submission file |

**Key Achievement**: Guaranteed biological validity while maintaining statistical quality from Phases 1–3.

---

## End-to-End Pipeline Complete

✅ **Phase 1**: Multi-modal feature engineering  
✅ **Phase 2**: Diverse base model training (Level 1)  
✅ **Phase 3**: Hierarchy-aware GCN stacking (Level 2)  
✅ **Phase 4**: Strict post-processing & submission  

**Result**: Competition-ready, biologically valid, high-performance protein function predictions.