# CAFA-6 Protein Function Prediction - Complete Summary

**Date:** November 23, 2025  
**Project:** CAFA-6 (Critical Assessment of protein Function Annotation)  
**Current Best Score:** F1 = 0.2331

---

## 🎯 The Task

Predict **Gene Ontology (GO) terms** for protein sequences to determine:
1. **What the protein does** (Molecular Function - MF)
2. **Which biological processes it participates in** (Biological Process - BP)
3. **Where in the cell it operates** (Cellular Component - CC)

This is a **multi-label classification** problem with extreme class imbalance:
- Input: Raw amino acid sequences (e.g., `MVLSPADKT...`)
- Output: Multiple GO terms per protein (average 6 out of 26,125 possible terms)
- Evaluation: Weighted F1-score (weighted by information accretion)

---

## 📁 Input Files Provided

### 1. Training Data

#### `Train/train_sequences.fasta` (82,404 proteins)
```
>sp|A0A0C5B5G6|MOTSC_HUMAN Mitochondrial-derived peptide MOTS-c OS=Homo sapiens OX=9606 GN=MT-RNR1 PE=1 SV=1
MRWQEMGYIFYPRKLR
>sp|A0JNW5|BLT3B_HUMAN Bridge-like lipid transfer protein family member 3B OS=Homo sapiens OX=9606 GN=BLTP3B PE=1 SV=2
MAGIIKKQILKHLSRFTKNLSPDKINLSTLKGEGELKNLELDEEVLQNMLDLPTWLAINK...
```
- 82,404 protein sequences
- Length range: 16 - 34,350 amino acids
- Average length: 612 amino acids

#### `Train/train_terms.tsv` (537,027 annotations)
```
EntryID	term	aspect
Q5W0B1	GO:0000785	C
Q5W0B1	GO:0004842	F
Q5W0B1	GO:0051865	P
```
- Column 1: Protein ID
- Column 2: GO term ID
- Column 3: Ontology aspect (C=Cellular, F=Function, P=Process)
- Average: 6.52 terms per protein

#### `Train/go-basic.obo` (26,125 GO terms)
```
format-version: 1.2
data-version: releases/2025-06-01
...
[Term]
id: GO:0000001
name: mitochondrion inheritance
namespace: biological_process
is_a: GO:0048308 ! organelle inheritance
```
- Ontology structure (directed acyclic graph)
- Defines term hierarchy and relationships
- Used for label propagation

#### `Train/train_taxonomy.tsv`
- Maps proteins to species (taxon IDs)
- Covers eukaryotes, 13 bacteria, 1 archaea

#### `IA.tsv`
- Information Accretion weights for each GO term
- Used to calculate weighted precision/recall

### 2. Test Data

#### `Test/testsuperset.fasta` (Test proteins)
```
>A0A0C5B5G6 9606
MRWQEMGYIFYPRKLR
>A0A1B0GTW7 9606
MLLLLLLLLLLPPLVLRVAASRCLHDETQKSVSLLRPPFSQLPSKSRSSSLTLPSSRDPQ...
```
- Proteins on which predictions must be made
- Format: `>ProteinID TaxonID`

#### `Test/testsuperset-taxon-list.tsv`
- List of species in test set

#### `sample_submission.tsv`
- Example submission format
- Tab-separated: `ProteinID	GO_term	confidence`

---

## 🚀 Our Approach & Progress

### Phase 1: Data Infrastructure ✅
**Built:** `src/data/loaders.py`
- `SequenceLoader`: Parses FASTA files
- `LabelLoader`: Loads GO term annotations
- `OntologyLoader`: Handles GO graph structure

### Phase 2: Baseline Models (Week 3)

#### Baseline 1: Frequency Predictor ✅
**File:** `src/models/baseline_frequency.py`

**Strategy:** Predict the most common GO terms for every protein (ignore sequence).

**Result:** F1 = **0.1412**

**Key Insight:** Better than random (0.00), but completely ignores protein identity.

---

#### Baseline 2: Embedding KNN ("Neural BLAST") ✅
**File:** `src/models/baseline_embedding_knn.py`

**Strategy:**
1. Generate ESM-2 embeddings (320-dim vectors) for all training proteins
2. For a new protein, find K=5 nearest neighbors
3. Copy their GO term labels

**Model:** `facebook/esm2_t6_8M_UR50D` (8M parameters)

**Result:** F1 = **0.1776**

**Training Time:** ~3 hours (CPU)

**Key Insight:** Similarity-based approach works well. ESM-2 pre-training captures protein structure knowledge.

---

#### Baseline 3: MLP on Frozen Embeddings ✅
**Files:** `src/models/architecture.py`, `src/training/trainer.py`

**Strategy:**
1. Generate ESM-2 embeddings (frozen)
2. Train 2-layer neural network (512 → 256 → 2000 GO terms)
3. Use BCEWithLogitsLoss

**Result:** F1 = **0.1672** (Threshold 0.10)

**Key Insight:** Underperformed KNN. Frozen embeddings may lack task-specific signal.

---

### Phase 3: ESM-2 Fine-Tuning (Weeks 4-5)

#### Attempt 1: Fine-Tuning with BCE Loss ✅
**Files:** `src/data/finetune_dataset.py`, `src/models/esm_classifier.py`, `src/training/finetune_esm.py`

**Strategy:**
1. Tokenize raw sequences (no frozen embeddings)
2. Fine-tune all layers of ESM-2
3. Add classification head (320 → 5000 GO terms)
4. Train with BCEWithLogitsLoss

**Initial Problem:** F1 = **0.0000** (using fixed threshold 0.5)

**Solution:** Implemented adaptive threshold tuning (try 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)

**Result:** F1 = **0.1806** (Threshold 0.10)

**Training Time:** ~3 hours (GPU, 4 epochs, early stopping)

**Key Insight:** Class imbalance (5000 terms, ~6 positives/protein) requires adaptive thresholding.

---

#### Attempt 2: Fine-Tuning with Asymmetric Loss ✅ **[CURRENT BEST]**
**Files:** `src/training/loss.py` (updated), `src/training/finetune_esm.py` (updated)

**Strategy:**
1. Replace BCEWithLogitsLoss with AsymmetricLoss
2. Hyperparameters:
   - `gamma_neg=2.0`: Down-weight easy negatives (the 4994 terms protein doesn't have)
   - `gamma_pos=1.0`: Focus on hard positives (the 6 terms it does have)
   - `clip=0.05`: Asymmetric clipping

**Bug Fixed:** Loss was returning `.sum()` instead of `.mean()` → Loss exploded to 446

**Result:** F1 = **0.2331** (Threshold 0.40)

**Metrics:**
- Precision: 0.3397
- Recall: 0.2379
- Training Loss: 0.0018 (stable, no overfitting)

**Training Time:** ~7.5 hours (GPU, 10 epochs)

**Improvement:**
- vs. Frequency: +65.1%
- vs. KNN: +31.3%
- vs. BCE Fine-Tuning: +29.1%

**Key Insight:** Asymmetric Loss is critical for extreme class imbalance. Standard BCE wastes capacity learning "protein is NOT ribosome" when it should focus on "protein IS kinase."

---

## 📊 Model Comparison

| Model | F1 Score | Threshold | Training Time | Status |
|-------|----------|-----------|---------------|--------|
| Frequency Baseline | 0.1412 | N/A | < 1 min | ✅ |
| Embedding KNN | 0.1776 | N/A | ~3 hours (CPU) | ✅ |
| MLP (Frozen Embeddings) | 0.1672 | 0.10 | ~3 mins (GPU) | ✅ |
| **ESM-2 Fine-Tuned (BCE)** | 0.1806 | 0.10 | ~3 hours (GPU) | ✅ |
| **ESM-2 Fine-Tuned (Asym Loss)** | **0.2331** | 0.40 | ~7.5 hours (GPU) | 🥇 **BEST** |

**Winner:** ESM-2 Fine-Tuned with Asymmetric Loss

---

## 🐛 Key Problems Solved

### Problem 1: F1 = 0.0000 with Fixed Threshold
**Cause:** With 5000 classes and only ~6 positives per sample, threshold 0.5 was too high. Model predicted nothing.

**Solution:** Implemented adaptive threshold tuning (try 8 thresholds, pick best F1).

**Impact:** F1 jumped from 0.00 → 0.18

---

### Problem 2: Class Imbalance
**Cause:** BCE treats all 4994 negatives equally, wasting capacity on obvious negatives.

**Solution:** AsymmetricLoss with `gamma_neg=2.0` down-weights easy negatives.

**Impact:** F1 improved from 0.18 → 0.23 (+29%)

---

### Problem 3: Loss Explosion
**Cause:** AsymmetricLoss was returning `.sum()` instead of `.mean()` → loss=446.

**Solution:** Changed return statement to `.mean()`.

**Impact:** Training stabilized.

---

## 💡 Key Insights

### What Worked Well:
1. **Asymmetric Loss** → +29% improvement over BCE
2. **Adaptive threshold tuning** → Critical for imbalanced data
3. **Fine-tuning ESM-2** → Beats frozen embeddings + MLP
4. **Early stopping** → Prevented overfitting (best epoch often early)

### What Didn't Work:
1. **Fixed threshold 0.5** → Too high for imbalanced data
2. **Standard BCE loss** → Treats all negatives equally
3. **Frozen embeddings** → Limited capacity to adapt to task

### What We'd Do Differently:
1. **Start with larger model** → ESM-2 35M or 150M (current: 8M)
2. **Use more GO terms** → 10k-15k instead of 5k
3. **Implement label propagation** → Use ontology structure in loss

### Technical Learnings:
- **Class imbalance requires specialized losses** (not just reweighting)
- **Threshold optimization is not optional** (especially for multi-label)
- **Fine-tuning > Transfer Learning > Feature Engineering**
- **GPU time well spent** (7.5 hours → 31% improvement)

---

## 🎯 Current Status

### Completed ✅
- Data loading pipeline
- 3 baseline models (Frequency, KNN, MLP)
- ESM-2 fine-tuning infrastructure
- Adaptive threshold tuning
- Asymmetric Loss implementation
- GPU training (RTX 2070)
- Model saved: `models/esm_finetuned/best_model/`

### In Progress 🔄
- Generating test predictions

### Not Started ⬜
- Label propagation (ontology-aware predictions)
- Ensemble (KNN + Fine-Tuned model)
- Larger model (ESM-2 35M/150M)
- Longer sequences (1024 tokens vs 512)
- More GO terms (10k-15k vs 5k)

---

## 📈 Performance Trajectory

```
Frequency:           ████████████████ 0.1412
KNN:                 ██████████████████ 0.1776
MLP (Frozen):        ████████████████▌ 0.1672
ESM-2 (BCE):         ██████████████████▏ 0.1806
ESM-2 (Asym Loss):   ████████████████████████ 0.2331 ← CURRENT
```

**Target for Competitive Performance:** F1 > 0.30

---

## 🚀 Next Steps (Potential Improvements)

### Option 1: Ensemble (Quick Win)
- Combine ESM-2 + KNN predictions
- Expected: F1 → 0.24-0.25
- Time: 30 minutes

### Option 2: Larger Model
- Switch to ESM-2 35M or 150M
- Expected: F1 → 0.26-0.30
- Time: 12-24 hours

### Option 3: More Terms + Longer Sequences
- Train on 10k-15k terms
- Use 1024-token sequences
- Expected: F1 → 0.25-0.28
- Time: 10-15 hours

### Option 4: Label Propagation
- Use ontology structure during prediction
- Propagate predictions to ancestors
- Expected: F1 → +0.01-0.02
- Time: 2 hours

---

## 📂 Key Files

### Data Loading
- `src/data/loaders.py`: FASTA, TSV, OBO parsers
- `src/data/finetune_dataset.py`: PyTorch Dataset with tokenization

### Models
- `src/models/baseline_frequency.py`: Frequency predictor
- `src/models/baseline_embedding_knn.py`: KNN with ESM-2 embeddings
- `src/models/architecture.py`: MLP classifier
- `src/models/esm_classifier.py`: ESM-2 with classification head

### Training
- `src/training/trainer.py`: MLP training loop
- `src/training/finetune_esm.py`: ESM-2 fine-tuning script
- `src/training/loss.py`: AsymmetricLoss implementation

### Tracking
- `PROGRESS_TRACKER.md`: Detailed progress log
- `docs/milestones/1.md`: Project explanation with analogies
- `notebooks/esm_finetuning.ipynb`: Step-by-step notebook

---

## 🏆 Achievement Summary

- ✅ Built end-to-end prediction pipeline
- ✅ Trained 5 different models
- ✅ Achieved 65% improvement over naive baseline
- ✅ Solved threshold optimization problem
- ✅ Solved class imbalance problem
- ✅ GPU-accelerated training operational
- ✅ Model saved and ready for test predictions

**Current Ranking:** Top third of typical CAFA performance (F1 0.23 is competitive but not top-tier)

**To Reach Top Tier (F1 > 0.35):** Need ensemble + larger model + more terms

---

**Project Duration:** 4 days  
**GPU Hours:** ~11 hours  
**Lines of Code:** ~2,500  
**Models Trained:** 5  
**Best F1:** 0.2331
