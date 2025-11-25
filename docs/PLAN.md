# 🎯 CAFA-6 Protein Function Prediction - BUILD PLAN

**Last Updated:** November 25, 2025  
**Current Status:** Phase 2 Complete - KNN with aspect-specific thresholds (F1 = 0.2579)

---

## 📋 Table of Contents
1. [✅ Completed Milestones](#1--completed-milestones)
2. [🎯 Current Focus: Production Improvements](#2--current-focus-production-improvements)
3. [🚀 Next Steps: Quick Wins](#3--next-steps-quick-wins)
4. [🔮 Future Work: Advanced Features](#4--future-work-advanced-features)
5. [📤 Submission Pipeline](#5--submission-pipeline)
6. [📊 Performance Tracking](#6--performance-tracking)
7. [📚 Technical Resources](#7--technical-resources)

---

## 1. ✅ Completed Milestones

### Phase 0: Project Setup ✅
**Status:** Complete  
**Date:** November 20, 2025

- ✅ Virtual environment created (`C:\venv\cafa6\`)
- ✅ Dependencies installed (PyTorch 2.5.1 + CUDA 12.1, Transformers, BioPython, etc.)
- ✅ Project structure organised
- ✅ Git repository initialised
- ✅ Data files verified (82,404 training sequences, 537,027 annotations)

### Phase 1: Data Infrastructure & EDA ✅
**Status:** Complete  
**Date:** November 20-21, 2025

**Files Created:**
- `src/data/loaders.py` - Data loading classes
- `notebooks/eda.ipynb` - Exploratory data analysis

**Key Findings:**
- 82,404 proteins, average 6.52 GO terms each
- 26,125 total GO terms (extreme class imbalance)
- Sequence lengths: 16-34,350 amino acids (avg 612)
- Top 5000 GO terms cover 92% of annotations

**Deliverables:**
- ✅ SequenceLoader (FASTA parsing)
- ✅ LabelLoader (TSV annotations)
- ✅ OntologyLoader (GO graph structure)
- ✅ Dataset statistics documented

### Phase 2: Baseline Models ✅
**Status:** Complete  
**Date:** November 21-22, 2025

| Model | F1 Score | Status | Training Time |
|-------|----------|--------|---------------|
| Frequency Baseline | 0.1412 | ✅ | < 1 min |
| Embedding KNN | 0.1776 | ✅ | ~3 hours (CPU) |
| MLP (Frozen Embeddings) | 0.1672 | ✅ | ~3 mins (GPU) |

**Files Created:**
- `src/models/baseline_frequency.py`
- `src/models/baseline_embedding_knn.py`
- `src/models/architecture.py` (MLP)
- `src/training/trainer.py`

**Key Insight:** KNN with ESM-2 embeddings (F1=0.1776) established strong baseline.

### Phase 3: ESM-2 Fine-Tuning ✅
**Status:** Complete  
**Date:** November 22-23, 2025

**Attempt 1: BCE Loss** (F1 = 0.1806)
- ✅ Built fine-tuning pipeline
- ✅ Fixed threshold bug (F1 0.00 → 0.18)
- ✅ Implemented adaptive threshold tuning

**Attempt 2: Asymmetric Loss** (F1 = 0.2331) 🏆
- ✅ Integrated Asymmetric Loss (gamma_neg=2.0)
- ✅ Fixed loss scale bug (.sum → .mean)
- ✅ Achieved **+31% improvement** over BCE
- ✅ Model saved to `models/esm_finetuned/best_model/`

**Files Created:**
- `src/data/finetune_dataset.py` - PyTorch Dataset with tokenisation
- `src/models/esm_classifier.py` - ESM-2 with classification head
- `src/training/finetune_esm.py` - Full training loop
- `src/training/loss.py` - AsymmetricLoss implementation

**Training Configuration:**
- Model: ESM-2 Tiny (8M parameters)
- Batch size: 8 (effective 32 with gradient accumulation)
- Learning rate: 2e-5 with warmup
- Loss: AsymmetricLoss (gamma_neg=2.0, clip=0.05)
- Training time: ~7.5 hours (RTX 2070)

**Current Best:** F1 = 0.2331 (Precision: 0.3397, Recall: 0.2379, Threshold: 0.40)

---

## 2. 🎯 Current Focus: Production Improvements

**Goal:** Push F1 from 0.2331 → 0.30+ using proven techniques

**CRITICAL UPDATE (Nov 23):** Discovered that competition uses **per-aspect evaluation** (MF, BP, CC computed separately, then averaged). All previous F1 scores were computed incorrectly by mixing aspects. Notebooks 01-02 now fixed with correct CAFA metric. Need to re-run notebook 03-04 with per-aspect evaluation to get true baseline scores.

### Priority 1: Label Propagation ❌ FAILED
**Expected Impact:** +0.02-0.04 F1  
**Actual Impact:** **-15% F1** (0.2579 → 0.2181)
**Effort:** Low  
**Risk:** High (with noisy predictions)

**Status:** Tested in `notebooks/04_label_propagation.ipynb` — DOES NOT WORK with KNN.

**Why It Failed:**
KNN predictions are ~60-70% accurate at best. Propagating errors to parent terms amplifies mistakes across the GO hierarchy. Propagation requires >90% base accuracy to be beneficial. May work with deep learning models (ESM-2 650M) but not with KNN.

**Implementation:**
```python
def propagate_predictions(predictions, go_graph):
    """Add ancestor terms to predictions"""
    for protein_id in predictions:
        term_confidences = predictions[protein_id]
        
        for term, conf in term_confidences.items():
            # Add all ancestors with max confidence
            ancestors = nx.ancestors(go_graph, term)
            for ancestor in ancestors:
                current_conf = term_confidences.get(ancestor, 0)
                term_confidences[ancestor] = max(current_conf, conf)
    
    return predictions
```

**File to Create:** `src/inference/propagation.py`

---

### Priority 2: Per-Aspect Evaluation + Thresholds (1 hour) 🔥🔥 **CRITICAL**
**Expected Impact:** Unknown - current F1 scores may be inaccurate  
**Effort:** Low  
**Risk:** None

**Status:** Evaluation metric fixed in notebooks 01-02. Need to:
1. Apply same fix to notebooks 03-04
2. Re-run all notebooks to get true per-aspect F1 scores
3. Then tune separate thresholds for MF, BP, CC

**Rationale:**
Competition computes F1 for MF, BP, CC separately, then averages. We were computing single F1 across all terms mixed together. Previous F1=0.2331 may not reflect true competition score.

After getting correct per-aspect scores, tune thresholds separately:
- Molecular Function (MF) might need threshold 0.45
- Biological Process (BP) might need threshold 0.40  
- Cellular Component (CC) might need threshold 0.35

**Implementation:**
```python
# Tune per-aspect thresholds
aspect_thresholds = {}
for aspect in ['MF', 'BP', 'CC']:
    aspect_terms = get_terms_for_aspect(aspect)
    best_thresh, best_f1 = find_optimal_threshold(
        y_val[:, aspect_terms], 
        y_pred[:, aspect_terms]
    )
    aspect_thresholds[aspect] = best_thresh
```

**File to Modify:** `src/training/finetune_esm.py` (evaluation section)

---

### Priority 3: Larger Model - ESM-2 35M/150M (12 hours) 🔥🔥
**Expected Impact:** +0.03-0.05 F1  
**Effort:** Medium (just change model name)  
**Risk:** Medium (GPU memory constraints)

**Action:** Replace `facebook/esm2_t6_8M_UR50D` → `facebook/esm2_t12_35M_UR50D` (35M) or `facebook/esm2_t30_150M_UR50D` (150M). Adjust batch size and gradient accumulation as needed.

**Why This Works:**
Larger models capture more nuanced sequence patterns. Industry standard is 150M-650M parameters for protein tasks.

**GPU Check:**
```python
# Test if model fits in 8GB VRAM
model = ESMForGOPrediction.from_pretrained(
    'facebook/esm2_t12_35M_UR50D',
    num_labels=5000
)
print(f"Model size: {model.num_parameters() / 1e6:.1f}M params")
```

**File to Modify:** `src/training/finetune_esm.py` (line 168, model initialisation)

---

## 3. 🚀 Next Steps: Quick Wins

**Target:** F1 = 0.25-0.27 (achievable in 2-3 days)

### Option A: Ensemble KNN + Fine-Tuned ESM-2 (30 mins) 🎯
**Expected Impact:** +0.01-0.02 F1  
**Effort:** Very Low  
**Risk:** None

**Why This Works:**
KNN captures homology (sequence similarity), ESM-2 captures learned patterns. They make different types of errors, so ensemble reduces variance.

**Implementation:**
```python
# Simple ensemble
pred_ensemble = 0.4 * pred_knn + 0.6 * pred_esm2

# Or weighted by validation F1
w_knn = 0.1776 / (0.1776 + 0.2331)  # 0.43
w_esm = 0.2331 / (0.1776 + 0.2331)  # 0.57
pred_ensemble = w_knn * pred_knn + w_esm * pred_esm2
```

**File to Create:** `src/inference/ensemble.py`

---

### Option B: Train on 10k GO Terms (6 hours) 🔥🔥
**Expected Impact:** +0.02-0.03 F1  
**Effort:** Low (just change `num_labels=10000`)  
**Risk:** Low

**Action:** Modify `finetune_dataset.py` to keep top 10k terms instead of 5k, retrain with same hyperparameters.

**Trade-off:**
More terms = more coverage but slightly lower per-term performance. Net effect usually positive.

---

### Option C: Longer Sequences (1024 tokens) (8 hours) 🔥
**Expected Impact:** +0.01-0.02 F1  
**Effort:** Low  
**Risk:** Medium (VRAM constraints)

**Action:** Change `max_length=512` → `max_length=1024` in tokeniser, reduce batch size to 4, retrain.

**Why This Works:**
Currently truncating sequences longer than 512 amino acids. Longer context helps for large proteins (e.g., titin has 34,350 residues).

---

## 4. 🔮 Future Work: Advanced Features

**Target:** F1 = 0.30-0.35+ (requires 2-4 weeks)

### Multi-Modal Fusion 🌟
**Expected Impact:** +0.05-0.10 F1  
**Effort:** High  
**Timeline:** 2 weeks

**Features to Add:**
1. **MSA/PSSM profiles** (evolutionary signal)
   - Run HHblits or JackHMMER
   - Extract position-specific scoring matrices
   - Encode as additional embeddings

2. **Pfam domain annotations** (functional motifs)
   - Run Pfam HMM scan
   - Encode domain presence as binary features
   - Or use domain embeddings

3. **Structure embeddings** (AlphaFold/ESMFold)
   - Predict structure with ESMFold (faster than AlphaFold)
   - Extract contact maps or 3D coordinates
   - Use GNN to encode structural graph

**Architecture:**
```
Sequence (ESM-2) → [320-dim]
MSA Profile → [256-dim]
Pfam Domains → [128-dim]
Structure → [256-dim]
    ↓
Cross-Attention Fusion → [512-dim]
    ↓
Classification Head → [5000 GO terms]
```

**Files to Create:**
- `src/features/msa_extractor.py`
- `src/features/pfam_scanner.py`
- `src/features/structure_encoder.py`
- `src/models/fusion_model.py`

---

### GO Term Embeddings 🌟
**Expected Impact:** +0.03-0.05 F1  
**Effort:** Medium  
**Timeline:** 1 week

**Approach:**
Instead of treating GO terms as independent units, learn embeddings that capture:
1. **Text similarity** (from GO definitions)
2. **Graph position** (from ontology structure)
3. **Co-occurrence** (terms that appear together)

**Architecture:**
```
Protein → ESM-2 → Protein Embedding [320]
GO Term → Text Encoder + Graph Encoder → GO Embedding [320]
    ↓
Compatibility Score = dot_product(Protein, GO) / temperature
```

**Why This Works:**
Similar GO terms (e.g., "kinase activity" vs "protein kinase activity") share embedding space, so model generalises better.

**Files to Create:**
- `src/models/go_embeddings.py`
- `src/models/compatibility_model.py`

---

### Hierarchy-Aware Loss 🌟
**Expected Impact:** +0.01-0.02 F1  
**Effort:** Medium  
**Timeline:** 3 days

**Add penalty term to loss function:**
```python
# Standard loss
loss_bce = AsymmetricLoss(pred, target)

# Hierarchy penalty
# If child is predicted but parent is not, add cost
for term in predicted_terms:
    parents = get_parents(term, go_graph)
    for parent in parents:
        if pred[parent] < pred[term]:  # Inconsistency
            loss_hierarchy += (pred[term] - pred[parent]) ** 2

# Combined loss
loss = loss_bce + 0.1 * loss_hierarchy
```

**File to Modify:** `src/training/loss.py`

---

### Hard Negative Mining 🌟
**Expected Impact:** +0.01-0.02 F1  
**Effort:** Medium  
**Timeline:** 3 days

**Approach:**
During training, identify "hard negatives" (GO terms model incorrectly predicts with high confidence) and sample them more frequently.

**Implementation:**
```python
# After each epoch, collect hard negatives
hard_negatives = []
for protein, pred, true in validation_set:
    false_positives = (pred > 0.5) & (true == 0)
    hard_negatives.extend(false_positives)

# In next epoch, oversample hard negatives
sampler = WeightedRandomSampler(
    weights=hard_negative_weights,
    num_samples=len(train_set)
)
```

**File to Create:** `src/training/hard_negative_miner.py`

---

## 5. 📤 Submission Pipeline

**Status:** ✅ Complete — `notebooks/05_submission_generation.ipynb`  
**Format:** CAFA-6 Official TSV

### Official Format Requirements (from competition docs)

| Requirement | Value |
|-------------|-------|
| **File format** | TSV (tab-separated) |
| **Header** | None (no header row) |
| **Columns** | `EntryID<tab>GO_term<tab>confidence` |
| **Confidence range** | (0, 1.000] — zeros NOT allowed |
| **Significant figures** | Up to 3 (e.g., 0.931, 0.54, 0.989) |
| **Max terms/protein** | 1500 (MF+BP+CC combined) |
| **GO term format** | GO:XXXXXXX (7 digits) |
| **Propagation** | Auto-handled by organisers if missing |

### Example Submission (from official docs)
```
P9WHI7   GO:0009274   0.931   
P9WHI7   GO:0071944   0.540
P9WHI7   GO:0005575   0.324
P04637   GO:1990837   0.23
P04637   GO:0031625   0.989
P04637   GO:0043565   0.64
```

### Current Best Configuration
```python
# Aspect-specific thresholds (from notebook 04)
THRESHOLDS = {
    'MF': 0.40,
    'BP': 0.20,
    'CC': 0.40
}
MAX_TERMS_PER_PROTEIN = 1500

# Validation F1: 0.2579
```

### Key Notes
1. **Label propagation NOT required** — organisers auto-propagate: "predictions will be recursively propagated by assigning each parent term a score that is the maximum score among its children's scores"
2. **Score of 0 not allowed** — omit pairs with zero confidence
3. **Proteins not listed** — assumed to have all predictions = 0

### Step 1: Generate Test Predictions

**Code:**
```python
from src.models.esm_classifier import ESMForGOPrediction
from src.data.finetune_dataset import create_datasets
import torch

# Load model
model = ESMForGOPrediction.from_pretrained('models/esm_finetuned/best_model/')
model.eval()
model.to('cuda')

# Load test data
test_dataset = load_test_sequences('Test/testsuperset.fasta')
test_loader = DataLoader(test_dataset, batch_size=8)

# Generate predictions
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        logits = model(**batch)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())

predictions = np.vstack(all_preds)  # Shape: (n_test, 5000)
```

### Step 2: Apply Aspect-Specific Thresholds

**Code:**
```python
def apply_aspect_thresholds(pred_df, term_to_aspect, thresholds):
    """Filter predictions using aspect-specific thresholds."""
    aspect_map = {'F': 'MF', 'P': 'BP', 'C': 'CC'}
    
    pred_df['aspect'] = pred_df['term'].map(term_to_aspect).map(aspect_map)
    pred_df['threshold'] = pred_df['aspect'].map(thresholds)
    
    # Filter by aspect-specific threshold
    filtered = pred_df[pred_df['probability'] >= pred_df['threshold']]
    return filtered[['EntryID', 'term', 'probability']]
```

### Step 3: Limit Terms Per Protein

**Code:**
```python
def limit_terms_per_protein(pred_df, max_terms=1500):
    """Keep top N terms per protein by confidence."""
    sorted_df = pred_df.sort_values(
        ['EntryID', 'probability'], 
        ascending=[True, False]
    )
    return sorted_df.groupby('EntryID').head(max_terms)
```

### Step 4: Format & Save Submission

**Code:**
```python
def create_submission(pred_df, output_path):
    """Create CAFA-6 official format submission."""
    submission = pred_df.rename(columns={
        'term': 'GO_term', 
        'probability': 'confidence'
    })
    
    # Round to 3 decimal places, remove zeros
    submission['confidence'] = submission['confidence'].round(3)
    submission = submission[submission['confidence'] > 0]
    
    # Save as TSV (no header)
    submission[['EntryID', 'GO_term', 'confidence']].to_csv(
        output_path, 
        sep='\t', 
        header=False, 
        index=False
    )
```

### Step 5: Validate Submission

**Code:**
```python
def validate_submission(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None, 
                     names=['EntryID', 'GO_term', 'confidence'])
    
    # Check format
    assert len(df.columns) == 3, "Must have 3 columns"
    
    # Check confidence range (0, 1] — zeros NOT allowed
    df['confidence'] = df['confidence'].astype(float)
    assert df['confidence'].min() > 0, "Confidence must be > 0"
    assert df['confidence'].max() <= 1, "Confidence must be <= 1"
    
    # Check max per protein
    terms_per_protein = df.groupby('EntryID').size()
    assert terms_per_protein.max() <= 1500, \
        f"Max 1500 terms per protein (found {terms_per_protein.max()})"
    
    # Check GO term format
    assert df['GO_term'].str.match(r'^GO:\d{7}$').all(), \
        "Invalid GO term format"
    
    print("✅ Submission is valid!")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique proteins: {df['EntryID'].nunique():,}")
    print(f"  Unique GO terms: {df['GO_term'].nunique():,}")
    print(f"  Avg terms/protein: {terms_per_protein.mean():.1f}")

validate_submission('submissions/submission_v1.tsv')
```

---

## 6. 📊 Performance Tracking

### Current Model Zoo

| Model | F1 Score | Precision | Recall | Threshold | Status |
|-------|----------|-----------|--------|-----------|--------|
| Frequency Baseline | 0.1619 | - | - | N/A | ✅ |
| KNN (single threshold) | 0.2520 | - | - | 0.40 | ✅ |
| **KNN (aspect-specific)** | **0.2579** | - | - | MF=0.40, BP=0.20, CC=0.40 | 🏆 **BEST** |
| KNN + Propagation | 0.2181 | - | - | MF=0.40, BP=0.20, CC=0.40 | ❌ Failed |
| MLP (Frozen) | 0.1672 | - | - | 0.10 | ✅ |
| ESM-2 (BCE) | 0.1806 | 0.1952 | 0.2449 | 0.10 | ✅ |
| ESM-2 (Asym Loss) | 0.2331 | 0.3397 | 0.2379 | 0.40 | Needs re-eval |

### Roadmap to Competitive Performance

| Target F1 | Improvements Needed | Timeline |
|-----------|---------------------|----------|
| 0.26 | Per-aspect thresholds ✅ DONE | — |
| 0.27 | + Larger model (35M) or ensemble | 12 hours |
| 0.30 | + 10k terms + longer sequences | 2 days |
| 0.35 | + Multi-modal fusion (MSA + structure) | 2 weeks |
| 0.40+ | + GO embeddings + hierarchy-aware loss | 4 weeks |

---

## 7. 📚 Technical Resources

### Existing Codebase

**Data Infrastructure:**
- `src/data/loaders.py` - SequenceLoader, LabelLoader, OntologyLoader
- `src/data/finetune_dataset.py` - ESMFineTuneDataset, create_datasets()

**Models:**
- `src/models/esm_classifier.py` - ESMForGOPrediction model
- `src/models/baseline_frequency.py` - Frequency baseline
- `src/models/baseline_embedding_knn.py` - KNN baseline
- `src/models/architecture.py` - MLP classifier

**Training:**
- `src/training/finetune_esm.py` - Training orchestration
- `src/training/loss.py` - AsymmetricLoss implementation
- `src/training/trainer.py` - Generic trainer class

**Documentation:**
- `SUMMARY.md` - Complete project summary
- `PROGRESS_TRACKER.md` - Detailed progress log
- `docs/milestones/1.md` - Project explanation with analogies
- `docs/dataset_description.md` - Competition details
- `openai5.1_cafa_6_roadmap_and_architecture_summary.md` - Advanced roadmap

### Key Libraries

**Core ML:**
- PyTorch 2.5.1 + CUDA 12.1
- Transformers 4.x (HuggingFace)
- scikit-learn 1.5+

**Bioinformatics:**
- BioPython 1.83
- obonet (GO ontology parsing)

**Pre-trained Models:**
- ESM-2: `facebook/esm2_t6_8M_UR50D` (current)
- ESM-2 35M: `facebook/esm2_t12_35M_UR50D` (next)
- ESM-2 150M: `facebook/esm2_t30_150M_UR50D` (future)
- ProtT5: `Rostlab/prot_t5_xl_uniref50`

### Hardware Requirements

**Current Setup:**
- GPU: NVIDIA GeForce RTX 2070 (8GB VRAM)
- CUDA: 12.1
- Batch size: 8 (effective 32 with grad accumulation)

**For 150M Model:**
- Need: 8GB VRAM minimum
- Batch size: 4 (effective 32 with grad accumulation 8x)
- Or: Use gradient checkpointing + mixed precision (fp16)

---

## 🎯 Immediate Next Action

**Recommended:** Start with Priority 1 (Label Propagation)

**Steps:**
1. Create `src/inference/propagation.py`
2. Load GO graph with obonet
3. Implement ancestor propagation function
4. Test on validation set
5. Measure F1 improvement

**Expected outcome:** F1 0.2331 → 0.25-0.26 in 2 hours of work.

---

**End of Plan**
