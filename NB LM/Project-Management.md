# Comprehensive Project Plan: Professional Protein Function Predictor

## Project Overview

**Project Goal**: Accurately predict protein function (Molecular Function, Biological Process, and Cellular Component) based on amino acid sequence using a multi-modal ensemble model stack, maximizing performance as measured by the IA-weighted F1 score across no-knowledge, limited-knowledge, and partial-knowledge targets.

**Core Methodology**: Modular Stacking Approach (GBDTs & LogReg → GCNs) combined with Strict GO Hierarchy Post-Processing.

**Role**: Project Manager and Professional Protein Function Predictor

---

## Phase 1: Feature Engineering and Data Scale

This phase establishes the foundational environment and generates the robust, multi-modal, and large-scale feature inputs necessary for Level 1 model training.

### 1.1 Environment Setup

| Aspect | Details |
|--------|---------|
| **Milestone** | Creation of two specialized Python environments |
| **Deliverables** | `pytorch-env` (Deep Learning), `rapids-env` (Preprocessing/GBDT/LogReg), `config.yaml` |
| **Timeline** | Initial Setup |
| **Requirements** | Shell scripts, dependency management |

### 1.2 Core Data Structuring

| Aspect | Details |
|--------|---------|
| **Milestone** | Parse and structure all input data |
| **Deliverables** | `train_seq.feather`, `test_seq.feather`, `real_targets.parquet`, `priors.pkl`, `nulls.pkl` |
| **Timeline** | ~1 hour for parsing and initial structuring |
| **Requirements** | RAPIDS_ENV, `parse_fasta.py`, `create_helpers.py` |

### 1.3 External Data Acquisition

| Aspect | Details |
|--------|---------|
| **Milestone** | Download and parse external GO annotation datasets |
| **Deliverables** | `prop_train_no_kaggle.tsv` (No-Kaggle electronic labels as powerful stacker features) |
| **Timeline** | Long download time (EBI datasets are large) |
| **Requirements** | Network access, `dw_goant.py`, `parse_go_single.py` |

**Key Insight**: No-Kaggle electronic labels serve as foundational features for the stacker models.

### 1.4 Hierarchy-Aware Preprocessing

| Aspect | Details |
|--------|---------|
| **Milestone** | Propagate external labels through GO hierarchy |
| **Deliverables** | Hierarchy-consistent annotations ready for feature extraction |
| **Timeline** | Depends on dataset size |
| **Requirements** | RAPIDS_ENV, `prop_tsv.py`, `go-basic.obo` |

**Result**: Every specific GO term linked to all general parent terms before training.

### 1.5 Multi-Modal Feature Generation

| Aspect | Details |
|--------|---------|
| **Milestone** | Generate high-dimensional embeddings for all proteins |
| **Deliverables** | 7 feature modalities: T5 (1024D), ESM2 (2560D), Ankh (1536D), Taxonomy (70D), Text Abstract (10279D) |
| **Timeline** | Multi-day GPU inference |
| **Requirements** | PYTORCH_ENV, significant GPU VRAM, embedding model scripts (`t5.py`, `esm2sm.py`) |

---

## Phase 2: Diverse Base Model Training (Level 1)

This phase trains multiple foundational models using different algorithms and feature subsets, generating Out-Of-Fold (OOF) predictions that will serve as features for the Level 2 GCN stacker.

### 2.1 GBDT (Py-Boost) Training

| Aspect | Details |
|--------|---------|
| **Milestone** | Train four distinct GBDT model sets |
| **Deliverables** | `oof_pred.pkl`, `test_pred.pkl` for each model variant |
| **Model Variants** | 4 models: raw & conditional targets, T5/Taxon, T5/ESM/Taxon features |
| **Target Coverage** | 4,500 GO terms |
| **Timeline** | ~15 hours per 5-fold CV loop per model |
| **Requirements** | RAPIDS_ENV, 32 GB GPU RAM (V100/A100), `train_pb.py` |

**Critical**: IA weights used as training weights to prioritize rare terms.

### 2.2 Logistic Regression (LogReg) Training

| Aspect | Details |
|--------|---------|
| **Milestone** | Train LogReg models on expanded feature space |
| **Deliverables** | `oof_pred.pkl`, `test_pred.pkl` for LogReg models |
| **Feature Set** | T5/Taxon features |
| **Target Coverage** | Up to 13,500 GO terms for diversity |
| **Timeline** | ~10 hours (large target set), ~2 hours (smaller set) |
| **Requirements** | RAPIDS_ENV, 32 GB GPU RAM, `train_lin.py` |

### 2.3 DNN Ensemble Training

| Aspect | Details |
|--------|---------|
| **Milestone** | Train colossal multi-input deep learning model |
| **Deliverables** | 25 trained DNN models (`oof_pred.pkl`, `test_pred.pkl`) |
| **Input Architecture** | 7 simultaneous feature modalities (all embeddings + taxa + external) |
| **Ensembling Strategy** | 5-fold CV × 5 random states = 25 models |
| **Class Weights** | IA weights to prioritize rare terms |
| **Timeline** | Multi-day training cycles (highly computationally intensive) |
| **Requirements** | PYTORCH_ENV, 32 GB GPU RAM, `train_models.py`, `create_gkf.py` |

**Key Insight**: Extreme ensembling dampens noise and stabilizes predictions across random initializations and fold splits.

---

## Phase 3: Hierarchy-Aware Stacking (Level 2 - The Winning Edge)

This phase uses Graph Convolutional Networks (GCNs) to process the Level 1 predictions, imposing graph structure and biological logic onto the statistical features.

### 3.1 Stacker Input Preparation

| Aspect | Details |
|--------|---------|
| **Milestone** | Consolidate Level 1 predictions |
| **Deliverables** | Aggregated OOF predictions formatted as GCN input features |
| **Input Source** | Level 1 OOF predictions from GBDTs, LogRegs, DNNs |
| **Key Change** | Statistical predictions replace raw embeddings as primary features |
| **Timeline** | Data aggregation and formatting |
| **Requirements** | RAPIDS_ENV |

**Critical Design Decision**: Using Level 1 predictions instead of raw embeddings allows the GCN to learn from diverse statistical perspectives.

### 3.2 GCN Training and Specialization

| Aspect | Details |
|--------|---------|
| **Milestone** | Train three ontology-specific GCN models |
| **Deliverables** | `gcn_bp_model`, `gcn_mf_model`, `gcn_cc_model` |
| **Ontologies** | Biological Process (BP), Molecular Function (MF), Cellular Component (CC) |
| **Graph Learning** | GCNs inherently learn GO hierarchy structure |
| **Timeline** | BP: ~13 hours, MF: ~4 hours, CC: ~2 hours (can run in parallel) |
| **Requirements** | PYTORCH_ENV, 32 GB GPU RAM, `train_gcn.py` |

**Total Sequential Time**: ~19 hours (if run serially); ~13 hours (if run in parallel).

### 3.3 GCN Inference and Test-Time Augmentation

| Aspect | Details |
|--------|---------|
| **Milestone** | Generate multiple TTA prediction sets |
| **Deliverables** | `pred_tta_0.tsv`, `pred_tta_1.tsv`, `pred_tta_2.tsv`, `pred_tta_3.tsv` |
| **TTA Strategy** | Multiple augmented inference runs per GCN model |
| **Benefit** | Dampen noise and boost generalization |
| **Timeline** | Varies by augmentation count |
| **Requirements** | PYTORCH_ENV, `predict_gcn.py` |

### 3.4 TTA Aggregation

| Aspect | Details |
|--------|---------|
| **Milestone** | Average TTA predictions |
| **Deliverables** | `pred.tsv` (final statistical probabilities before constraints) |
| **Aggregation Method** | Simple average across all TTA runs |
| **Timeline** | Quick aggregation step |
| **Requirements** | RAPIDS_ENV, `collect_ttas.py` |

---

## Phase 4: Strict Post-Processing and Submission

This crucial final phase enforces the fundamental biological rules of the Gene Ontology hierarchy onto the statistical outputs, guaranteeing adherence before final submission.

### 4.1 Max Propagation (Parent Rule)

| Aspect | Details |
|--------|---------|
| **Milestone** | Apply upward constraint enforcement |
| **Rule Enforced** | If child term predicted, all parent terms must have score ≥ child score |
| **Deliverables** | `pred_max.tsv` (upward-consistent predictions) |
| **Timeline** | Varies by dataset size |
| **Requirements** | RAPIDS_ENV, `step.py` (direction: max, lr: 0.7) |

**Biological Justification**: Prevents prediction of specific function without general category.

### 4.2 Min Propagation (Child Rule)

| Aspect | Details |
|--------|---------|
| **Milestone** | Apply downward constraint enforcement |
| **Rule Enforced** | If parent term has low score, child terms cannot exceed parent score |
| **Deliverables** | `pred_min.tsv` (downward-consistent predictions) |
| **Timeline** | Varies by dataset size |
| **Requirements** | RAPIDS_ENV, `step.py` (direction: min, lr: 0.7) |

**Biological Justification**: Prevents overly specific predictions when general category is uncertain.

### 4.3 Final Submission Generation

| Aspect | Details |
|--------|---------|
| **Milestone** | Create final submission file |
| **Deliverables** | `submission.tsv` (averaged max/min propagated scores) |
| **Aggregation** | Average `pred_max.tsv` and `pred_min.tsv` |
| **External Data** | Incorporate leakage data (cafa-terms-diff, quickgo51) |
| **Validation** | Score range (0, 1.000], max 1,500 terms per protein |
| **Timeline** | Final aggregation and output |
| **Requirements** | RAPIDS_ENV, `make_submission.py` (max-rate: 0.5) |

### 4.4 Final Constraint Verification

| Aspect | Details |
|--------|---------|
| **Validation Checks** | ✅ All scores in (0, 1.000] (no zero scores)<br>✅ Max 1,500 combined GO terms per protein<br>✅ Correct file format |
| **Automated Handling** | `make_submission.py` handles filtering and sorting |

---

## Key Constraints and Insights

### 1. Resource Intensity

**GPU Requirements**:
- Minimum: 32 GB GPU RAM (V100/A100 cards)
- Multi-day, multi-GPU training cycles required
- Phases 2 & 3 are computationally bottlenecks

**Time Estimates**:
- Phase 1: 1–7 days (data acquisition is unpredictable)
- Phase 2: 10–15 days (multiple large models)
- Phase 3: 1–2 days (GCNs, can run in parallel)
- Phase 4: 1–2 days (post-processing)
- **Total Project Duration**: 3–4 weeks

### 2. IA Metric Focus

**Strategy**: All training utilizes Information Accretion (IA) weights as `class_weight` to focus model performance on:
- Predicting rare, highly specific GO terms
- Maximizing reward in the IA-weighted F1 evaluation metric
- Ensuring scarce predictions have high confidence

**Implementation**:
- Load `IA.tsv` during data preprocessing
- Pass IA weights to all base models (GBDT, LogReg, DNN)
- IA weighting is **crucial** to winning strategy

### 3. Hierarchy Enforcement Philosophy

**Key Design Decision**: Post-hoc constraint enforcement (Min/Max Propagation in Phase 4) proved more effective than attempting to "bake in" constraints during training using custom loss functions.

**Why?**
- Statistical models should maximize prediction accuracy first
- Biological rules applied afterward as a "final refinement"
- Separation of concerns: let ML learn patterns, then enforce biology
- Avoids conflict between statistical optimization and hard constraints

**Analogy**: Like a skilled carpenter who first cuts and shapes all pieces perfectly (statistical prediction), then uses precise final joints and hardware (Min/Max propagation) to ensure the structure is sound and adheres strictly to the blueprint (Gene Ontology hierarchy).

---

## Success Metrics

| Metric | Target |
|--------|--------|
| **IA-Weighted F1** | Maximize across all ontologies (BP, MF, CC) |
| **Coverage** | All proteins predicted with valid GO terms |
| **Constraint Adherence** | 100% biological consistency post-Phase 4 |
| **Submission Validity** | Passes all format and range checks |

---

## Milestones and Timeline

| Phase | Duration | Critical Path | Dependencies |
|-------|----------|----------------|--------------|
| **Phase 1** | 1–7 days | Data acquisition | Network, storage |
| **Phase 2** | 10–15 days | GBDT training | Phase 1 completion, 32 GB GPU |
| **Phase 3** | 1–2 days | GCN training (parallel) | Phase 2 completion |
| **Phase 4** | 1–2 days | Post-processing | Phase 3 completion |
| **TOTAL** | **3–4 weeks** | All phases sequential | Continuous GPU access |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| GPU out-of-memory during Phase 2 | Batch model training, use smaller fold/state subsets if needed |
| Long data acquisition in Phase 1 | Parallelize downloads, cache intermediate results |
| GCN training exceeds estimated time | Run BP/MF/CC in parallel on separate GPUs |
| Post-processing errors | Validate constraint enforcement with sample predictions before full run |

---

## Next Steps

1. **Immediate**: Set up `pytorch-env` and `rapids-env` with all dependencies
2. **Week 1**: Complete Phase 1 (data structuring, external acquisition, embeddings)
3. **Weeks 2–3**: Execute Phase 2 (base models) and Phase 3 (GCN stacking) in parallel
4. **Week 4**: Phase 4 (post-processing) and final submission
