# Rank 1 Winning Strategy: Modular GCN Stacking with Extreme Multimodal Input

## Overview

This comprehensive guide details the end-to-end execution of the **Rank 1 Winning Strategy** for protein function prediction, emphasizing the **Modular GCN Stacking with Extreme Multimodal Input** methodology.

### Resource Requirements

- **GPU RAM**: 32 GB minimum (V100/A100 cards recommended)
- **Storage**: Large dataset capacity via Google Drive or equivalent
- **Libraries**: protlib, protnn, nn_solution helper scripts

### Prerequisites

- Google Drive mounted (or equivalent cloud storage)
- Helper library scripts accessible within your `BASE_PATH`

---

## Phase 0: Initial Setup and Environment Configuration

The winning strategy requires **two specialized environments**:
- `rapids-env`: for preprocessing and GBDTs (Py-Boost)
- `pytorch-env`: for deep learning (embeddings and GCNs)

### 0.1 Mount Drive & Set Paths

**Purpose**: Essential for managing large input data and saving model checkpoints.

```python
from google.colab import drive
drive.mount('/content/drive')
BASE_PATH = '/content/drive/MyDrive/CAFA5'
```

### 0.2 Define Configuration

**Purpose**: Load the config.yaml file, which sets paths for the two specialized environments and data files.

```python
with open('config.yaml') as f:
    CONFIG = yaml.safe_load(f)

RAPIDS_ENV = os.path.join(BASE_PATH, CONFIG['rapids-env'])
PYTORCH_ENV = os.path.join(BASE_PATH, CONFIG['pytorch-env'])
```

### 0.3 Create Environments

**Purpose**: Install necessary libraries (PyTorch, RAPIDS, Py-Boost).

```bash
./create-rapids-env.sh {BASE_PATH}
./create-pytorch-env.sh {BASE_PATH}
```

⚠️ **Note**: High GPU requirement (32 GB RAM) for subsequent steps.

---

## Phase 1: Feature Engineering and Data Scale

This phase generates the multi-modal, large-scale, and hierarchy-aware inputs needed for Level 1 model training, focusing on maximizing the **Information Accretion (IA) weighted F1 score**.

### 1.1 Core Data Structuring

**Purpose**: Parse FASTA sequences and prepare ground truth targets.

```bash
{RAPIDS_ENV} protlib/scripts/parse_fasta.py --config-path {CONFIG_PATH}
{RAPIDS_ENV} protlib/scripts/create_helpers.py --config-path {CONFIG_PATH} --batch-size 10000
```

**Outputs**:
- Sequences in `.feather` format
- Targets in Parquet
- `priors.pkl`: prior means for IA weighting
- `nulls.pkl`: NaN rates for IA weighting

### 1.2 Acquire External Data

**Purpose**: Download external GO annotation datasets.

```bash
{RAPIDS_ENV} protlib/scripts/downloads/dw_goant.py --config-path {CONFIG_PATH}
```

### 1.3 Separate by Evidence Code

**Purpose**: Parse external files into experimental (Kaggle) and electronic labels (No-Kaggle).

```bash
{RAPIDS_ENV} protlib/scripts/parse_go_single.py --file goa_uniprot_all.gaf.216.gz ...
{RAPIDS_ENV} protlib/scripts/parse_go_single.py --file goa_uniprot_all.gaf.214.gz ...
```

**Key Insight**: No-Kaggle electronic labels (`prop_train_no_kaggle.tsv`) serve as powerful features for stacker models.

### 1.4 Hierarchy Propagation

**Purpose**: Apply GO hierarchy rules to external labels using go-basic.obo.

```bash
{RAPIDS_ENV} {BASE_PATH}/protlib/scripts/prop_tsv.py \
  --path {file} \
  --graph {BASE_PATH}/Train/go-basic.obo \
  --output {name} \
  --device 0
```

**Result**: Every specific term is linked to general parent terms before training.

### 1.5 Generate Embeddings

**Purpose**: Create high-dimensional feature vectors required for multi-modal input.

```bash
mkdir embeds

# T5 Embeddings (1024D)
{PYTORCH_ENV} {BASE_PATH}/nn_solution/t5.py --config-path {CONFIG_PATH} --device 0

# ESM2 Large Embeddings (2560D)
{PYTORCH_ENV} {BASE_PATH}/nn_solution/esm2sm.py --config-path {CONFIG_PATH} --device 0

# Also generate:
# - Ankh Embeddings (1536D)
# - Taxonomic ID Data (70D)
# - Text Abstract Embeddings (10279D)
```

**Input**: 7 separate feature modalities for the multi-modal architecture.

---

## Phase 2: Diverse Base Model Training (Level 1)

Train foundational models (GBDTs, LogRegs, DNNs) to generate **Out-Of-Fold (OOF) predictions**, which serve as structured statistical features for the Level 2 GCN stacker.

### 2.1 GBDT (Py-Boost) Training

**Purpose**: Train four distinct GBDT models on different feature combinations.

```bash
for model_name in ['pb_t54500_raw', 'pb_t54500_cond', 'pb_t5esm4500_raw', 'pb_t5esm4500_cond']:
    {RAPIDS_ENV} {BASE_PATH}/protlib/scripts/train_pb.py \
      --config-path {CONFIG_PATH} \
      --model-name {model_name} \
      --device 0
```

**Models**:
- Target: 4,500 GO terms
- Feature sets: T5/Taxon, T5/ESM/Taxon
- Raw and conditional targets

⚠️ **Critical**: Use IA weights during training to prioritize rare terms.

### 2.2 Logistic Regression Training

**Purpose**: Train linear models on T5/Taxon features for ensemble diversity.

```bash
for model_name in ['lin_t5_raw', 'lin_t54500_cond']:
    {RAPIDS_ENV} {BASE_PATH}/protlib/scripts/train_lin.py \
      --config-path {CONFIG_PATH} \
      --model-name {model_name} \
      --device 0
```

**Coverage**: Up to 13,500 GO terms.

### 2.3 DNN Ensemble Training (4th Place Strategy Insight)

**Purpose**: Implement brute-force blending with seven separate feature inputs.

```bash
# Prepare helper files
{RAPIDS_ENV} {BASE_PATH}/protlib/scripts/create_gkf.py --config-path {CONFIG_PATH}

# Train 25-model ensemble (5-fold CV × 5 random states)
{PYTORCH_ENV} {BASE_PATH}/nn_solution/train_models.py --config-path {CONFIG_PATH} --device 0
```

**Architecture**:
- Input: 7 simultaneous feature modalities (all embeddings + Taxa)
- Class weights: IA weights
- Output: 25 trained models (5 CV folds × 5 random states)
- Benefit: Stabilizes predictions and dampens noise

---

## Phase 3: Hierarchy-Aware Stacking (Level 2 - GCNs)

**Core Winning Strategy**: Graph Convolutional Networks (GCNs) impose graph structure and biological logic onto statistical predictions.

### 3.1 Stacker Input Preparation

**Purpose**: Consolidate Level 1 OOF predictions to replace raw sequence embeddings.

```bash
# Aggregate all Phase 2 oof_pred.pkl files
# Format for GCN input
```

**Key Point**: Level 1 predictions become Level 2 features (not raw embeddings).

### 3.2 GCN Training and Specialization

**Purpose**: Train three independent GCN models (one per GO ontology).

```bash
for ont in ['bp', 'mf', 'cc']:
    {PYTORCH_ENV} {BASE_PATH}/protnn/scripts/train_gcn.py \
      --config-path {CONFIG_PATH} \
      --ontology {ont} \
      --device 0
```

**Training Time**:
- BP (Biological Process): ~13 hours
- MF (Molecular Function): ~4 hours
- CC (Cellular Component): ~2 hours

⚠️ **GPU Requirement**: 32 GB RAM essential.

### 3.3 GCN Inference and Test-Time Augmentation

**Purpose**: Generate multiple prediction outputs via TTA for improved generalization.

```bash
{PYTORCH_ENV} {BASE_PATH}/protnn/scripts/predict_gcn.py --config-path {CONFIG_PATH} --device 0
```

**Outputs**: pred_tta_0.tsv, pred_tta_1.tsv, pred_tta_2.tsv, pred_tta_3.tsv

### 3.4 TTA Aggregation

**Purpose**: Average TTA predictions to produce final statistical probability scores.

```bash
{RAPIDS_ENV} {BASE_PATH}/protlib/scripts/postproc/collect_ttas.py --config-path {CONFIG_PATH} --device 0
```

**Output**: pred.tsv (before biological constraints)

---

## Phase 4: Strict Post-Processing and Submission

**Crucial Phase**: Enforce fundamental biological rules of the GO hierarchy onto statistical outputs.

**Key Insight**: Post-processing constraints proved more effective than "baking in" constraints during training.

### 4.1 Max Propagation (Parent Rule)

**Purpose**: Enforce upward consistency in GO hierarchy.

**Rule**: If a child term is predicted, all parent terms (ancestors) must have scores ≥ child score.

```bash
{RAPIDS_ENV} {BASE_PATH}/protlib/scripts/postproc/step.py \
  --config-path {CONFIG_PATH} \
  --device 0 \
  --lr 0.7 \
  --direction max
```

**Output**: pred_max.tsv

### 4.2 Min Propagation (Child Rule)

**Purpose**: Enforce downward consistency in GO hierarchy.

**Rule**: If a parent term has a low score, child terms cannot have higher confidence than that parent.

```bash
{RAPIDS_ENV} {BASE_PATH}/protlib/scripts/postproc/step.py \
  --config-path {CONFIG_PATH} \
  --device 0 \
  --lr 0.7 \
  --direction min
```

**Output**: pred_min.tsv

### 4.3 Final Submission Generation

**Purpose**: Combine max and min propagated results.

```bash
{RAPIDS_ENV} {BASE_PATH}/protlib/scripts/postproc/make_submission.py \
  --config-path {CONFIG_PATH} \
  --device 0 \
  --max-rate 0.5
```

**Process**:
1. Average pred_max.tsv and pred_min.tsv
2. Incorporate external leakage data (cafa-terms-diff, quickgo51)

### 4.4 Final Constraint Check

**Purpose**: Verify submission adheres to competition rules.

**Validation Checklist**:
- ✅ Scores in range (0, 1.000] (no zero scores)
- ✅ Maximum 1,500 combined GO terms per protein ID
- ✅ Proper file format

The `make_submission.py` script handles filtering and sorting automatically.

---

## Phase 5 (Optional): Free Text Prediction

While not contributing to the core F1 score, this phase addresses the optional free text prediction challenge.

### 5.1 Generate Textual Descriptions

**Purpose**: Create descriptive English text about protein function using LLM or literature summarization.

**Output Format**:
```
P9WHI7 Text 0.123 P9WHI7 is involved in homologous recombinational repair...
```

**Constraints**:
- Up to 5 lines per protein
- Include confidence score
- Maximum 3,000 ASCII characters total

### 5.2 Integrate Text into Submission

**Purpose**: Combine text predictions with the final submission file.

**Format**: Same submission.tsv file incorporating text predictions.

### 5.3 Verify Text Constraints

Ensure all textual descriptions adhere to character limits before final export.

---

## Summary: Why This Strategy Won

| Component | Benefit |
|-----------|---------|
| **Multimodal Input** | 7 feature sources reduce overfitting and capture diverse biological signals |
| **Diverse Level 1** | GBDTs, LinRegs, DNNs provide complementary statistical perspectives |
| **GCN Stacking** | Graph structure enforces biological hierarchy constraints during learning |
| **Post-Processing** | Strict propagation rules guarantee biologically valid predictions |
| **TTA + Averaging** | Multiple augmentations stabilise final scores |
| **IA Weighting** | Prioritises rare, high-impact GO terms throughout pipeline |

---

## Quick Reference: File Paths

| Data | Location |
|------|----------|
| FASTA sequences | `Train/train_sequences.fasta` |
| Training terms | `Train/train_terms.tsv` |
| Taxonomy data | `Train/train_taxonomy.tsv` |
| GO ontology graph | `Train/go-basic.obo` |
| Test sequences | `Test/testsuperset.fasta` |
| Test taxonomy | `Test/testsuperset-taxon-list.tsv` |
| IA weights | `IA.tsv` |
| Sample submission | `sample_submission.tsv` |
