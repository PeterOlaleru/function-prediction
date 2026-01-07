# KNN Standalone Notebook (Baseline-Aligned)

## Overview
`knn_standalone.ipynb` implements KNN using **baseline methodology** that achieved F1 ~0.216. This notebook fixes critical bugs in the e2e pipeline that caused F1 to drop to ~0.072.

## Baseline Alignment Fixes

Based on detailed auditor feedback, this notebook addresses all 5 performance regressions:

| Issue | E2E Pipeline (Wrong) | Baseline (Correct) | Status |
|-------|---------------------|-------------------|--------|
| **Term Universe** | Filtered to 13,500 terms | Full vocabulary (~25,000+ terms) | ✅ **FIXED** |
| **Score Aggregation** | IA-weighted (double-weighting bug) | Pure similarity | ✅ **FIXED** |
| **Normalization** | No per-protein normalization | Per-protein max normalization | ✅ **FIXED** |
| **Thresholds** | Low thresholds (0.01-0.30) | Higher thresholds (0.1-0.8) | ✅ **FIXED** |
| **Hierarchy** | Explicit GO propagation (redundant) | No propagation (pre-propagated labels) | ✅ **FIXED** |

**Expected Result:** F1 should improve from ~0.072 to ~0.20-0.25 (baseline methodology with better 3B embeddings).

## What's Included
This notebook contains 9 cells for complete KNN training and Kaggle submission:

1. **Cell 0 (Markdown)**: Overview and alignment notes
2. **Cell 1 (Code)**: Basic environment setup
3. **Cell 2 (Code)**: Simplified configuration (NO HuggingFace)
4. **Cell 3 (Code)**: Data loading with FULL term vocabulary
   - **BASELINE FIX:** Uses ALL terms (~25,000+), not filtered to 13,500
   - Loads training/test features
   - Creates target matrix Y with full vocabulary
5. **Cell 4 (Code)**: KNN helper functions
   - **BASELINE FIX:** Uniform weights (NO IA to avoid double-weighting)
   - L2 normalization, Y_knn setup, X_knn_test setup
6. **Cell 5 (Code)**: KNN training (baseline-aligned)
   - **BASELINE FIX:** Pure similarity aggregation (NO IA weighting)
   - **BASELINE FIX:** Per-protein max normalization
   - cuML/sklearn KNN with cosine similarity
   - 5-fold cross-validation
   - Dynamic F1 evaluation with thresholds [0.1-0.8]
   - FORCE_RETRAIN defaults to True
7. **Cell 6 (Code)**: Generate submission.tsv (baseline-aligned)
   - **BASELINE FIX:** NO GO hierarchy propagation (labels already pre-propagated)
   - Applies per-aspect thresholds or default 0.3
   - Formats submission per CAFA rules
8. **Cell 7 (Code)**: Install Kaggle CLI (`!pip install kaggle -q`)
9. **Cell 8 (Code)**: Submit to Kaggle
   - Loads credentials from `.env` (parent directory)
   - Submits to competition

## What's NOT Included
- ❌ HuggingFace file downloading/uploading
- ❌ Kaggle dataset publishing
- ❌ Term filtering to 13,500
- ❌ IA-weighted aggregation
- ❌ GO hierarchy propagation

## Prerequisites

### Required Files
Must exist before running the notebook:
- `cafa6_data/parsed/train_seq.feather`
- `cafa6_data/parsed/train_terms.tsv`
- `cafa6_data/parsed/test_seq.feather`
- `cafa6_data/parsed/train_taxa.feather`
- `cafa6_data/parsed/test_taxa.feather`
- `cafa6_data/features/train_embeds_esm2_3b.npy`
- `cafa6_data/features/test_embeds_esm2_3b.npy`
- `cafa6_data/features/train_embeds_*.npy` (t5, esm2, ankh, text)
- `cafa6_data/features/test_embeds_*.npy` (t5, esm2, ankh, text)
- `cafa6_data/features/IA.tsv`
- `cafa6_data/go-basic.obo`

### Optional Files
- `cafa6_data/features/aspect_thresholds.json` (for per-aspect thresholding)
- `.env` file in parent directory with `KAGGLE_USERNAME` and `KAGGLE_KEY`

## Usage

### 1. Run All Cells
Simply execute all cells in order:
```bash
# In Jupyter/Colab
Run → Run All Cells
```

### 2. Force Retraining
By default, the notebook retrains KNN every time. To skip retraining and load existing predictions:
```python
# Before running Cell 5
import os
os.environ['FORCE_RETRAIN'] = '0'
```

### 3. Kaggle Submission
Create `.env` file in parent directory:
```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Then run Cells 7 and 8 to install Kaggle CLI and submit.

## Troubleshooting

### Issue: "FileNotFoundError: Missing KNN test predictions"
**Solution:** Run Cell 5 (KNN training) first with `FORCE_RETRAIN=1`

### Issue: "Kaggle CLI not found"
**Solution:** Run Cell 7 to install: `!pip install kaggle -q`

### Issue: "Kaggle credentials not found"
**Solution:** Create `.env` file with `KAGGLE_USERNAME` and `KAGGLE_KEY` in parent directory

### Issue: Low F1 score (~0.07)
**Solution:** This is the bug we fixed! Make sure you're running the latest version with:
- Full term vocabulary (no 13,500 filtering)
- Per-protein max normalization
- No IA weighting during aggregation
- No GO hierarchy propagation

## Performance Expectations

| Metric | E2E Pipeline (Broken) | Baseline-Aligned (Fixed) |
|--------|----------------------|--------------------------|
| F1 Score | ~0.072 | ~0.20-0.25 |
| Term Universe | 13,500 | ~25,000+ |
| Score Aggregation | IA-weighted (wrong) | Pure similarity (correct) |
| Normalization | None (wrong) | Per-protein max (correct) |
| Best Threshold | N/A | ~0.3-0.4 |

## Key Differences from Baseline KNN

While this notebook follows baseline methodology, it uses:
- **Better embeddings:** ESM2-3B (vs ESM2-8M in baseline)
- **More modalities:** T5, ESM2-650M, ESM2-3B, Ankh, Text, Taxa (vs single ESM2-8M)
- **Expected improvement:** Richer embeddings should provide 10-20% F1 boost over baseline

