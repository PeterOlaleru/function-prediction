# KNN Standalone Notebook

## Overview
`knn_standalone.ipynb` is a streamlined notebook extracted from `05_cafa_e2e.ipynb` that runs **only** the KNN model training without the overhead of HuggingFace file downloading/uploading.

## What's Included
This notebook contains the minimal required cells to train the KNN model and generate a Kaggle submission:

1. **Cell 0 (Markdown)**: Title and introduction
2. **Cell 1 (Code)**: Basic environment setup (NO REPO)
3. **Cell 2 (Code)**: Simplified configuration
   - Environment detection (Kaggle/Colab/Local)
   - WORK_ROOT setup
   - TRAIN_LEVEL1 flag
   - Stub CheckpointStore (no HuggingFace operations)
4. **Cell 3 (Code)**: Data loading (Phase 2 setup)
   - Loads training/test features
   - Loads target labels (Y)
   - Defines helper functions
5. **Cell 4 (Code)**: KNN helper functions
   - L2 normalization function
   - IA weights loading
   - Y_knn preparation
   - X_knn_test preparation
6. **Cell 5 (Code)**: KNN training (FIXED - fully functional)
   - cuML/sklearn KNN with cosine similarity
   - 5-fold cross-validation
   - IA-weighted neighbor voting
   - Per-protein max normalization
   - **Dynamic F1 evaluation** (calculated, not hardcoded)
   - Saves OOF and test predictions
7. **Cell 6 (Code)**: Generate submission.tsv (NEW)
   - Loads KNN test predictions
   - Applies hierarchy propagation (Max/Min)
   - Applies per-aspect thresholds
   - Formats submission per CAFA rules
   - Saves to `submission.tsv`
8. **Cell 7 (Code)**: Submit to Kaggle (NEW)
   - Submits `submission.tsv` to the competition
   - Uses Kaggle CLI
   - Customizable submission message

## What's NOT Included
This notebook intentionally excludes:
- ❌ HuggingFace file downloading (Cells 02b, 02c)
- ❌ HuggingFace file uploading
- ❌ Kaggle dataset publishing
- ❌ Checkpoint pull/push operations
- ❌ Unnecessary setup cells
- ❌ Other model training cells (GBDT, DNN, Logistic Regression)

## Prerequisites
Before running this notebook, you must have:

### Required Files
- Pre-computed embeddings: `features/train_embeds_esm2_3b.npy`
- Pre-computed embeddings: `features/test_embeds_esm2_3b.npy`
- Parsed training data: `parsed/train_terms.parquet`
- Parsed training sequences: `parsed/train_seq.feather`
- Parsed test sequences: `parsed/test_seq.feather`
- Top terms list: `features/top_terms_13500.json`
- IA weights: `IA.tsv`

### Directory Structure
```
cafa6_data/
├── features/
│   ├── train_embeds_esm2_3b.npy
│   ├── test_embeds_esm2_3b.npy
│   ├── top_terms_13500.json
│   └── level1_preds/         # Output directory
├── parsed/
│   ├── train_terms.parquet
│   ├── train_seq.feather
│   └── test_seq.feather
└── IA.tsv
```

## Usage

### Environment Variables
- `CAFA_TRAIN_LEVEL1`: Set to `1` to train (default), `0` to skip
- `KNN_K`: Number of neighbors (default: 10)
- `KNN_BATCH`: Batch size for predictions (default: 256)

### Running the Notebook
1. Ensure all prerequisites are in place
2. Run cells sequentially:
   - Cell 1: Environment setup
   - Cell 2: Configuration
   - Cell 3: Data loading
   - Cell 4: KNN helper functions
   - Cell 5: KNN training (generates predictions + **dynamic F1 evaluation**)
   - Cell 6: Generate submission.tsv (optional)
   - Cell 7: Submit to Kaggle (optional - requires Kaggle CLI)

### Output
The notebook produces:
- `features/level1_preds/oof_pred_knn.npy`: Out-of-fold predictions
- `features/level1_preds/test_pred_knn.npy`: Test set predictions
- `features/oof_pred_knn.npy`: Backward-compatible copy
- `features/test_pred_knn.npy`: Backward-compatible copy
- `submission.tsv`: Competition submission file (if Cell 6 is run)
- **F1 scores**: Dynamically calculated and printed during training

## Important Notes

### Fixed Issues
This standalone notebook includes fixes for issues in the original e2e notebook:
- ✓ Added missing `_l2_norm` function
- ✓ Added missing `Y_knn` variable setup
- ✓ Added missing `X_knn_test` variable setup
- ✓ Added missing `weights_full` loading
- ✓ Fixed incomplete sklearn KNN initialization in fold loop
- ✓ Fixed missing KNN fitting and prediction code in CV folds
- ✓ Added proper similarity conversion for both cuML and sklearn backends

The KNN implementation in this notebook is **fully functional** and will run successfully.

### Output
The notebook produces:
- `features/level1_preds/oof_pred_knn.npy`: Out-of-fold predictions
- `features/level1_preds/test_pred_knn.npy`: Test set predictions
- `features/oof_pred_knn.npy`: Backward-compatible copy
- `features/test_pred_knn.npy`: Backward-compatible copy

## Performance
- **Runtime**: ~30-60 minutes on A100 GPU (with cuML)
- **Memory**: ~20-30GB RAM, ~10GB VRAM
- **Expected F1**: ~0.25-0.26 (IA-weighted, dynamically calculated)

## F1 Evaluation
The notebook includes **dynamic F1 evaluation** in Cell 5:
- Calculates IA-weighted F1 at multiple thresholds (0.1, 0.2, 0.3, 0.4, 0.5)
- Reports precision and recall for each threshold
- Identifies the best threshold automatically
- F1 scores are **calculated on the fly**, not hardcoded

Example output:
```
  Threshold   F1      Precision  Recall
  ------------------------------------------
  0.10       0.2234  0.1523      0.4156
  0.20       0.2512  0.2145      0.3234
  0.30       0.2579  0.2567      0.2591
  0.40       0.2401  0.2987      0.2012
  0.50       0.2145  0.3245      0.1678
  ------------------------------------------
  Best: F1=0.2579 @ threshold=0.30
```

## Differences from Full Pipeline
The CheckpointStore in this notebook is a stub that:
- Does **not** download files from HuggingFace
- Does **not** upload files to HuggingFace
- Only prints status messages

This means you must manually ensure all required files are present before running.

## Troubleshooting

### Kaggle Submission (Cell 7)
If you encounter issues submitting to Kaggle:

**Kaggle CLI not found:**
```bash
pip install kaggle
```

**API credentials not configured:**
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Save `kaggle.json` to:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

**Submission failed:**
- Check that `submission.tsv` exists in `WORK_ROOT`
- Verify you have accepted the competition rules
- Check file format: tab-separated, no header, 3 columns (EntryID, term, score)

**Customize submission message:**
Edit the `SUBMISSION_MESSAGE` variable in Cell 7:
```python
SUBMISSION_MESSAGE = 'Your custom message here'
```

### Missing Files
If you get "Missing required modality 'esm2_3b'" error:
- Ensure `features/train_embeds_esm2_3b.npy` exists
- Ensure `features/test_embeds_esm2_3b.npy` exists

### Missing Dependencies
If you get import errors:
- Install cuML for GPU acceleration (optional)
- Install sklearn, numpy, pandas (required)

### Performance Issues
If KNN is too slow:
- Install cuML for 10-20x speedup on A100 GPU
- Reduce `KNN_K` for faster neighbor search
- Increase `KNN_BATCH` for better GPU utilization

## Related Files
- `05_cafa_e2e.ipynb`: Full end-to-end pipeline
- `02_baseline_knn.ipynb`: Original KNN baseline implementation
- `docs/KNN_FIX_IMPLEMENTATION.md`: KNN implementation details
- `docs/KNN_PERFORMANCE_ANALYSIS.md`: KNN performance analysis
