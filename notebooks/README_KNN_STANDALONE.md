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
   - **Dynamic F1 evaluation** (calculated, not hardcoded)
   - **FORCE_RETRAIN defaults to True** (always retrains)
   - Saves OOF and test predictions
7. **Cell 6 (Code)**: Generate submission.tsv (NEW)
   - Loads KNN test predictions
   - Applies hierarchy propagation (Max/Min)
   - Applies per-aspect thresholds
   - Formats submission per CAFA rules
   - Saves to `submission.tsv`
8. **Cell 7 (Code)**: Install Kaggle CLI (NEW)
   - Installs kaggle package via pip
9. **Cell 8 (Code)**: Submit to Kaggle (NEW)
   - Loads credentials from `.env` (parent directory)
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
- `FORCE_RETRAIN`: Set to `0` to skip retraining if predictions exist (default is `1` - always retrain)
- `KNN_K`: Number of neighbors (default: 10)
- `KNN_BATCH`: Batch size for predictions (default: 256)
- `KAGGLE_USERNAME`: Your Kaggle username (for submission - loaded from parent dir `.env`)
- `KAGGLE_KEY`: Your Kaggle API key (for submission - loaded from parent dir `.env`)

### Running the Notebook
1. Ensure all prerequisites are in place
2. Run cells sequentially:
   - Cell 1: Environment setup
   - Cell 2: Configuration
   - Cell 3: Data loading
   - Cell 4: KNN helper functions
   - Cell 5: KNN training (generates predictions + **dynamic F1 evaluation**)
     - **Note:** FORCE_RETRAIN defaults to True (always retrains)
     - Set `FORCE_RETRAIN=0` to skip retraining if predictions exist
   - Cell 6: Generate submission.tsv (optional)
   - Cell 7: Install Kaggle CLI (optional)
   - Cell 8: Submit to Kaggle (optional - requires credentials in parent dir `.env`)

### Output
The notebook produces:
- `features/level1_preds/oof_pred_knn.npy`: Out-of-fold predictions
- `features/level1_preds/test_pred_knn.npy`: Test set predictions
- `features/oof_pred_knn.npy`: Backward-compatible copy
- `features/test_pred_knn.npy`: Backward-compatible copy
- `submission.tsv`: Competition submission file (if Cell 6 is run)
- **F1 scores**: Dynamically calculated and printed during training

## Important Notes

### Score Calibration (Critical Fix)

**Version 2 Fix:** Removed per-protein max normalization that was destroying score calibration.

**Previous issue:**
- Cell 5 was normalizing each protein's scores by dividing by the max value
- This made every protein have max score = 1.0
- Low thresholds (0.05-0.10) then let almost everything through
- Result: Over-prediction and poor F1 scores (~0.07)

**Current implementation:**
- Scores are IA-weighted and normalized by sum of similarities
- Scores naturally range in [0, 1] without per-protein normalization
- Proper score calibration maintained across proteins
- Expected F1: ~0.25-0.30 with appropriate thresholds

**Recommended thresholds:**
- Test multiple values: 0.01, 0.02, 0.05, 0.10, 0.15, 0.20
- Cell 5 automatically tests these and reports best threshold
- Per-aspect thresholds from training data are preferred

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

### Kaggle Submission (Cell 8)

**Setting up Kaggle credentials:**

The notebook loads credentials from a `.env` file in the **parent directory** (one level up from where the notebook runs). Create a `.env` file at that location with:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

**File location:**
- If running from `/workspace/`, create `.env` at `/` (parent directory)
- If running from a project subfolder, create `.env` in the parent folder

**Get your API key:**
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json` containing your credentials
5. Extract the username and key from the file

**Install Kaggle CLI:**
Cell 7 automatically installs the Kaggle CLI with:
```bash
!pip install kaggle
```

**Alternative:** Set environment variables directly:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

If you encounter issues submitting to Kaggle:

**Kaggle CLI not installed:**
Run Cell 7 to install it automatically, or manually:
```bash
pip install kaggle
```

**API credentials not configured:**
The cell will automatically load from parent directory's `.env` file.

**Submission failed:**
- Check that `submission.tsv` exists in `WORK_ROOT`
- Verify you have accepted the competition rules
- Check file format: tab-separated, no header, 3 columns (EntryID, term, score)
- Ensure daily submission limit not reached

**Customize submission message:**
Edit the `SUBMISSION_MESSAGE` variable in Cell 8:
```python
SUBMISSION_MESSAGE = 'Your custom message here'
```

### Missing Files
If you get "Missing required modality 'esm2_3b'" error:
- Ensure `features/train_embeds_esm2_3b.npy` exists
- Ensure `features/test_embeds_esm2_3b.npy` exists

### KNN Training Issues

**Training always runs (FORCE_RETRAIN=True by default):**
The notebook now retrains by default to ensure fresh predictions. To skip retraining if predictions exist:
```python
import os
os.environ['FORCE_RETRAIN'] = '0'
# Then run Cell 5
```

Or set the environment variable before starting Jupyter:
```bash
export FORCE_RETRAIN=0
jupyter notebook
```

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
