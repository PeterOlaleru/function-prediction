# KNN Standalone Notebook

## Overview
`knn_standalone.ipynb` is a streamlined notebook extracted from `05_cafa_e2e.ipynb` that runs **only** the KNN model training without the overhead of HuggingFace file downloading/uploading.

## What's Included
This notebook contains the minimal required cells to train the KNN model:

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
   - Loads IA weights
   - Defines helper functions
5. **Cell 4 (Code)**: KNN training
   - cuML/sklearn KNN with cosine similarity
   - 5-fold cross-validation
   - IA-weighted neighbor voting
   - Per-protein max normalization
   - Saves OOF and test predictions

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
│   └── train_seq.feather
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
   - Cell 4: KNN training

### Output
The notebook produces:
- `features/level1_preds/oof_pred_knn.npy`: Out-of-fold predictions
- `features/level1_preds/test_pred_knn.npy`: Test set predictions
- `features/oof_pred_knn.npy`: Backward-compatible copy
- `features/test_pred_knn.npy`: Backward-compatible copy

## Performance
- **Runtime**: ~30-60 minutes on A100 GPU (with cuML)
- **Memory**: ~20-30GB RAM, ~10GB VRAM
- **Expected F1**: ~0.25-0.26 (IA-weighted)

## Differences from Full Pipeline
The CheckpointStore in this notebook is a stub that:
- Does **not** download files from HuggingFace
- Does **not** upload files to HuggingFace
- Only prints status messages

This means you must manually ensure all required files are present before running.

## Troubleshooting

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
