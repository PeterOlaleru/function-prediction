# Colab Notebook Optimizations & Fixes
**Date:** 2025-12-18
**Context:** Fixes applied to `Colab_04b_first_submission_no_ankh.ipynb` to resolve crashes (OOM), 404 errors, and stability issues. These should be ported to `Colab_04_all_in_one.ipynb`.

## 1. Kaggle Checkpoint Pull (404 & Crash Fixes)
**Problem:**
- `api.dataset_download_file` returns 404 for nested files (e.g., `parsed/term_counts.parquet`) if the dataset wasn't uploaded with `--dir-mode zip`.
- `subprocess.run(['kaggle', ...])` can cause SIGKILL (-9) on Colab due to high memory usage during unzip.

**Solution (`KaggleCheckpointStore.pull`):**
- **Hybrid Download Strategy:**
    1.  Try Python API (`api.dataset_download_file`) first.
    2.  If 404/Error, fallback to CLI (`kaggle datasets download ...`).
    3.  **Structure Repair:** CLI flattens nested files (e.g., `parsed/foo` -> `foo`). The script now detects this and moves the file back to `parsed/foo`.
- **Full Fallback:** If individual file downloads fail, it downloads the full zip but **only extracts** the requested folders (`parsed/`, `external/`) to avoid filling disk space.

## 2. RAM Optimization (Phase 2 / Cell 13)
**Problem:**
- Holding `features_train` (dict of arrays) AND `X` (flat concatenated array) simultaneously exceeds 12GB RAM.

**Solution:**
- **Sequential Loading:**
    1.  Load `features_train` (dict).
    2.  Build `X` (flat).
    3.  `del features_train; gc.collect()` (**Crucial Step**).
    4.  Train LR & GBDT using `X`.
    5.  `del X; gc.collect()`.
    6.  Reload `features_train` (dict) for DNN training.- **Feature Reduction:**
    - **Excluded `esm2_3b`** from the "Flat" models (LR, GBDT).
    - `esm2_3b` is 2560 dimensions. Including it alongside T5 (1024) + ESM2-650M (1280) creates a massive array.
    - **Update (High-RAM):** User has 53GB RAM. Reverted to use **ALL** features including `esm2_3b`.
    - **Aggressive GC:** Added explicit `del` and `gc.collect()` between `X` and `X_test` creation to minimize peak RAM.
    - **Sequential Loading:** Refactored `load_features_dict` to support `split='train'` and `split='test'` modes.
    - **Batched Prediction:**
        - **Problem:** Creating `X_test` (21GB) while holding `features_test` (21GB) and `X` (8GB) caused OOM (50GB+).
        - **Solution:** Removed `X_test` creation. Implemented `predict_proba_batched` to generate `X_test` chunks on the fly during prediction.
        - Peak RAM is now ~30GB (safe for 53GB runtime).
- **Self-Contained Scope:**
    - Imports (`pd`, `np`, `gc`, `os`) are inside the `if TRAIN_LEVEL1:` block.
    - `WORK_ROOT` is auto-recovered if missing.

## 3. Dataset Hygiene (Ankh Removal)
**Problem:**
- `features.zip` contained massive (~2GB) Ankh embeddings that caused OOM/Timeouts.

**Solution:**
- Added `CAFA_CLEAN_CHECKPOINT_REMOVE_ANKH` logic.
- Filters out `*ankh*.npy` files during republishing.
- Default pull is now "Lean": `parsed.zip,external.zip` only.

## 4. Visualization
- **FASTA Lengths:** Updated to use `bins=np.linspace(0, 3000, 60)` and `density=True` to handle outliers and compare Train vs Test distributions properly.
