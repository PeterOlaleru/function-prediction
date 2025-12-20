# Notebook Audit: Colab_04b_first_submission_no_ankh.ipynb

## 1. Phase 2a: Logistic Regression (Cell 8b)
- [x] **Streaming Logic**: Verify `X_train_mmap.npy` creation handles all modalities (t5, esm2, esm2_3b, text, taxa).
- [x] **Memory Safety**: Ensure `mmap_mode='r'` is used during training.
- [x] **Model Config**: Check `OneVsRestClassifier` instantiation (CPU vs GPU/RAPIDS).
- [x] **Regex/Attribute Error**: Verify the fix for `predict_proba` availability on `clf_chunk` (the "regex issue" likely referred to `AttributeError: 'OneVsRestClassifier' object has no attribute 'predict_proba'` or similar).

## 2. Phase 2b: GBDT (Cell 8b continued)
- [x] **Input Data**: Verify it uses the same `X` (mmap) as LogReg.
- [x] **Library**: Check `py_boost` import and usage.
- [x] **Output**: Ensure `oof_pred_gbdt.npy` is saved correctly.

## 3. Phase 2c: DNN (Cell 8c/9)
- [x] **Dataset Class**: Verify `CAFA6Dataset` handles memory-mapped numpy arrays correctly (no accidental copies to RAM).
- [x] **Architecture**: Check input dimension matching (`n_features`).
- [x] **Training Loop**: Verify batching and GPU usage.

## 4. Phase 3: Level-2 Stacking (Cell 10/11)
- [x] **Inputs**: Verify it loads `oof_pred_logreg.npy`, `oof_pred_gbdt.npy`, `oof_pred_dnn.npy`.
- [x] **External Priors**: Verify `prop_train_no_kaggle.tsv` usage.
- [x] **Logic**: Check GCN or Stacker implementation.

## 5. Phase 4: Submission (Cell 12+)
- [x] **Test Data**: Verify test feature loading (streaming/mmap for test set?).
- [x] **Inference**: Check prediction generation for all Level-1 models on Test.
- [ ] **Formatting**: Verify `submission.tsv` format (EntryID, term, score).

## 6. Global Checks
- [x] **File Paths**: Consistency of `WORK_ROOT` usage.
- [ ] **Dependencies**: `requirements.txt` vs imports.
