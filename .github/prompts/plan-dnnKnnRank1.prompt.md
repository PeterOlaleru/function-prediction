# Rank-1 DNN & KNN Implementation — Auditor-Approved v2

**Date:** 31 Dec 2025  
**Status:** Planning Complete → Awaiting Implementation  
**Auditor:** Approved with mandatory additions (modality-specific scaling, cuML migration, IA-weighted voting, degenerate-awareness)  
**Target Cells:** 13D (DNN), 13E (KNN)  
**Expected Impact:** +12-17% IA-F1 boost (validated against Rank-1 benchmarks)

---

## Critical Prerequisites

- [ ] **Cell 13F executed first** — Generates `aspect_thresholds.json` (BP: 0.25, MF: 0.35, CC: 0.35)
- [ ] **IA.tsv available** — Required for IA-weighted loss/voting
- [ ] **top_terms_13500.json exists** — Label contract from Cell 13a
- [ ] **Colab A100 environment** — cuML KNN requires Linux (not supported on Windows local)

---

## Phase 1: DNN Upgrades (Cell 13D)

### 1.1 Architecture: Modality-Specific Head Sizing ⚡
**Auditor Requirement:** Dynamic layer sizing prevents parameter blow-up on Text (12,288D) while ensuring ESM2-3B (5,120D) capacity.

- [ ] Modify `ModalityHead.__init__()` in [05_cafa_e2e.ipynb](../notebooks/05_cafa_e2e.ipynb)
- [ ] Implement dynamic sizing logic:
  - `input_dim >= 8000` → `[1024, 1024, 512]` (3 layers)
  - `input_dim >= 2000` → `[2048, 1024, 512]` (3 layers)
  - `else` → `[1024, 512, 256]` (3 layers)
- [ ] Add third FC layer to all modality heads (deeper representation)
- [ ] Verify `MultiBranchDNN` fusion trunk still concatenates correctly
- [ ] Test instantiation with actual feature dimensions (local smoke test)

### 1.2 Extreme Ensembling (5×5) with Aggressive Cleanup ⚡
**Auditor Requirement:** Same fix pattern as LogReg v1.1 — prevents cuBLAS handle accumulation across 25 models.

- [ ] Wrap existing 5-fold loop with outer `for seed in range(5):` loop
- [ ] Set `torch.manual_seed(42 + seed)` at start of each seed iteration
- [ ] After each seed completes (5 folds done), add aggressive cleanup:
  ```python
  del model
  gc.collect()
  torch.cuda.empty_cache()
  ```
- [ ] Verify OOF/test prediction accumulators handle 25 models correctly
- [ ] Add heartbeat logging: `[Seed {seed+1}/5] Fold {fold+1}/5 | Epoch {epoch+1}/3`

### 1.3 Degenerate-Aware Prediction Helper ⚡
**Auditor Requirement:** Prevents AttributeErrors on GO terms with <50 training positives (mirrors LogReg `safe_predict_proba_gpu`).

- [ ] Modify `predict_on_split()` function
- [ ] Add finite-value check after sigmoid:
  ```python
  probs = torch.sigmoid(logits).cpu().numpy()
  # Fallback for degenerate terms (all-zero or all-NaN columns)
  for col_idx in range(probs.shape[1]):
      if np.isnan(probs[:, col_idx]).all() or (probs[:, col_idx] == 0).all():
          probs[:, col_idx] = 1e-5  # Near-zero probability
  ```
- [ ] Test with synthetic degenerate labels (GO term with 0 positives in fold)
- [ ] Verify no NaN/Inf values propagate to OOF/test memmaps

### 1.4 IA-Weighted Loss Integration ⚡
**Status:** ✅ Already implemented (per subagent findings)

- [x] IA weights loaded from `IA.tsv` and mapped to `top_terms`
- [x] Per-term weights broadcast as `(1, 13500)` tensor in loss:
  ```python
  loss_per = F.binary_cross_entropy_with_logits(logits, yb, reduction='none')
  loss = (loss_per * w).mean()  # w: IA weights
  ```
- [ ] **Verify weights are non-uniform** — add diagnostic print:
  ```python
  print(f"IA weight range: [{w.min():.4f}, {w.max():.4f}], mean={w.mean():.4f}")
  ```

### 1.5 Aspect-Specific Thresholds ⚡
**Auditor Requirement:** Proven +3.3% F1 boost from KNN baseline.

- [ ] Load `aspect_thresholds.json` at cell start:
  ```python
  thr_path = FEAT_DIR / 'aspect_thresholds.json'
  if not thr_path.exists():
      raise FileNotFoundError("Run Cell 13F first to generate aspect_thresholds.json")
  aspect_thresholds = json.loads(thr_path.read_text())
  ```
- [ ] Apply aspect-specific thresholds in validation scoring loop:
  ```python
  for aspect in ['BP', 'MF', 'CC']:
      thr = aspect_thresholds[aspect]
      preds_binary = (oof_probs[:, aspect_mask] > thr).astype(int)
  ```
- [ ] Remove hardcoded global `thr=0.3`

---

## Phase 2: KNN Upgrades (Cell 13E)

### 2.1 cuML Migration with L2 Pre-Normalization ⚡
**Auditor Requirement:** Transforms cosine→dot-product (Manual GEMM Fast Path); 10-20× speedup on A100.

- [ ] Add runtime detection for cuML availability:
  ```python
  try:
      from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
      USE_CUML = True
  except ImportError:
      from sklearn.neighbors import NearestNeighbors
      USE_CUML = False
      print("[WARN] cuML not available; using sklearn (slower)")
  ```
- [ ] Pre-normalize ESM2-3B embeddings to unit L2 norm **before** fitting:
  ```python
  def _l2_norm(x, eps=1e-12):
      norms = np.linalg.norm(x, axis=1, keepdims=True)
      return x / np.maximum(norms, eps)
  
  X_knn = _l2_norm(features_train['esm2_3b'])
  X_knn_test = _l2_norm(features_test['esm2_3b'])
  ```
- [ ] For cuML: use `metric='euclidean'` (normalized vectors make L2=cosine)
- [ ] For sklearn fallback: keep `metric='cosine'`
- [ ] Verify cuML index fit succeeds on Colab A100
- [ ] Measure speedup: log fit + inference time (target: <5 min for full train set)

### 2.2 IA-Weighted Neighbor Voting ⚡
**Auditor Requirement:** Prioritizes rare/high-value terms; expected +4-6% IA-F1 boost.

- [ ] Load IA weights (same logic as DNN/LogReg):
  ```python
  ia_path = next((p for p in [WORK_ROOT/'IA.tsv', FEAT_DIR/'IA.tsv'] if p.exists()), None)
  ia_df = pd.read_csv(ia_path, sep='\t', header=None, names=['term', 'ia'])
  ia_map = dict(zip(ia_df['term'].astype(str), ia_df['ia'].astype(np.float32)))
  weights_full = np.asarray([ia_map.get(t, 1.0) for t in top_terms], dtype=np.float32)
  ```
- [ ] Modify voting logic to multiply neighbor contributions by IA weights:
  ```python
  # Before: scores = (sims @ Y_nei) / denom
  # After: scores = (sims @ (Y_nei * w_ia)) / denom
  w_ia_broadcast = weights_full[np.newaxis, np.newaxis, :]  # (1, 1, L)
  scores = (sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1) / denom
  ```
- [ ] Verify shape: `scores.shape == (batch_size, 13500)`
- [ ] Add diagnostic: print IA weight impact on top-5 predicted terms

### 2.3 Aspect-Specific Thresholds ⚡
**Status:** Same as DNN (shared requirement).

- [ ] Load `aspect_thresholds.json` (same guard as DNN)
- [ ] Apply per-aspect thresholds in OOF/test prediction loops
- [ ] Remove hardcoded global `thr=0.3`

### 2.4 Finite-Value Quality Gates ⚡
**Auditor Requirement:** Prevents NaN/Inf propagation to GCN stacker.

- [ ] Add quality check after KNN prediction:
  ```python
  assert np.isfinite(oof_pred_knn).all(), "KNN OOF contains NaN/Inf"
  assert np.isfinite(test_pred_knn).all(), "KNN test contains NaN/Inf"
  ```
- [ ] If assertion fails, clip to valid range:
  ```python
  oof_pred_knn = np.clip(oof_pred_knn, 0.0, 1.0)
  test_pred_knn = np.clip(test_pred_knn, 0.0, 1.0)
  ```

---

## Phase 3: Integration & Validation

### 3.1 Execution Order Enforcement
- [ ] Add dependency check at start of Cells 13D and 13E:
  ```python
  if not (FEAT_DIR / 'aspect_thresholds.json').exists():
      raise FileNotFoundError("Run Cell 13F first to generate aspect thresholds")
  ```
- [ ] Document execution order in cell markdown:
  > **Prerequisites:** Cell 13F must run first to calibrate aspect-specific thresholds.

### 3.2 Local Smoke Tests (Before Colab)
- [ ] Create `scripts/test_dnn_modality_heads.py` — verify dynamic sizing with mock inputs
- [ ] Create `scripts/test_knn_ia_voting.py` — verify IA-weighted scoring on synthetic data
- [ ] Run both scripts on Windows local (CPU-only, small sample)
- [ ] Estimate VRAM usage: DNN peak memory per model, KNN index size

### 3.3 Colab A100 Full Run
- [ ] Upload updated notebook to Colab
- [ ] Run Cell 13F → verify `aspect_thresholds.json` created
- [ ] Run Cell 13D (DNN) → monitor for:
  - [ ] 25 models complete (5 seeds × 5 folds)
  - [ ] No VRAM OOM errors
  - [ ] OOF predictions saved to `oof_pred_dnn.npy`
  - [ ] Test predictions saved to `test_pred_dnn.npy`
  - [ ] Runtime <20 min per seed (target: <100 min total)
- [ ] Run Cell 13E (KNN) → monitor for:
  - [ ] cuML index fit succeeds
  - [ ] IA-weighted voting executes without errors
  - [ ] OOF predictions saved to `oof_pred_knn.npy`
  - [ ] Test predictions saved to `test_pred_knn.npy`
  - [ ] Runtime <10 min total (cuML speedup)

### 3.4 Ablation Diagnostics (Cell 18)
- [ ] Run ablation study to measure impact:
  - [ ] DNN baseline (no IA-weighting) vs. IA-weighted
  - [ ] KNN uniform voting vs. IA-weighted voting
  - [ ] Global threshold (0.3) vs. aspect-specific thresholds
- [ ] Expected IA-F1 improvements:
  - [ ] DNN IA-weighting: +4-6% F1
  - [ ] KNN IA-weighting: +4-6% F1
  - [ ] Aspect thresholds: +3.3% F1
  - [ ] **Combined: +12-17% F1 boost**

---

## Phase 4: Artifact Synchronization

### 4.1 STORE.maybe_push Integration
- [ ] Add checkpoint push at end of Cell 13D:
  ```python
  if 'STORE' in globals() and STORE is not None:
      STORE.maybe_push(
          stage='stage_07b_level1_dnn_rank1',
          required_paths=[
              str(PRED_DIR / 'oof_pred_dnn.npy'),
              str(PRED_DIR / 'test_pred_dnn.npy'),
              str(FEAT_DIR / 'top_terms_13500.json'),
          ],
          note='Rank-1 DNN: 5x5 Extreme Ensembling + IA-Weighted BCE'
      )
  ```
- [ ] Add checkpoint push at end of Cell 13E:
  ```python
  if 'STORE' in globals() and STORE is not None:
      STORE.maybe_push(
          stage='stage_07c_level1_knn_rank1',
          required_paths=[
              str(PRED_DIR / 'oof_pred_knn.npy'),
              str(PRED_DIR / 'test_pred_knn.npy'),
          ],
          note='Rank-1 KNN: cuML + IA-Weighted Voting'
      )
  ```

### 4.2 Progress Tracking
- [ ] Update [docs/overview.md](../docs/overview.md) section 4a:
  - [ ] Tick "DNN Rank-1 Upgrades Complete"
  - [ ] Tick "KNN Rank-1 Upgrades Complete"
- [ ] Update [docs/PLAN.md](../docs/PLAN.md):
  - [ ] Mark Phase 2 (Level-1 Models) as "In Progress → Complete"
  - [ ] Update next priority: "Phase 3 (GCN Stacker)"

---

## Success Criteria

### Performance Targets
- [ ] **DNN Runtime:** <20 min per seed (total <100 min for 5 seeds)
- [ ] **KNN Runtime:** <10 min total (cuML acceleration)
- [ ] **VRAM Stability:** No OOM errors across 25 DNN models
- [ ] **Quality Gates:** Zero NaN/Inf in OOF/test predictions

### Metric Targets
- [ ] **IA-F1 Improvement:** +12-17% on validation set (vs baseline DNN/KNN)
- [ ] **Aspect-Specific F1:**
  - BP: ≥0.30 (current: ~0.20)
  - MF: ≥0.50 (current: ~0.40)
  - CC: ≥0.50 (current: ~0.40)

### Code Quality
- [ ] **No regression:** LogReg Cell 13c still runs without errors
- [ ] **Degenerate-aware:** No crashes on rare GO terms (<50 positives)
- [ ] **Reproducible:** Deterministic results with fixed seeds (42, 43, 44, 45, 46)

---

## Rollback Plan

If Colab run fails or IA-F1 boost <8%:

1. **Immediate:** Revert to baseline DNN/KNN (commit hash before changes)
2. **Diagnose:** Check ablation study logs for which upgrade failed
3. **Incremental:** Re-apply upgrades one at a time (IA-weighting → thresholds → cuML)
4. **Escalate:** If cuML causes instability, document and use sklearn fallback

---

## Notes

- **Windows Local Limitation:** cuML KNN requires Linux; local testing uses sklearn fallback
- **Colab Secrets:** Ensure `userdata.get('KAGGLE_USERNAME')` etc. are set for artifact push
- **Cell 13F Dependency:** Never skip threshold calibration; it's mandatory for Rank-1 compliance
- **A100 Handle Limit:** DNN cleanup logic prevents same issue that crashed LogReg at chunk 6

**Last Updated:** 31 Dec 2025  
**Next Review:** After Colab A100 full run completes
