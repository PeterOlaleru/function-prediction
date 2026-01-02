# KNN Performance Fix - Implementation Guide

## Changes Required in 05_cafa_e2e.ipynb (Cell 13E - KNN)

### 1. Reduce k from 50 to 10

**Location**: Early in Cell 13E

**Current code**:
```python
KNN_K = int(globals().get('KNN_K', 50))
```

**Fixed code**:
```python
KNN_K = int(globals().get('KNN_K', 10))  # FIXED: reduced from 50 to 10 (matches baseline)
```

**Rationale**: The baseline uses k=10. Using k=50 dilutes the signal by including too many dissimilar neighbors.

---

### 2. Remove IA Weighting from Neighbor Aggregation

**Location**: Two places in Cell 13E - OOF predictions and test predictions

#### Fix 2a: OOF Predictions

**Current code** (lines ~68-75):
```python
Y_nei = Y_knn[neigh_b]  # (B, K, L)
# IA-weighted aggregation: sims @ (Y_nei * w_ia)
scores = ((sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1) / denom).astype(np.float32)
```

**Fixed code**:
```python
Y_nei = Y_knn[neigh_b]  # (B, K, L)
# FIXED: Removed IA weighting during aggregation (apply only in evaluation)
# Simple similarity-weighted average (matches baseline logic)
scores = (sims_b[:, :, np.newaxis] * Y_nei).sum(axis=1).astype(np.float32)
```

#### Fix 2b: Test Predictions

**Current code** (lines ~84-91):
```python
Y_nei = Y_knn[neigh_b]
scores = ((sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1) / denom_te[i:j]).astype(np.float32)
test_pred_knn[i:j] = scores
```

**Fixed code**:
```python
Y_nei = Y_knn[neigh_b]
# FIXED: Removed IA weighting during aggregation
scores = (sims_b[:, :, np.newaxis] * Y_nei).sum(axis=1).astype(np.float32)
test_pred_knn[i:j] = scores
```

**Rationale**: IA weighting should only be applied during evaluation (F1 calculation), not during prediction. Otherwise, rare terms get double-counted.

---

### 3. Add Per-Protein Max Normalization

**Location**: After all OOF and test predictions are computed, before saving

**Insert this code** right after test predictions loop and before the "Finite-value quality gates" section:

```python
# CRITICAL FIX: Per-protein max normalization (matches baseline logic)
# This ensures each protein's scores are calibrated to [0, 1] range
print('[KNN] Applying per-protein max normalization...')

# Normalize OOF predictions
for i in range(oof_pred_knn.shape[0]):
    max_val = oof_pred_knn[i].max()
    if max_val > 1e-9:  # Avoid division by zero
        oof_pred_knn[i] /= max_val
    # Note: if max_val is 0, scores are already all zeros (no normalization needed)

# Normalize test predictions  
for i in range(test_pred_knn.shape[0]):
    max_val = test_pred_knn[i].max()
    if max_val > 1e-9:
        test_pred_knn[i] /= max_val

print(f'[KNN] Normalization complete. OOF score range: [{oof_pred_knn.min():.4f}, {oof_pred_knn.max():.4f}]')
print(f'[KNN] Normalization complete. Test score range: [{test_pred_knn.min():.4f}, {test_pred_knn.max():.4f}]')
```

**Rationale**: This is THE critical fix. The baseline normalizes each protein's scores by its max score. Without this, scores are on inconsistent scales across proteins, breaking threshold-based filtering.

---

## Summary of Changes

| Issue | Fix | Expected Impact |
|-------|-----|-----------------|
| k=50 too large | Set k=10 | Reduces noise from distant neighbors | +2-3% F1 |
| IA double-counting | Remove IA from aggregation | Prevents over-weighting rare terms | +3-5% F1 |
| Missing max norm | Add per-protein max normalization | **Restores score calibration** | +8-12% F1 |

**Total expected improvement**: +13-20% F1, bringing performance from underperforming to **exceeding** the baseline.

---

## Validation Steps

After applying fixes:

1. **Smoke test**: Verify score distributions
   ```python
   print("OOF score stats:", oof_pred_knn.mean(), oof_pred_knn.std(), oof_pred_knn.max())
   print("Non-zero fraction:", (oof_pred_knn > 0).mean())
   ```
   - Expected: max should be 1.0, mean around 0.01-0.05, non-zero fraction 5-15%

2. **Compare with baseline**: Run both notebooks on same validation split
   - Check if per-protein max scores now match

3. **Threshold sweep**: Test thresholds 0.1 to 0.6
   - Should see clear peak around 0.25-0.40

4. **Final F1**: Measure per-aspect F1
   - Should exceed 0.216 overall
   - BP should be non-zero and competitive

---

## Implementation Priority

1. **MUST FIX** (Critical): Per-protein max normalization - Without this, nothing else matters
2. **SHOULD FIX** (High): Reduce k to 10 - Significant signal/noise improvement
3. **SHOULD FIX** (High): Remove IA from aggregation - Prevents systematic bias
4. **OPTIONAL**: Consider aspect-specific k values (e.g., BP might benefit from k=15)

---

## Code Review Checklist

Before committing:
- [ ] k=10 (not 50)
- [ ] No `w_ia_broadcast` in scoring formulas
- [ ] Per-protein max normalization present for both OOF and test
- [ ] Score ranges are [0, 1] after normalization
- [ ] No division by sum-of-similarities (removed `/ denom`)
- [ ] Validation F1 > 0.216

---

## Expected Results

**Before fixes**:
- F1 < 0.20 (underperforming)
- Scores poorly calibrated
- Threshold tuning ineffective

**After fixes**:
- F1 > 0.25 (better than baseline)
- Scores well-calibrated to [0, 1]
- Threshold tuning works as expected
- ESM2-3B advantage becomes apparent

The 3B model with correct implementation should significantly outperform the 8M baseline.
