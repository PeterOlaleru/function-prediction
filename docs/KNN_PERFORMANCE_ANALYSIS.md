# KNN Performance Gap Analysis - Diagnostic Report

## Executive Summary

**Problem**: The older KNN baseline (02_baseline_knn.ipynb) achieves F1 > 0.216, while the newer end-to-end pipeline (05_cafa_e2e.ipynb) with a larger ESM2-3B model + GCN stacking significantly underperforms.

**Root Causes Identified**:
1. **Different scoring aggregation logic** (critical bug)
2. **5x more neighbors** diluting signal (k=50 vs k=10)
3. **Matrix-based prediction** missing per-protein normalization
4. **IA-weighted averaging** changes score distribution
5. **No aspect-specific threshold tuning** in final predictions

---

## Detailed Comparison

### 1. Embedding Model
- **Old KNN**: `facebook/esm2_t6_8M_UR50D` (8M parameters, 320 dimensions)
- **New Pipeline**: `esm2_t36_3B_UR50D` (3B parameters, much higher dimensional)
- **Impact**: Larger model should be BETTER, not worse. This rules out model quality as the issue.

### 2. KNN Configuration
| Parameter | Old KNN (02) | New Pipeline (05) | Impact |
|-----------|--------------|-------------------|--------|
| **k (neighbors)** | 10 | 50 | 5x more neighbors dilutes signal from closest matches |
| **Metric** | cosine | euclidean on L2-norm | Mathematically equivalent but different code paths |
| **Distance → Similarity** | `1 - distance` | `1 - distance` (clipped) | Same concept |

**Critical Issue**: k=50 is too many neighbors. With k=10, only the most similar proteins vote. With k=50, you're averaging in proteins that may not be functionally similar at all.

### 3. Scoring/Aggregation Logic - **THE SMOKING GUN**

#### Old KNN (02_baseline_knn.ipynb) - CORRECT ✅
```python
for val_protein in val_proteins:
    term_scores = Counter()
    
    for nei_idx, distance in zip(neighbour_indices, neighbour_distances):
        nei_protein = train_proteins[nei_idx]
        nei_terms = protein_to_terms.get(nei_protein, [])
        
        similarity = 1 - distance
        
        for term in nei_terms:
            term_scores[term] += similarity  # Accumulate weighted votes
    
    # Normalize scores to [0, 1] range PER PROTEIN
    if term_scores:
        max_score = max(term_scores.values())
        for term, score in term_scores.items():
            predictions.append({
                'probability': score / max_score  # Per-protein normalization
            })
```

**Key features**:
- Iterates per protein
- Accumulates similarity-weighted votes per term
- **Normalizes by max_score PER PROTEIN** (critical!)
- This ensures each protein's predictions are calibrated to [0, 1]

#### New Pipeline (05_cafa_e2e.ipynb) - BUGGY ❌
```python
# Matrix-based batch processing
for i in range(0, len(va_idx), KNN_BATCH):
    j = min(i + KNN_BATCH, len(va_idx))
    neigh_b = neigh_global[i:j]
    sims_b = sims[i:j]
    denom = np.maximum(sims_b.sum(axis=1, keepdims=True), 1e-8)
    
    Y_nei = Y_knn[neigh_b]  # (B, K, L) - batch, neighbors, labels
    
    # IA-weighted aggregation
    scores = ((sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1) / denom).astype(np.float32)
    #                                                                              ^^^^^^
    #                      WRONG! Dividing by sum of similarities, NOT max_score
```

**Critical Bug**:
- Divides by `sum of similarities` across neighbors, not by `max_score` per protein
- This is fundamentally different from the old implementation
- **Missing per-protein max normalization step**
- IA weighting further distorts the score distribution

**Impact**: 
- Scores are no longer calibrated to [0, 1] properly
- The relative scale between proteins is broken
- Threshold tuning becomes ineffective
- Rare terms with high IA weights get artificially boosted

### 4. IA Weighting

**Old KNN**: No IA weighting during neighbor aggregation (IA weights only used in evaluation)

**New Pipeline**: Applies IA weights during neighbor voting:
```python
scores = ((sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1) / denom)
                                              ^^^^^^^^^^^^^^^^
```

**Problem**: 
- Rare terms get up-weighted during prediction
- But evaluation also up-weights rare terms
- **Double-counting** the importance of rare terms
- This breaks the calibration that thresholds rely on

### 5. Evaluation Differences

**Old KNN**:
- Explicit per-aspect threshold tuning (MF=0.50, BP=0.25, CC=0.35)
- Aspect-specific F1 maximization
- Clear evaluation function with threshold filtering

**New Pipeline**:
- Threshold tuning code exists but may not be applied correctly in final predictions
- Complex multi-stage pipeline makes it unclear where thresholds are applied
- OOF predictions go through additional GCN stacking

---

## Why Small ESM Can Outperform Large ESM

The paradox: **8M-parameter model beats 3B-parameter model + GCN stacking**

### Explanation:

1. **Model quality is NOT the issue** - ESM2-3B embeddings are objectively better
2. **The issue is in the PREDICTION LOGIC**, not the embeddings
3. **Key failures in new pipeline**:
   - Wrong normalization (sum of similarities vs max_score)
   - Too many neighbors (k=50 vs k=10)
   - IA double-counting
   - Missing per-protein calibration

4. **Why this matters**:
   - The old KNN produces well-calibrated probabilities [0, 1]
   - Thresholds (0.25-0.50) work correctly on calibrated scores
   - The new KNN produces poorly-calibrated scores
   - No amount of threshold tuning can fix mis-calibrated scores

5. **Analogy**: 
   - Old KNN: Good thermometer (cheap) reading temperature correctly
   - New KNN: Excellent thermometer (expensive) but using wrong calibration scale
   - The expensive thermometer gives worse results because of calibration bug, not because it's a worse sensor

---

## Recommendations

### Immediate Fixes (Priority Order):

1. **Fix score normalization** (CRITICAL):
   ```python
   # After neighbor aggregation, add per-protein max normalization:
   for i in range(oof_pred_knn.shape[0]):
       max_val = oof_pred_knn[i].max()
       if max_val > 0:
           oof_pred_knn[i] /= max_val
   ```

2. **Reduce k from 50 to 10-15**:
   - More neighbors ≠ better for KNN
   - Keep only the most similar proteins

3. **Remove IA weighting from neighbor aggregation**:
   - Apply IA weights only during evaluation, not during prediction
   - Prevents double-counting

4. **Verify aspect-specific thresholds are applied**:
   - Ensure MF/BP/CC use their optimized thresholds
   - Not just a single global threshold

5. **Test incrementally**:
   - Fix normalization first → measure F1
   - Then reduce k → measure F1  
   - Then remove IA from aggregation → measure F1
   - Each fix should show improvement

### Validation Protocol:

1. Create a small test with known proteins
2. Compare old KNN vs new KNN predictions for same proteins
3. Verify score distributions match
4. Verify threshold behavior matches
5. Only then run full evaluation

---

## Conclusion

**The new pipeline doesn't underperform because it uses a worse model.**

**It underperforms because the prediction logic has bugs:**
- Missing per-protein max normalization
- Too many neighbors
- IA double-counting
- Potentially incorrect threshold application

**Expected outcome after fixes**: The ESM2-3B KNN should **exceed** the 8M baseline performance, not fall short. The better embeddings should translate to better predictions once the bugs are fixed.

**Confidence level**: High - the scoring logic difference is unambiguous and directly explains the performance gap.
