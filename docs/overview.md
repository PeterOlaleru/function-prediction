# 🧬 CAFA-6 Protein Function Prediction - CONSOLIDATED OVERVIEW

**Date:** 25 Nov 2025  
**Current Best Model:** KNN with aspect-specific thresholds — F1 = 0.2579  
**Keep for detailed execution:** `docs/PLAN.md`

---
## 1. 🎯 Problem Snapshot
Predict multi-label Gene Ontology (GO) terms (MF, BP, CC) for protein sequences.  
Challenge: Extreme class imbalance (≈26k terms, ~6 positives per protein).  
Metric: Weighted F1 (information accretion).

---
## 2. ✅ What’s Already Done
- Data ingestion (FASTA, GO terms, ontology, taxonomy, IA weights)
- EDA completed (length distributions, term frequency, ontology coverage)
- Baselines: Frequency (0.1412), KNN with embeddings (0.1776), MLP (0.1672)
- Fine-tuning pipeline built (tokenisation, Dataset, Trainer)
- Threshold optimisation (grid over 0.01–0.50)
- Asymmetric Loss integrated (gamma_neg=2.0, clip=0.05)
- Best F1 improved to 0.2331 (Precision 0.3397, Recall 0.2379, Thr=0.40)

---
## 3. 📊 Current Performance
| Model | F1 | Notes |
|-------|----|-------|
| Frequency | 0.1619 | Predicts common terms (per-aspect metric) |
| KNN (single threshold 0.40) | 0.2520 | Homology-style transfer |
| **KNN (aspect-specific thresholds)** | **0.2579** | MF=0.40, BP=0.20, CC=0.40 🏆 |
| KNN + Label Propagation | 0.2181 | ❌ Propagation hurts (-15%) |
| MLP (Frozen embeddings) | 0.1672 | Underperformed KNN |
| ESM-2 Fine-Tuned (BCE) | 0.1806 | Needed threshold tuning |
| ESM-2 Fine-Tuned (Asym Loss) | 0.2331 | Needs re-eval with per-aspect metric |

Trajectory target: Short-term 0.25–0.27; Mid-term 0.30+; Long-term 0.35–0.40.

---
## 4. 🔥 Immediate High-ROI Actions (Next 2 Months)
| Action | Impact | Effort | Notes |
|--------|--------|--------|-------|
| ~~Label propagation~~ | ❌ -15% | 2h | Does NOT work with KNN (errors amplify) |
| Per-aspect thresholds | ✅ +2.3% | 1h | MF=0.40, BP=0.20, CC=0.40 → F1=0.2579 |
| ESM-2 150M (1 epoch validation) | +0.06–0.09 | 4h | Pipeline validation before 650M |
| ESM-2 650M (competition model) | +0.08–0.12 | 60h | Main model, multi-GPU setup |
| Simple ensemble (KNN + ESM-650M) | +0.02–0.04 | 1h | Weighted average |

Recommended order: Propagation (Week 1) → 150M validation (Week 2) → 650M training (Week 3-5) → Ensemble + tuning (Week 6-7).

## 4a. ✅ Progress Checklist
- [x] Data ingestion (01, 02, 03, 04)
- [x] Exploratory data analysis (EDA) — DELETED
- [x] Baselines: Frequency (01) — **Per-aspect CAFA metric implemented**  F1: 0.1619
- [x] Baselines: KNN (02) — **Per-aspect CAFA metric implemented** F1: 0.2520
- [x] Baselines: MLP — SKIPPED (underperformed KNN)
- [x] Fine-tuning pipeline (ESM-2 8M) (03) — **Per-aspect CAFA metric implemented**
- [x] Threshold optimisation (global sweep) (03)
- [x] Asymmetric loss integration (03)
- [x] **Per-aspect evaluation (MF/BP/CC split)** — Competition metric now correctly implemented in ALL notebooks
- [x] Label propagation (ancestor add) (04) — **FAILED: -15% F1 with KNN (error amplification)**
- [x] Per-aspect thresholds (MF/BP/CC) (04) — **DONE: MF=0.40, BP=0.20, CC=0.40 → F1=0.2579**
- [ ] ESM-2 150M validation (1 epoch) (03 - modify MODEL_NAME to facebook/esm2_t30_150M_UR50D)
- [ ] ESM-2 650M training (03 - modify MODEL_NAME to facebook/esm2_t33_650M_UR50D, implement multi-GPU)
- [ ] Simple ensemble (KNN + ESM-650M) (05 - new notebook)
- [x] CAFA5-style stacker prototype (min/max GO propagation + averaging) — added `notebooks/06_cafa5_style_stack.ipynb`
- [ ] Expand GO vocabulary (10k terms) (03 - modify VOCAB_SIZE)
- [ ] Increase max sequence length (448→1024 residues) (03 - modify max_length)
- [ ] Evolutionary features (MSA / PSSM) (06 - future)
- [ ] Structure features (AlphaFold embeddings) (06 - future)
- [ ] Domain features (Pfam) (06 - future)
- [ ] GO term embeddings (text + graph) (07 - future)
- [ ] Hierarchy consistency loss (03 - modify loss function)
- [ ] Hard negative mining (03 - modify training loop)
- [ ] Data tier weighting (evidence levels) (03 - modify dataset)

---
## 4b. 📚 Plain-English Feature Cheatsheet
> **Analogy:** Solving a crime with better clues & tools

**CRITICAL:** Competition evaluates **three subontologies separately** (MF, BP, CC), then averages. **ALL notebooks (01, 02, 03, 04) now correctly implement this per-aspect evaluation.** This was missed initially — all previous F1 scores were computed incorrectly by mixing aspects together.

**Data ingestion** ✅  
Loading protein sequences, GO annotations, ontology structure, taxonomy mapping, and IA weights from raw files. Like gathering all evidence at a crime scene — foundation for everything (+baseline).

**Exploratory data analysis (EDA)** ✅  
Understanding data distributions, sequence lengths, term frequencies, and class imbalance. Like profiling suspects before investigation — reveals what you're up against (+insight).

**Baselines (Frequency, KNN, MLP)** ✅  
Simple models to beat: predicting common terms, nearest-neighbour transfer, shallow neural nets. Like starting with obvious suspects — establishes minimum performance bar (F1 0.14–0.18).

**Fine-tuning pipeline (ESM-2 8M)** ✅  
Training protein language model end-to-end on GO prediction task. Like teaching a detective domain-specific skills — learns task-relevant patterns (F1 0.18→0.23).

**Threshold optimisation (global sweep)** ✅  
Finding best confidence cutoff for predictions across all terms. Like calibrating when to make an arrest — critical for imbalanced data (+0.18 F1).

**Asymmetric loss integration** ✅  
Down-weighting easy negatives, focusing on hard positives. Like spending investigation time on unclear cases, not obvious innocents — handles extreme imbalance (F1 0.18→0.23, +29%).

**Label propagation** (ancestor propagation) ❌ FAILED  
If you predict a very specific term, auto-add its broader parents. Like saying "Golden Retriever" implies "Dog". **Does NOT work with KNN** — predictions are too noisy (~60-70% accuracy), so propagation amplifies errors up the hierarchy. Result: -15% F1. May help with high-accuracy deep learning models (>90% accuracy).

**Per-aspect evaluation** ✅  
Competition metric: compute F1 separately for MF, BP, CC, then average the three. Not a single F1 across all terms. Kaggle does this automatically; local validation must match. Now correctly implemented in notebooks 01-02.

**Per-aspect thresholds** (separate tuning per subontology)  
After fixing evaluation, next step: find optimal threshold separately for MF, BP, CC instead of global threshold. MF might need 0.45, CC needs 0.35. Optimises precision/recall trade-off per domain (+0.01–0.02).

**Simple ensemble** (KNN + ESM weighted average)  
Combine homology-based (KNN) with learned patterns (ESM). Like asking both an experienced practitioner and an AI — they catch different errors (+0.01–0.02).

**Larger backbone** (ESM-2 35M vs current 8M)  
More parameters = better pattern recognition. Like upgrading from a pocket calculator to a supercomputer — captures subtler amino acid relationships (+0.03–0.05).

**Expand GO vocabulary** (10k terms vs current 5k)  
Cover more rare functions. Like expanding your dictionary from common words to technical jargon — improves rare term recall (+0.01–0.02).

**Increase max sequence length** (1024 vs current 512 residues)  
Don't truncate long proteins. Like reading full book chapters instead of summaries — preserves context for large proteins (+0.01–0.02).

**Evolutionary features** (MSA/PSSM)  
Asking a protein's relatives what they do. Multiple sequence alignment shows conserved "important" positions — like interviewing a big family. Strong lift (+0.03–0.05).

**Structure features** (AlphaFold)  
Knowing the 3D shape, not just the letters. Like seeing how a folded tool fits into a machine → reveals functional pockets (+0.02–0.04).

**Domain features** (Pfam)  
Predefined Lego blocks. If you spot a known block, you guess its role faster. Small but steady gain (+0.01–0.02).

**GO embeddings** (Text + Graph)  
Turning term definitions + hierarchy into numbers. Like mapping related job titles ("chef", "cook", "baker") closer together → helps predict rare terms (+0.03–0.05).

**Hierarchy loss** (consistency penalty)  
Enforces parent ≥ child logic. Like making sure you don't claim "Brakes specialist" without "Mechanic". Cleans logical mistakes (+0.01–0.02).

**Hard negative mining** (adaptive sampling)  
Drill on the mistakes you keep making. Like flashcards of the ones you get wrong — sharpens discrimination (+0.01–0.02).

**Data tiers** (evidence-based weighting)  
Trust high-quality annotations more. Like weighting eyewitnesses over rumours — reduces noise (+0.01–0.03).

---
## 5. 🔮 Strategic Roadmap (2-Month GPU Timeline)
| Tier | Goal | Feature Set | Timeline | Est. F1 Gain |
|------|------|-------------|----------|-----------|
| Week 1 | 0.26 | KNN + Aspect-specific thresholds | ✅ Done | F1=0.2579 |
| Week 2-3 | 0.34–0.37 | ESM-2 150M (validation) | 4h | +0.06–0.09 |
| Week 4-5 | 0.38–0.40 | ESM-2 650M (main model) | 60h | +0.08–0.12 |
| Week 6-7 | 0.40–0.42 | Ensemble + per-aspect thresholds + calibration | 8h | +0.02–0.04 |
| Week 8 (optional) | 0.42+ | GO embeddings / hierarchy loss / MSA features | 20h | +0.02–0.03 |

---
## 6. 🐛 Issues Solved
| Problem | Cause | Fix | Result |
|---------|-------|-----|--------|
| F1 = 0.0000 | Threshold 0.5 too high | Grid search thresholds | F1 → 0.1806 |
| Poor focus on positives | BCE treats all negatives equally | AsymmetricLoss | F1 → 0.2331 |
| Loss explosion (446) | `.sum()` in custom loss | Use `.mean()` | Stable training |
| Evaluation mismatch (04 vs 02) | Matrix-based eval differed from DataFrame | Use exact KNN notebook function | F1 synced at 0.2520 |
| Label propagation hurt F1 | KNN predictions too noisy for propagation | Skip propagation for KNN | Baseline preserved |
| BP threshold suboptimal | Global 0.40 too high for BP | Per-aspect: BP=0.20 | F1 0.252→0.2579 |

---
## 7. 💡 Lessons Learned
- Threshold tuning is mandatory for extreme imbalance.  
- Focal-style (asymmetric) loss outperforms vanilla BCE here.  
- Fine-tuning backbone > frozen embeddings + shallow head.  
- Early stopping protects against plateau wastage.  
- Homology (KNN) remains a strong complementary signal.
- **Label propagation requires high-accuracy base predictions (>90%).** KNN at ~60-70% accuracy → propagation amplifies errors.
- **Per-aspect thresholds matter:** BP needs lower threshold (0.20) than MF/CC (0.40).
- **Always verify evaluation consistency** between notebooks before comparing results.

---
## 8. 🗂️ Key Files (Active Set)
| Area | File |
|------|------|
| Data | `src/data/loaders.py`, `src/data/finetune_dataset.py` |
| Models | `src/models/esm_classifier.py`, `src/models/baseline_embedding_knn.py`, `src/models/baseline_frequency.py` |
| Training | `src/training/finetune_esm.py`, `src/training/loss.py`, `src/training/trainer.py` |
| Saved Model | `models/esm_finetuned/best_model/` |

---
## 9. 🧪 Evaluation Approach (Current)
- Collect logits → sigmoid probabilities.  
- Global threshold chosen via validation sweep.  
- Macro label averaging (sample-wise F1).  
Next upgrades: Per-aspect thresholds, per-term dynamic thresholding, calibration (temperature / isotonic).

---
## 10. 🚀 Next Concrete Steps (Immediate)
1. ~~Complete notebook 04 (label propagation)~~ — ✅ Done, propagation doesn't help KNN.
2. Re-evaluate ESM-2 8M with per-aspect metric (may beat KNN's 0.2579).
3. Scale to ESM-2 150M for 1 epoch (pipeline validation) — 4h.  
4. Implement ESM-2 650M training:
   - Update MODEL_NAME to `facebook/esm2_t33_650M_UR50D`
   - Set batch_size=1, gradient_accumulation=16
   - Consider multi-GPU DDP if 2+ GPUs available
   - Add mixed precision (fp16) for memory efficiency
   - Train full 10 epochs — ~60h GPU time
5. Build ensemble (KNN + ESM-650M weighted average) — 1h.  
6. Tune per-aspect thresholds (MF/BP/CC separate) — 1h.  

---
## 11. 📈 Success Targets (2-Month Timeline)
| Milestone | Success Criteria | Week |
|-----------|------------------|------|
| KNN + Aspect Thresholds | F1 = 0.2579 ✅ | 1 |
| ESM-2 150M (validation) | F1 ≥ 0.34 | 2-3 |
| ESM-2 650M (main) | F1 ≥ 0.38 | 4-5 |
| Ensemble + Tuning | F1 ≥ 0.40 | 6-7 |
| Optional Polish | F1 ≥ 0.42 | 8 |

---
## 12. ❓ Open Decisions
| Decision | Options | Recommendation |
|----------|---------|----------------|
| Backbone scale | 150M vs 650M | 150M for 1-epoch validation, then 650M |
| Multi-GPU strategy | DDP vs DataParallel | DDP (2 devices, batch_size=1, grad_accum=16) |
| Term count | 5k vs 10k vs 15k | Start 10k with 650M |
| Sequence length | 512 vs 1024 | Start 448 with 650M, expand to 1024 if memory allows |
| Ensemble strategy | Simple avg vs stacking | Weighted avg (70% ESM-650M + 30% KNN) |

---
## 13. 🧠 Future Enhancements (Outline)
- Multi-modal fusion: Cross-attention across sequence, profile, structure embeddings.  
- GO term embedding: Text + graph + co-occurrence → similarity scoring head.  
- Hierarchy consistency loss: Penalise child > parent probability gaps.  
- Hard negative mining: Replay buffer of frequent false positives.  
- Active learning loop: Surface high-uncertainty proteins.

---
## 14. 🔍 Risk Check
| Risk | Mitigation |
|------|------------|
| GPU memory with 650M model | batch_size=1 + grad_accum=16 + mixed precision fp16 |
| 650M training time (60h) | Use 150M validation run first to catch bugs in 4h |
| Multi-GPU setup complexity | Optional — single GPU works, DDP only if 2+ available |
| MSA generation cost | Skip for initial 650M run, add only if time remains |
| Ontology misuse | Validate propagation coverage + ancestor integrity |
| Overfitting with 650M | Early stopping + validation monitoring |

---
## 15. 🏁 Summary in One Line
KNN baseline with aspect-specific thresholds achieves F1=0.2579 (propagation failed). Next: re-evaluate ESM-2 with per-aspect metric, then scale to 150M/650M over 2-month GPU timeline.

---
**If happy with this consolidation, I can remove `ROADMAP.md`, `SUMMARY.md`, `PROGRESS_TRACKER.md` and keep only `PLAN.md` + `OVERVIEW.md`.**
