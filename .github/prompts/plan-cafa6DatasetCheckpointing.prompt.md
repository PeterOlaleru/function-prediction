## Decisions locked in
- Backend: Kaggle Dataset (canonical store)
- Checkpointing: Milestones (not every cell)
- Policy: Idempotent stages + manifest validation + minimal uploads

## Recommended approach
### Replace `ARTEFACTS_DIR` with a checkpoint store
- Keep a local cache (ephemeral per runtime): `WORK_ROOT`
- Treat the Kaggle Dataset as truth: `DATASET_ID`
- Two core operations used everywhere:
  - `store.pull()` — hydrate local cache from the latest dataset version (fresh machine/resume)
  - `store.push(stage_name, paths=[...])` — publish a new dataset version containing new/updated artefacts

Sceptic check: “Scrap ARTEFACTS_DIR” cannot mean “no local folder” (impossible). It should mean “local is never authoritative”.

## Milestone stages (minimal but sufficient)
1. stage_01_parsed
   - `parsed/train_seq.feather`, `parsed/test_seq.feather`, `parsed/train_terms.parquet`, `parsed/train_taxa.feather`, `parsed/test_taxa.feather`, `parsed/term_priors.parquet`
2. stage_02_external_text
   - `external/entryid_text.tsv` (caches optional)
3. stage_03_tfidf_text
   - `features/text_vectorizer.joblib`, `features/train_embeds_text.npy`, `features/test_embeds_text.npy`
4. stage_04_external_goa_priors
   - `external/prop_train_no_kaggle.tsv.gz`, `external/prop_test_no_kaggle.tsv.gz`
5. stage_05_embeddings_t5
   - `features/train_embeds_t5.npy`, `features/test_embeds_t5.npy`
6. stage_06_embeddings_esm2
   - `features/train_embeds_esm2.npy`, `features/test_embeds_esm2.npy`
7. stage_07_level1_preds
   - `features/oof_*.npy`, `features/test_pred_*.npy` (plus any reusable fitted artefacts)
8. stage_08_stacker
   - `features/top_terms_1500.json`, `features/test_pred_gcn.npy`
9. stage_09_submission
   - `submission.tsv`

## Manifest + diagnostics
- After each `push()`, write/update `manifest.json` with:
  - stage name, timestamp, file list, sizes
  - row-count sanity checks (train/test)
  - optional hashes
- Add a diagnostics cell that:
  - plots top file sizes (bar plot)
  - plots file size distribution (histogram)

## Practical Kaggle constraints
- Upload time + dataset version churn will dominate if we checkpoint too often.
- Embedding artefacts are large; if the dataset becomes unwieldy:
  - split into two Kaggle datasets:
    - `…-core-checkpoints` (parsed/external/tfidf/priors/preds/submission)
    - `…-embeddings` (t5/esm2)
  - keep a single manifest that records both dataset IDs.

## Implementation steps (notebook edits)
1. Refactor Cell 3 (setup) to create `STORE` (cache root + Kaggle push/pull).
2. Update each milestone cell to follow a strict pattern:
   - `STORE.pull()` (once per run)
   - if outputs exist: skip compute
   - else compute → write locally
   - `STORE.push(stage_name, paths=...)`
3. Update the artefact manifest cell to read from the store root and to write `manifest.json`.

## Open choice
- Dataset visibility: private (recommended) vs public.
