import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import gc
import psutil
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Setup Paths
DEBUG_ROOT = Path('artefacts_local/debug_download').resolve()
WORK_ROOT = DEBUG_ROOT

print(f"DEBUG_ROOT: {DEBUG_ROOT}")

# Fix Directory Structure (Move parsed files if needed)
parsed_dir = DEBUG_ROOT / 'parsed'
parsed_dir.mkdir(exist_ok=True)

for f in ['train_terms.parquet', 'train_seq.feather', 'train_taxa.feather']:
    src = DEBUG_ROOT / f
    dst = parsed_dir / f
    if src.exists():
        print(f"Moving {src} to {dst}")
        shutil.move(str(src), str(dst))

# Mock Global Variables
TRAIN_LEVEL1 = True

def log_mem(tag=""):
    try:
        mem = psutil.virtual_memory()
        print(f"[MEM] {tag:<30} | Used: {mem.used/1e9:.2f}GB / {mem.total/1e9:.2f}GB ({mem.percent}%)")
    except:
        pass

# Load Targets
print("\n--- Loading Targets ---")
try:
    train_terms = pd.read_parquet(WORK_ROOT / 'parsed' / 'train_terms.parquet')
    print(f"train_terms shape: {train_terms.shape}")
    print(train_terms.head())
except Exception as e:
    print(f"FAILED to load train_terms: {e}")
    exit(1)

# Load IDs
train_ids = pd.read_feather(WORK_ROOT / 'parsed' / 'train_seq.feather')['id'].astype(str)
print(f"train_ids shape: {train_ids.shape}")

# Construct Y (Simplified for Debug)
print("Constructing Y (Simplified)...")
# Use top 100 terms for debug speed
top_terms = train_terms['term'].value_counts().head(100).index.tolist() 
print(f"Selected {len(top_terms)} top terms for debug.")

train_terms_top = train_terms[train_terms['term'].isin(top_terms)]
Y_df = train_terms_top.pivot_table(index='EntryID', columns='term', aggfunc='size', fill_value=0)

# Align with train_ids
# Notebook says: train_ids_clean = train_ids.str.extract(r'\|(.*?)\|')[0]
train_ids_clean = train_ids.str.extract(r'\|(.*?)\|')[0].fillna(train_ids)
Y_df = Y_df.reindex(train_ids_clean, fill_value=0)
Y = Y_df.values.astype(np.float32)
print(f"Y shape: {Y.shape}")

# Load Features (Mock load_features_dict)
print("\n--- Loading Features (Subset) ---")
FEAT_DIR = WORK_ROOT / 'features'

# We will just load T5 and ESM2 for debug
features_train = {}
# Note: Using mmap_mode='r' to avoid loading full file, then slicing
for key, fname in [('t5', 'train_embeds_t5.npy'), ('esm2', 'train_embeds_esm2.npy')]:
    p = FEAT_DIR / fname
    if p.exists():
        print(f"Loading {fname}...")
        # Load only first 1000 rows to save time/RAM
        arr = np.load(p, mmap_mode='r')
        print(f"  Original shape: {arr.shape}")
        features_train[key] = arr[:1000] # Slice!
    else:
        print(f"WARNING: {fname} not found.")

if not features_train:
    print("ERROR: No features found. Cannot proceed.")
    exit(1)

# Create Flat X
print("Creating Flat X...")
FLAT_KEYS = list(features_train.keys())
X = np.concatenate([features_train[k] for k in FLAT_KEYS], axis=1)
print(f"X shape: {X.shape}")

# Slice Y to match X
Y_subset = Y[:1000]
print(f"Y_subset shape: {Y_subset.shape}")

# Train Loop
print("\n--- Starting Training Loop (CPU) ---")
kf = KFold(n_splits=2, shuffle=True, random_state=42)

for fold, (idx_tr, idx_val) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    log_mem("Start Fold")
    
    X_tr, X_val = X[idx_tr], X[idx_val]
    Y_tr, Y_val = Y_subset[idx_tr], Y_subset[idx_val]
    
    print(f"  Train shapes: X={X_tr.shape}, Y={Y_tr.shape}")
    
    # Scaling
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    
    # Model
    print("  Training SGDClassifier...")
    clf = OneVsRestClassifier(SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=100, tol=1e-3, n_jobs=4))
    clf.fit(X_tr, Y_tr)
    print("  Training done.")
    
    # Predict
    val_probs = clf.predict_proba(X_val)
    print(f"  Preds shape: {val_probs.shape}")
    
    log_mem("End Fold")
    break # One fold is enough

print("\nSUCCESS: Local training debug passed.")