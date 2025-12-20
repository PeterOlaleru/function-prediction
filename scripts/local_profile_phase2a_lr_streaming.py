from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import psutil
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def iter_chunks(idxs: np.ndarray, chunk_rows: int):
    for i in range(0, len(idxs), chunk_rows):
        yield idxs[i : i + chunk_rows]


def main() -> int:
    ap = argparse.ArgumentParser(description='Local profiling: streaming scaler + streaming OVR-SGD fold')
    ap.add_argument('--x-mmap', type=Path, required=True, help='Path to X_train_mmap.npy')
    ap.add_argument('--y-npy', type=Path, required=True, help='Path to Y_train.npy (binary indicator)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--chunk-rows', type=int, default=2000)
    ap.add_argument('--max-rows', type=int, default=12000, help='Limit rows for quick local test')
    ap.add_argument(
        '--n-labels',
        type=int,
        default=8,
        help='Number of label columns to train (keeps sanity-check fast when Y is multi-label)',
    )
    args = ap.parse_args()

    X = np.load(args.x_mmap, mmap_mode='r')
    Y = np.load(args.y_npy, mmap_mode='r')

    n = min(args.max_rows, X.shape[0])
    X = X[:n]
    Y = Y[:n]

    print(f'X={X.shape} dtype={X.dtype} (mmap)')
    print(f'Y={Y.shape} dtype={Y.dtype}')
    print(f'RSS start: {rss_gb():.2f} GB')

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    idx_tr, idx_val = next(iter(kf.split(np.arange(n))))

    # 1) streaming scaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    for ch in iter_chunks(idx_tr, args.chunk_rows):
        Xch = np.asarray(X[ch], dtype=np.float32)
        scaler.partial_fit(Xch)
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    print(f'RSS after scaler pass: {rss_gb():.2f} GB')

    # 2) streaming multi-label SGD (explicit per-label loop)
    if Y.ndim == 1:
        Y2 = Y.reshape(-1, 1)
    else:
        Y2 = Y

    n_labels = min(int(args.n_labels), int(Y2.shape[1]))
    models = [
        SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=1, tol=None)
        for _ in range(n_labels)
    ]

    first = True
    for ch in iter_chunks(idx_tr, args.chunk_rows):
        Xch = np.asarray(X[ch], dtype=np.float32)
        Xch = (Xch - mean) / scale
        Ych = np.asarray(Y2[ch, :n_labels], dtype=np.int64)

        if first:
            for j, m in enumerate(models):
                m.partial_fit(Xch, Ych[:, j], classes=np.array([0, 1], dtype=np.int64))
            first = False
        else:
            for j, m in enumerate(models):
                m.partial_fit(Xch, Ych[:, j])

    print(f'RSS after training pass: {rss_gb():.2f} GB')

    # 3) streaming predict
    probs = np.zeros((len(idx_val), n_labels), dtype=np.float32)
    off = 0
    for ch in iter_chunks(idx_val, args.chunk_rows):
        Xch = np.asarray(X[ch], dtype=np.float32)
        Xch = (Xch - mean) / scale
        # models[j].predict_proba returns (n, 2), we want prob of class 1
        p = np.stack([m.predict_proba(Xch)[:, 1] for m in models], axis=1).astype(np.float32)
        probs[off : off + len(ch)] = p
        off += len(ch)

    print(f'RSS after predict pass: {rss_gb():.2f} GB')
    print('OK: streaming fold completed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
