from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def cpu_info() -> str:
    try:
        return f"cpus={os.cpu_count()} rss={rss_gb():.2f}GB"
    except Exception:
        return f"cpus={os.cpu_count()}"


def parse_go_obo_edges(obo_path: Path, allowed_terms: set[str]) -> tuple[int, int]:
    """Very small OBO parser: counts nodes and is_a edges within allowed_terms.

    Returns (n_terms_found, n_is_a_edges_in_subset).
    """

    term_id: str | None = None
    in_term = False
    n_terms_found = 0
    n_edges = 0

    id_re = re.compile(r"^id:\s*(GO:\d+)")
    isa_re = re.compile(r"^is_a:\s*(GO:\d+)")

    with obo_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "[Term]":
                term_id = None
                in_term = True
                continue
            if line.startswith("[") and line != "[Term]":
                in_term = False
                term_id = None
                continue
            if not in_term:
                continue

            m = id_re.match(line)
            if m:
                term_id = m.group(1)
                if term_id in allowed_terms:
                    n_terms_found += 1
                continue

            if term_id is None or term_id not in allowed_terms:
                continue

            m = isa_re.match(line)
            if m:
                parent = m.group(1)
                if parent in allowed_terms:
                    n_edges += 1

    return n_terms_found, n_edges


def open_memmap(path: Path, shape: tuple[int, int], mode: str, dtype=np.float32):
    return np.lib.format.open_memmap(str(path), mode=mode, dtype=dtype, shape=shape)


def fill_random_normal(mm: np.ndarray, chunk_rows: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    for start in range(0, mm.shape[0], chunk_rows):
        end = min(start + chunk_rows, mm.shape[0])
        mm[start:end] = rng.standard_normal((end - start, mm.shape[1]), dtype=np.float32)
    mm.flush()


def fit_scaler_state(X_mm: np.ndarray, chunk_rows: int) -> dict[str, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    for start in range(0, X_mm.shape[0], chunk_rows):
        end = min(start + chunk_rows, X_mm.shape[0])
        xb = np.asarray(X_mm[start:end], dtype=np.float32)
        scaler.partial_fit(xb)
    return {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
    }


def transform_to_memmap(X_mm: np.ndarray, state: dict[str, np.ndarray], out_path: Path, chunk_rows: int) -> np.ndarray:
    out = open_memmap(out_path, shape=X_mm.shape, mode="w+", dtype=np.float32)
    mean = state["mean"]
    scale = state["scale"]
    for start in range(0, X_mm.shape[0], chunk_rows):
        end = min(start + chunk_rows, X_mm.shape[0])
        xb = np.asarray(X_mm[start:end], dtype=np.float32)
        xb -= mean
        xb /= (scale + 1e-12)
        out[start:end] = xb
    out.flush()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate Phase 2a LogReg runtime from a local micro-benchmark.")
    ap.add_argument("--top-k", type=int, default=1500)
    ap.add_argument("--n-rows", type=int, default=6000, help="Rows for local benchmark (extrapolated linearly)")
    ap.add_argument("--n-features", type=int, default=23596)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--bench-labels", type=int, default=20, help="Number of labels to actually fit/predict")
    ap.add_argument("--chunk-rows", type=int, default=2048)
    ap.add_argument("--max-iter", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=Path("artefacts_local/tmp_phase2a_runtime_bench"))
    ap.add_argument(
        "--full-n-rows",
        type=int,
        default=82404,
        help="Target full row count for extrapolation (matches your Colab log)",
    )
    ap.add_argument(
        "--effective-n-jobs",
        type=int,
        default=None,
        help="Assumed effective parallelism across labels (default: min(4, cpu_count))",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bench] {cpu_info()}")

    # 1) Use GO OBO + train_terms to satisfy 'use all Terms and GO'
    train_terms_path = Path("Train/train_terms.tsv")
    if not train_terms_path.exists():
        raise FileNotFoundError(train_terms_path)

    print("[bench] Loading train_terms.tsv for TOP_K term list...")
    # Minimal TSV reader (avoid pandas dependency overhead here)
    term_counts: dict[str, int] = {}
    with train_terms_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            term = parts[1]
            term_counts[term] = term_counts.get(term, 0) + 1

    top_terms = sorted(term_counts.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]
    top_term_ids = [t for t, _ in top_terms]
    allowed = set(top_term_ids)
    print(f"[bench] TOP_K={len(top_term_ids)} (most frequent GO terms)")

    obo_path = Path("Train/go-basic.obo")
    if not obo_path.exists():
        raise FileNotFoundError(obo_path)

    t0 = time.perf_counter()
    n_terms_found, n_edges = parse_go_obo_edges(obo_path, allowed)
    t_go = time.perf_counter() - t0
    print(f"[bench] GO parse: terms_found={n_terms_found} edges_in_subset={n_edges} time={t_go:.2f}s")

    # 2) Build synthetic X/Y at realistic dimensionality
    n = int(args.n_rows)
    d = int(args.n_features)
    k = int(args.top_k)
    bench_labels = min(int(args.bench_labels), k)

    x_path = out_dir / "X.npy"
    y_path = out_dir / "Y.npy"

    if not x_path.exists():
        print(f"[bench] Creating synthetic X memmap: shape=({n},{d}) ~ {n*d*4/1e9:.2f} GB")
        X = open_memmap(x_path, shape=(n, d), mode="w+", dtype=np.float32)
        t0 = time.perf_counter()
        fill_random_normal(X, chunk_rows=args.chunk_rows)
        t_x = time.perf_counter() - t0
        print(f"[bench] X created in {t_x:.2f}s (RSS={rss_gb():.2f}GB)")
    else:
        X = np.load(x_path, mmap_mode="r")

    if not y_path.exists():
        print(f"[bench] Creating synthetic Y: shape=({n},{k}) (uint8)")
        rng = np.random.default_rng(123)
        # ~2% positives per label
        Y = (rng.random((n, k)) < 0.02).astype(np.uint8)
        np.save(y_path, Y)
    Y = np.load(y_path, mmap_mode="r")

    # 3) One fold benchmark for scaling + fit + predict
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    idx_tr, idx_val = next(iter(kf.split(np.arange(n))))

    X_tr_path = out_dir / "X_tr.npy"
    X_val_path = out_dir / "X_val.npy"

    # Materialise row subsets as memmaps (mirrors notebook behaviour)
    def materialise_rows(src: np.ndarray, idx: np.ndarray, out_path: Path) -> np.ndarray:
        out = open_memmap(out_path, shape=(len(idx), src.shape[1]), mode="w+", dtype=np.float32)
        for start in range(0, len(idx), args.chunk_rows):
            end = min(start + args.chunk_rows, len(idx))
            rows = idx[start:end]
            out[start:end] = np.asarray(src[rows], dtype=np.float32)
        out.flush()
        return out

    t0 = time.perf_counter()
    X_tr = materialise_rows(X, idx_tr, X_tr_path)
    X_val = materialise_rows(X, idx_val, X_val_path)
    t_mat = time.perf_counter() - t0
    print(f"[bench] Materialise train/val: {t_mat:.2f}s")

    t0 = time.perf_counter()
    state = fit_scaler_state(X_tr, chunk_rows=args.chunk_rows)
    t_fit_scaler = time.perf_counter() - t0

    X_trs_path = out_dir / "X_trs.npy"
    X_vals_path = out_dir / "X_vals.npy"

    t0 = time.perf_counter()
    X_trs = transform_to_memmap(X_tr, state, X_trs_path, chunk_rows=args.chunk_rows)
    X_vals = transform_to_memmap(X_val, state, X_vals_path, chunk_rows=args.chunk_rows)
    t_scale = time.perf_counter() - t0

    print(
        f"[bench] Scaler fit: {t_fit_scaler:.2f}s | scale write: {t_scale:.2f}s | RSS={rss_gb():.2f}GB"
    )

    # Bench fit/predict for a subset of labels
    y_tr = np.asarray(Y[idx_tr, :bench_labels], dtype=np.int64)
    y_val = np.asarray(Y[idx_val, :bench_labels], dtype=np.int64)

    # Fit (sequential) for timing; notebook parallelises over labels (OVR)
    models: list[SGDClassifier] = []
    t0 = time.perf_counter()
    for j in range(bench_labels):
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=args.max_iter,
            tol=None,
            n_jobs=1,
        )
        clf.fit(X_trs, y_tr[:, j])
        models.append(clf)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = np.stack([m.predict_proba(X_vals)[:, 1] for m in models], axis=1)
    t_pred = time.perf_counter() - t0

    per_label_fit = t_fit / bench_labels
    per_label_pred = t_pred / bench_labels

    print(
        f"[bench] Fit {bench_labels} labels: {t_fit:.2f}s ({per_label_fit:.3f}s/label) | "
        f"Predict: {t_pred:.2f}s ({per_label_pred:.3f}s/label)"
    )

    # Extrapolate to full
    effective_n_jobs = args.effective_n_jobs
    if effective_n_jobs is None:
        effective_n_jobs = max(1, min(4, os.cpu_count() or 1))

    row_factor = float(args.full_n_rows) / float(n)
    # scaling is roughly linear in rows; fit/pred linear in rows and labels
    est_scale_fold = (t_mat + t_fit_scaler + t_scale) * row_factor
    est_fit_fold = (per_label_fit * args.top_k / effective_n_jobs) * row_factor
    est_pred_fold = (per_label_pred * args.top_k / effective_n_jobs) * row_factor

    est_fold = est_scale_fold + est_fit_fold + est_pred_fold
    est_total = est_fold * args.n_folds

    print("\n[estimate] Assumptions")
    print(f"- TOP_K={args.top_k}, folds={args.n_folds}")
    print(f"- full_n_rows={args.full_n_rows} (from your Colab log)")
    print(f"- effective_n_jobs={effective_n_jobs} (label-parallelism assumption)")
    print(f"- max_iter={args.max_iter} (SGD epochs)")

    def fmt_h(x: float) -> str:
        if x < 60:
            return f"{x:.1f}s"
        if x < 3600:
            return f"{x/60:.1f}m"
        return f"{x/3600:.2f}h"

    print("\n[estimate] Per-fold breakdown")
    print(f"- scaling/materialise: {fmt_h(est_scale_fold)}")
    print(f"- fit (all labels):     {fmt_h(est_fit_fold)}")
    print(f"- predict (all labels): {fmt_h(est_pred_fold)}")
    print(f"- total per fold:       {fmt_h(est_fold)}")

    print("\n[estimate] Full run")
    print(f"- total for {args.n_folds} folds: {fmt_h(est_total)}")

    print("\n[so-what] Likely improvements")
    print("- Reduce folds 5→2 for first submission")
    print("- Reduce TOP_K (e.g. 1500→512) for iteration speed")
    print("- Keep max_iter=1 for pipeline validation; raise later if needed")
    print("- Consider sparse text/taxa or dimensionality reduction to shrink d")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
