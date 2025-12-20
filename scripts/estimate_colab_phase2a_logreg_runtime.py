from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass
class Timing:
    name: str
    seconds: float


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def clean_uniprot_id(raw: str) -> str:
    # Matches notebook behaviour: train_ids.str.extract(r'\|(.*?)\|')[0].fillna(train_ids)
    m = re.search(r"\|(.*?)\|", raw)
    return m.group(1) if m else raw


def iter_chunks_from_indices(idxs: np.ndarray, chunk_rows: int):
    for start in range(0, len(idxs), chunk_rows):
        end = min(len(idxs), start + chunk_rows)
        yield idxs[start:end]


def load_feature_mm(feature_dir: Path):
    # What we have locally in debug_download.
    # Colab can have more (esm2_3b, taxa); we’ll extrapolate.
    paths = {
        "t5": feature_dir / "train_embeds_t5.npy",
        "esm2_650m": feature_dir / "train_embeds_esm2.npy",
        "text": feature_dir / "train_embeds_text.npy",
    }
    mm = {}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k} at {p}")
        mm[k] = np.load(p, mmap_mode="r")
    return mm


def assemble_chunk(feature_mm: dict[str, np.ndarray], rows: np.ndarray, dims: dict[str, int]) -> np.ndarray:
    d_total = int(sum(dims.values()))
    X = np.empty((len(rows), d_total), dtype=np.float32)
    col = 0
    for k, arr in feature_mm.items():
        d = int(dims[k])
        X[:, col : col + d] = np.asarray(arr[rows, :d], dtype=np.float32)
        col += d
    return X


def build_y_bench(
    train_ids: list[str],
    train_terms: pd.DataFrame,
    top_terms: list[str],
    n_labels_bench: int,
) -> tuple[np.ndarray, list[str]]:
    terms_bench = top_terms[:n_labels_bench]
    term_to_col = {t: i for i, t in enumerate(terms_bench)}
    row_of = {entry_id: i for i, entry_id in enumerate(train_ids)}

    Y = np.zeros((len(train_ids), len(terms_bench)), dtype=np.uint8)
    tt = train_terms[train_terms["term"].isin(terms_bench)]
    # reduce to only EntryIDs we have
    tt = tt[tt["EntryID"].isin(row_of.keys())]

    for entry_id, term in zip(tt["EntryID"].values, tt["term"].values):
        r = row_of.get(entry_id)
        c = term_to_col.get(term)
        if r is not None and c is not None:
            Y[r, c] = 1

    return Y, terms_bench


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Estimate Phase 2a LogReg runtime using real artefacts_local/debug_download arrays. "
            "Times chunked scaling + chunked SGD on a small label subset, then extrapolates to TOP_K terms and folds."
        )
    )
    ap.add_argument("--debug-root", type=Path, default=Path("artefacts_local/debug_download"))
    ap.add_argument("--top-k", type=int, default=1500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--chunk-rows", type=int, default=2048)
    ap.add_argument(
        "--bench-labels",
        type=int,
        default=32,
        help="Number of labels to train locally for timing (extrapolated to --top-k)",
    )
    ap.add_argument(
        "--bench-chunks",
        type=int,
        default=8,
        help="Number of chunks to process for timing (extrapolated to full fold)",
    )
    ap.add_argument(
        "--dims",
        type=str,
        default=None,
        help=(
            "Optional per-feature dim caps, e.g. t5=1024,esm2_650m=1280,text=2048. "
            "Default: use full available dims for each local feature."
        ),
    )
    ap.add_argument(
        "--extrapolate-total-dims",
        type=int,
        default=23596,
        help="Target total feature dim to extrapolate to (Colab full stack).",
    )
    args = ap.parse_args()

    debug_root: Path = args.debug_root
    parsed_dir = debug_root / "parsed"
    feat_dir = debug_root / "features"

    train_seq_path = parsed_dir / "train_seq.feather"
    train_terms_path = parsed_dir / "train_terms.parquet"
    go_obo_path = Path("Train/go-basic.obo")

    if not train_seq_path.exists():
        raise FileNotFoundError(train_seq_path)
    if not train_terms_path.exists():
        raise FileNotFoundError(train_terms_path)

    feature_mm = load_feature_mm(feat_dir)

    n_rows = next(iter(feature_mm.values())).shape[0]
    for k, arr in feature_mm.items():
        if arr.shape[0] != n_rows:
            raise ValueError(f"Row mismatch: {k} has {arr.shape[0]} rows, expected {n_rows}")

    # Feature dims (local)
    dims: dict[str, int] = {k: int(arr.shape[1]) for k, arr in feature_mm.items()}
    if args.dims:
        for part in args.dims.split(","):
            if not part.strip():
                continue
            k, v = part.split("=")
            k = k.strip()
            v = int(v.strip())
            if k not in dims:
                raise ValueError(f"Unknown feature key in --dims: {k}. Known: {sorted(dims)}")
            dims[k] = min(dims[k], v)

    d_local_total = int(sum(dims.values()))

    print("=== Local Phase 2a LogReg runtime estimate (real debug_download) ===")
    print(f"Rows: {n_rows}")
    print("Feature shapes (local):")
    for k, arr in feature_mm.items():
        print(f"  {k}: {arr.shape} (using dims={dims[k]})")
    print(f"Total local dims used: {d_local_total}")
    print(f"RSS start: {rss_gb():.2f} GB")

    # Load IDs / terms
    t0 = time.perf_counter()
    train_seq = pd.read_feather(train_seq_path)
    train_ids_raw = train_seq["id"].astype(str).tolist()
    train_ids = [clean_uniprot_id(x) for x in train_ids_raw]

    train_terms = pd.read_parquet(train_terms_path)
    term_counts = train_terms["term"].value_counts()
    top_terms = term_counts.head(args.top_k).index.tolist()
    t1 = time.perf_counter()
    print(f"Loaded train_seq + train_terms, built top_terms: {t1 - t0:.2f}s")
    print(f"Unique terms in train_terms: {term_counts.shape[0]}")
    print(f"Top-K terms requested: {args.top_k}")

    # GO parse timing (include GO overhead)
    if go_obo_path.exists():
        try:
            import obonet

            g0 = time.perf_counter()
            graph = obonet.read_obo(go_obo_path)
            g1 = time.perf_counter()
            print(f"Parsed GO OBO ({go_obo_path}) in {g1 - g0:.2f}s; nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
        except Exception as e:
            print(f"[WARN] Failed to parse GO OBO ({go_obo_path}): {e}")
    else:
        print(f"[WARN] GO OBO not found at {go_obo_path} (skipping GO parse timing)")

    # Build a benchmark Y for a small number of labels (still uses real terms)
    bench_labels = min(int(args.bench_labels), len(top_terms))
    t0 = time.perf_counter()
    Y_bench, terms_bench = build_y_bench(train_ids, train_terms, top_terms, bench_labels)
    t1 = time.perf_counter()
    print(f"Built Y_bench: shape={Y_bench.shape} in {t1 - t0:.2f}s (labels={bench_labels})")

    # One fold indices
    kf = KFold(n_splits=int(args.folds), shuffle=True, random_state=42)
    idx_tr, idx_val = next(iter(kf.split(np.arange(n_rows))))
    idx_tr = np.asarray(idx_tr)

    # Prepare chunk schedule (only first N chunks for timing)
    chunk_rows = int(args.chunk_rows)
    bench_chunks = int(args.bench_chunks)

    # 1) Scaling pass timing (partial_fit) over N chunks
    scaler = StandardScaler(with_mean=True, with_std=True)
    times: list[Timing] = []

    t0 = time.perf_counter()
    for i, rows in enumerate(iter_chunks_from_indices(idx_tr, chunk_rows)):
        if i >= bench_chunks:
            break
        Xc = assemble_chunk(feature_mm, rows, dims)
        scaler.partial_fit(Xc)
    t1 = time.perf_counter()
    scale_seconds = t1 - t0
    times.append(Timing("scaler_partial_fit_bench", scale_seconds))

    # Extrapolate scaling time to full fold
    n_train = int(len(idx_tr))
    n_train_chunks_total = int((n_train + chunk_rows - 1) // chunk_rows)
    scaler_seconds_est = scale_seconds * (n_train_chunks_total / max(1, bench_chunks))

    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    # 2) Training pass timing over N chunks for bench_labels models
    models = [
        SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1, tol=None)
        for _ in range(bench_labels)
    ]

    t0 = time.perf_counter()
    for i, rows in enumerate(iter_chunks_from_indices(idx_tr, chunk_rows)):
        if i >= bench_chunks:
            break
        Xc = assemble_chunk(feature_mm, rows, dims)
        Xc -= mean
        Xc /= (scale + 1e-12)
        Yc = Y_bench[rows]

        # partial_fit per label
        for j, m in enumerate(models):
            if i == 0:
                m.partial_fit(Xc, Yc[:, j].astype(np.int64), classes=np.array([0, 1], dtype=np.int64))
            else:
                m.partial_fit(Xc, Yc[:, j].astype(np.int64))
    t1 = time.perf_counter()
    train_seconds = t1 - t0
    times.append(Timing("train_partial_fit_bench", train_seconds))

    train_seconds_per_chunk = train_seconds / max(1, bench_chunks)
    train_seconds_per_chunk_per_label = train_seconds_per_chunk / max(1, bench_labels)

    # Extrapolate: full train pass for all labels
    train_seconds_est_localdims = train_seconds_per_chunk_per_label * n_train_chunks_total * int(args.top_k)

    # 3) Dimensional extrapolation to full Colab dim
    d_target = int(args.extrapolate_total_dims)
    dim_factor = float(d_target) / float(d_local_total) if d_local_total > 0 else 1.0

    scaler_seconds_est_full = scaler_seconds_est * dim_factor
    train_seconds_est_full = train_seconds_est_localdims * dim_factor

    # Per-fold estimate (fit only; predict adds more, but smaller than fit)
    fold_seconds_est_full = scaler_seconds_est_full + train_seconds_est_full

    # Full CV estimate
    total_seconds_est_full = fold_seconds_est_full * int(args.folds)

    print("\n--- Bench timings (on subset) ---")
    for t in times:
        print(f"{t.name}: {t.seconds:.2f}s")

    print("\n--- Extrapolated estimates ---")
    print(f"Train rows in fold: {n_train}")
    print(f"Chunk rows: {chunk_rows} => train chunks per fold ~ {n_train_chunks_total}")
    print(f"Bench labels: {bench_labels}; Extrapolate to top_k={args.top_k}")
    print(f"Local dims used: {d_local_total}; Extrapolate to {d_target} => dim_factor={dim_factor:.2f}")

    print(f"Estimated scaler pass per fold: {scaler_seconds_est_full/60:.1f} min")
    print(f"Estimated training pass per fold (TOP_K): {train_seconds_est_full/60:.1f} min")
    print(f"Estimated total per fold (fit only): {fold_seconds_est_full/60:.1f} min")
    print(f"Estimated total CV ({args.folds} folds, fit only): {total_seconds_est_full/3600:.2f} hours")

    print("\n--- Practical speed levers (highest impact first) ---")
    print("1) Reduce TOP_K (linear speed-up): 1500 -> 256 is ~5.9x faster")
    print("2) Reduce folds for LogReg (linear): 5 -> 2 is 2.5x faster")
    print("3) Reduce modalities/dims (roughly linear): drop text/taxa if needed")
    print("4) Avoid CPU thread oversubscription: set OMP_NUM_THREADS=1 and keep n_jobs modest")
    print("5) Consider cuML only if you can keep it on GPU end-to-end; otherwise it adds transfer overhead")

    print(f"\nRSS end: {rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
