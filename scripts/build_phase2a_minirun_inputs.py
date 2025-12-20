from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def iter_row_chunks(n_rows: int, chunk_rows: int):
    for start in range(0, n_rows, chunk_rows):
        end = min(n_rows, start + chunk_rows)
        yield start, end


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build small, disk-backed Phase 2a inputs (X_train_mmap.npy + Y_train.npy) "
            "from artefacts_local/debug_download for local sanity-checking."
        )
    )
    ap.add_argument(
        "--debug-root",
        type=Path,
        default=Path("artefacts_local/debug_download"),
        help="Root containing parsed/ and features/",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <debug-root>/tmp_phase2a_minirun)",
    )
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--top-terms", type=int, default=128)
    ap.add_argument("--chunk-rows", type=int, default=512)
    ap.add_argument(
        "--embed-keys",
        type=str,
        default="esm2,t5,text",
        help="Comma-separated subset of embeddings to include: esm2,t5,text",
    )
    ap.add_argument(
        "--max-dims-per-embed",
        type=int,
        default=512,
        help="Limit columns taken from each embedding (keeps local run fast)",
    )
    args = ap.parse_args()

    debug_root: Path = args.debug_root
    out_dir = args.out_dir or (debug_root / "tmp_phase2a_minirun")
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_dir = debug_root / "parsed"
    feat_dir = debug_root / "features"

    train_seq_path = parsed_dir / "train_seq.feather"
    train_terms_path = parsed_dir / "train_terms.parquet"
    if not train_seq_path.exists():
        raise FileNotFoundError(train_seq_path)
    if not train_terms_path.exists():
        raise FileNotFoundError(train_terms_path)

    key_to_path = {
        "esm2": feat_dir / "train_embeds_esm2.npy",
        "t5": feat_dir / "train_embeds_t5.npy",
        "text": feat_dir / "train_embeds_text.npy",
    }
    embed_keys = [k.strip() for k in args.embed_keys.split(",") if k.strip()]
    invalid = sorted(set(embed_keys) - set(key_to_path))
    if invalid:
        raise ValueError(f"Unknown --embed-keys: {invalid}. Allowed: {sorted(key_to_path)}")

    emb_paths = [key_to_path[k] for k in embed_keys]
    for p in emb_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"Loading train_seq: {train_seq_path}")
    train_seq = pd.read_feather(train_seq_path)
    train_ids = train_seq["id"].astype(str)
    train_ids_clean = train_ids.str.extract(r"\|(.*?)\|")[0].fillna(train_ids)

    n = min(args.max_rows, len(train_ids_clean))
    ids_subset = train_ids_clean.iloc[:n].tolist()

    print(f"Loading train_terms: {train_terms_path}")
    train_terms = pd.read_parquet(train_terms_path)

    top_terms = train_terms["term"].value_counts().head(args.top_terms).index.tolist()
    term_to_col = {t: i for i, t in enumerate(top_terms)}

    print(f"Building Y: rows={n} top_terms={len(top_terms)}")
    row_of = {entry_id: i for i, entry_id in enumerate(ids_subset)}

    tt = train_terms[train_terms["term"].isin(top_terms)]
    tt = tt[tt["EntryID"].isin(row_of.keys())]

    Y = np.zeros((n, len(top_terms)), dtype=np.uint8)
    for entry_id, term in zip(tt["EntryID"].values, tt["term"].values):
        r = row_of.get(entry_id)
        c = term_to_col.get(term)
        if r is not None and c is not None:
            Y[r, c] = 1

    y_path = out_dir / "Y_train.npy"
    np.save(y_path, Y)
    print(f"Saved Y: {y_path} shape={Y.shape} dtype={Y.dtype}")

    print("Building X_train_mmap.npy (disk-backed)")
    embeds = [np.load(p, mmap_mode="r") for p in emb_paths]
    if any(e.shape[0] < n for e in embeds):
        raise ValueError(f"Not enough rows in embeddings for n={n}: {[e.shape for e in embeds]}")

    max_dims = int(args.max_dims_per_embed)
    dims = [min(int(e.shape[1]), max_dims) for e in embeds]
    d_total = int(sum(dims))
    x_path = out_dir / "X_train_mmap.npy"
    X_mm = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(n, d_total))

    for start, end in iter_row_chunks(n, args.chunk_rows):
        parts = [np.asarray(e[start:end, :d], dtype=np.float32) for e, d in zip(embeds, dims)]
        X_mm[start:end] = np.concatenate(parts, axis=1)

    X_mm.flush()
    del X_mm

    print(f"Saved X: {x_path} shape=({n}, {d_total}) dtype=float32")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
