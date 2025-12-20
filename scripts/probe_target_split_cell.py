from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_work_root(repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (repo_root / p)

    candidates = [
        repo_root / "artefacts_local" / "debug_download",
        repo_root / "artefacts_local" / "work",
        repo_root / "debug_download",
    ]
    for c in candidates:
        if (c / "parsed" / "train_terms.parquet").exists():
            return c
    raise FileNotFoundError(
        "Could not locate WORK_ROOT containing parsed/train_terms.parquet. "
        "Pass --work-root explicitly."
    )


def resolve_obo(repo_root: Path, work_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (repo_root / p)

    candidates = [
        work_root / "go-basic.obo",
        repo_root / "Train" / "go-basic.obo",
        repo_root / "go-basic.obo",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find go-basic.obo. Pass --obo-path explicitly.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Duplicates the notebook cell that selects Top-K terms per aspect and logs aspect category counts."
        )
    )
    ap.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--work-root", type=str, default=None, help="e.g. artefacts_local/debug_download")
    ap.add_argument("--obo-path", type=str, default=None, help="e.g. Train/go-basic.obo")
    ap.add_argument("--bp", type=int, default=10000)
    ap.add_argument("--mf", type=int, default=2000)
    ap.add_argument("--cc", type=int, default=1500)
    ap.add_argument("--global-k", type=int, default=13500)
    ap.add_argument("--allow-global-fallback", action="store_true")
    ap.add_argument("--show-sample-terms", type=int, default=10)
    ap.add_argument(
        "--materialise-sparse",
        action="store_true",
        help="If scipy is available, build a CSR matrix for extra sanity checks (not required for stats).",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    work_root = resolve_work_root(repo_root, args.work_root)
    obo_path = resolve_obo(repo_root, work_root, args.obo_path)

    train_terms_path = work_root / "parsed" / "train_terms.parquet"

    print("repo_root:", repo_root)
    print("work_root:", work_root)
    print("train_terms_path:", train_terms_path)
    print("obo_path:", obo_path)
    print()

    train_terms = pd.read_parquet(train_terms_path)
    if not {"EntryID", "term"}.issubset(train_terms.columns):
        raise RuntimeError(f"Unexpected train_terms columns: {list(train_terms.columns)}")

    train_seq_path = work_root / "parsed" / "train_seq.feather"
    if not train_seq_path.exists():
        raise FileNotFoundError(f"Missing {train_seq_path} (needed to mirror notebook Y reindex)")
    train_ids = pd.read_feather(train_seq_path)["id"].astype(str)
    # Mirror notebook ID cleaning
    train_ids_clean = train_ids.str.extract(r"\|(.*?)\|")[0]
    train_ids_clean = train_ids_clean.fillna(train_ids)

    print("train_terms shape:", train_terms.shape)
    print("unique proteins:", train_terms["EntryID"].nunique())
    print("unique terms:", train_terms["term"].nunique())
    print("train_seq ids:", len(train_ids))
    print("train_seq clean ids:", len(train_ids_clean))
    clean_changed = int((train_ids_clean != train_ids).sum())
    print(f"train_seq ids changed by cleaning: {clean_changed} ({clean_changed/len(train_ids):.2%})")

    # Alignment check: are the labels keyed by the cleaned ids?
    entry_set = set(train_terms["EntryID"].astype(str).unique())
    in_raw = float(train_ids.isin(entry_set).mean())
    in_clean = float(train_ids_clean.isin(entry_set).mean())
    print(f"ID alignment: train_ids in EntryID set = {in_raw:.2%}")
    print(f"ID alignment: train_ids_clean in EntryID set = {in_clean:.2%}")

    if "aspect" in train_terms.columns:
        print("\n[RAW] aspect value_counts:")
        print(train_terms["aspect"].value_counts(dropna=False).to_string())
    else:
        print("\n[RAW] aspect column missing")

    # ---- Mirror notebook cell logic ----
    import obonet

    print("\nLoading OBO...")
    graph = obonet.read_obo(str(obo_path))
    term_to_ns = {node: data.get("namespace", "unknown") for node, data in graph.nodes(data=True)}
    ns_map = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}

    # Normalise aspect encoding if present, else create from OBO
    aspect_aliases = {
        "biological_process": "BP",
        "molecular_function": "MF",
        "cellular_component": "CC",
        "BP": "BP",
        "MF": "MF",
        "CC": "CC",
    }

    if "aspect" in train_terms.columns:
        train_terms["aspect_norm"] = train_terms["aspect"].map(lambda a: aspect_aliases.get(str(a), "UNK"))
    else:
        train_terms["aspect_norm"] = train_terms["term"].map(lambda t: ns_map.get(term_to_ns.get(t), "UNK"))

    print("\n[NORMALISED] aspect_norm value_counts:")
    print(train_terms["aspect_norm"].value_counts(dropna=False).to_string())

    # Group and pick top terms per aspect
    term_counts = (
        train_terms.groupby(["aspect_norm", "term"]).size().reset_index(name="count")
    )

    targets_bp = (
        term_counts[term_counts["aspect_norm"] == "BP"].nlargest(args.bp, "count")["term"].tolist()
    )
    targets_mf = (
        term_counts[term_counts["aspect_norm"] == "MF"].nlargest(args.mf, "count")["term"].tolist()
    )
    targets_cc = (
        term_counts[term_counts["aspect_norm"] == "CC"].nlargest(args.cc, "count")["term"].tolist()
    )

    print("\nSelected per-aspect term counts:")
    print({"BP": len(targets_bp), "MF": len(targets_mf), "CC": len(targets_cc)})

    if len(targets_bp) == 0 and len(targets_mf) == 0 and len(targets_cc) == 0:
        msg = (
            "No BP/MF/CC split found after normalisation; would fall back to global Top-"
            f"{args.global_k}."
        )
        if args.allow_global_fallback:
            print("[WARNING]", msg)
            top_terms = train_terms["term"].value_counts().head(args.global_k).index.tolist()
        else:
            raise RuntimeError(msg + " Use --allow-global-fallback to reproduce the notebook fallback.")
    else:
        # NOTE: The notebook used list(set(...)) which is order-unstable across runs.
        # Here we preserve a stable order (BP then MF then CC) while keeping uniqueness.
        top_terms = list(dict.fromkeys(targets_bp + targets_mf + targets_cc))

    # Extra proof: show term samples and namespace mapping sanity
    print("\nFinal top_terms:")
    print("n_top_terms:", len(top_terms))
    print("sample_terms:", top_terms[: max(0, int(args.show_sample_terms))])

    # Sanity: how many selected terms have known namespace
    ns_selected = pd.Series([term_to_ns.get(t, None) for t in top_terms], name="obo_namespace")
    print("\nSelected term namespaces (OBO):")
    print(ns_selected.value_counts(dropna=False).to_string())

    # Show overlaps (not too deep, just for debugging)
    print("\nOverlaps between per-aspect lists:")
    set_bp, set_mf, set_cc = set(targets_bp), set(targets_mf), set(targets_cc)
    print(
        {
            "BP∩MF": len(set_bp & set_mf),
            "BP∩CC": len(set_bp & set_cc),
            "MF∩CC": len(set_mf & set_cc),
        }
    )

    # ---- Mirror next notebook lines: train_terms_top -> pivot -> reindex -> Y ----
    print("\n--- Building Y (mirrors notebook pivot_table + reindex) ---")
    n_samples = len(train_ids_clean)
    n_labels = len(top_terms)
    est_dense_gb = (n_samples * n_labels * 4) / (1024**3)  # float32
    print(f"Expected dense Y size (float32): ~{est_dense_gb:.2f} GB (not including pandas overhead)")

    # Filter to selected terms
    train_terms_top = train_terms[train_terms["term"].isin(top_terms)].copy()
    print("train_terms_top rows:", len(train_terms_top))

    row_map = {k: i for i, k in enumerate(train_ids_clean.tolist())}
    col_map = {t: j for j, t in enumerate(top_terms)}

    # Map EntryID/term to integer indices and report drops
    r = train_terms_top["EntryID"].astype(str).map(row_map)
    c = train_terms_top["term"].astype(str).map(col_map)
    valid = r.notna() & c.notna()
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"[WARNING] Dropping {n_dropped} annotations due to unmapped EntryID/term.")
    r = r[valid].astype(np.int64).to_numpy()
    c = c[valid].astype(np.int64).to_numpy()

    # De-duplicate repeated (row, col) pairs defensively
    if len(r) == 0:
        raise RuntimeError("No valid (EntryID, term) pairs after filtering; cannot construct Y stats.")
    pairs = np.stack([r, c], axis=1)
    pairs = np.unique(pairs, axis=0)
    r_u = pairs[:, 0]
    c_u = pairs[:, 1]
    nnz = int(pairs.shape[0])

    print(f"Y.shape = ({n_samples}, {n_labels})")
    density = nnz / float(n_samples * n_labels)
    print(f"Y nnz (unique protein-term pairs): {nnz:,}")
    print(f"Y density: {density:.6%}")
    print(f"avg labels/protein: {nnz / n_samples:.3f}")
    print(f"avg proteins/label: {nnz / n_labels:.3f}")

    row_counts = np.bincount(r_u, minlength=n_samples)
    col_counts = np.bincount(c_u, minlength=n_labels)
    print("row nnz quantiles (labels per protein):", {
        "p50": int(np.quantile(row_counts, 0.50)),
        "p90": int(np.quantile(row_counts, 0.90)),
        "p99": int(np.quantile(row_counts, 0.99)),
        "max": int(row_counts.max()),
    })
    print("col nnz quantiles (proteins per label):", {
        "p50": int(np.quantile(col_counts, 0.50)),
        "p90": int(np.quantile(col_counts, 0.90)),
        "p99": int(np.quantile(col_counts, 0.99)),
        "max": int(col_counts.max()),
    })

    # Per-aspect sparsity on the selected label set
    idx_bp = np.array([col_map[t] for t in targets_bp if t in col_map], dtype=np.int64)
    idx_mf = np.array([col_map[t] for t in targets_mf if t in col_map], dtype=np.int64)
    idx_cc = np.array([col_map[t] for t in targets_cc if t in col_map], dtype=np.int64)
    for name, idx in [("BP", idx_bp), ("MF", idx_mf), ("CC", idx_cc)]:
        if len(idx) == 0:
            print(f"[{name}] No labels selected")
            continue
        nnz_a = int(col_counts[idx].sum())
        den_a = nnz_a / float(n_samples * len(idx))
        print(f"[{name}] labels={len(idx):,} nnz={nnz_a:,} density={den_a:.6%} avg_labels/protein={nnz_a/n_samples:.3f}")

    # Optional: materialise a CSR for additional sanity checks
    if args.materialise_sparse:
        try:
            from scipy import sparse  # type: ignore

            data = np.ones_like(r_u, dtype=np.uint8)
            Y_csr = sparse.csr_matrix((data, (r_u, c_u)), shape=(n_samples, n_labels), dtype=np.uint8)
            print("\n[CSR] Built sparse Y:")
            print("  shape:", Y_csr.shape)
            print("  nnz:", Y_csr.nnz)
        except Exception as e:
            print(f"[WARNING] Could not build CSR (scipy missing or error): {e}")


if __name__ == "__main__":
    main()
