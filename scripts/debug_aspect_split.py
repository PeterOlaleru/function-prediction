from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_work_root(repo_root: Path, explicit: str | None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_absolute() else (repo_root / p)

    candidates = [
        repo_root / "artefacts_local" / "debug_download",
        repo_root / "debug_download",
        repo_root / "artefacts_local" / "work",
        repo_root / "artefacts_local" / "artefacts",
    ]
    for p in candidates:
        if (p / "parsed" / "train_terms.parquet").exists():
            return p
    raise FileNotFoundError(
        "Could not find parsed/train_terms.parquet under any default WORK_ROOT candidates. "
        "Pass --work-root explicitly."
    )


def _resolve_obo(repo_root: Path, explicit: str | None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_absolute() else (repo_root / p)

    candidates = [
        repo_root / "Train" / "go-basic.obo",
        repo_root / "go-basic.obo",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find go-basic.obo. Pass --obo-path explicitly.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--work-root", type=str, default=None)
    ap.add_argument("--obo-path", type=str, default=None)
    ap.add_argument("--sample-terms", type=int, default=2000)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    work_root = _resolve_work_root(repo_root, args.work_root)
    obo_path = _resolve_obo(repo_root, args.obo_path)

    train_terms_path = work_root / "parsed" / "train_terms.parquet"
    print("repo_root:", repo_root)
    print("work_root:", work_root)
    print("train_terms_path:", train_terms_path, "exists:", train_terms_path.exists())
    print("obo_path:", obo_path, "exists:", obo_path.exists())

    train_terms = pd.read_parquet(train_terms_path)
    print("train_terms shape:", train_terms.shape)
    print("train_terms columns:", list(train_terms.columns))

    col_id = "EntryID" if "EntryID" in train_terms.columns else train_terms.columns[0]
    col_term = "term" if "term" in train_terms.columns else train_terms.columns[1]
    print("id col:", col_id, "term col:", col_term)

    terms = train_terms[col_term].astype(str)
    print("term startswith GO: (first 2000 rows):", float(np.mean(terms.head(2000).str.startswith("GO:"))))

    if "aspect" in train_terms.columns:
        asp = train_terms["aspect"].astype(str)
        vc = asp.value_counts(dropna=False)
        print("\n[PARQUET] aspect value_counts (top 20):")
        print(vc.head(20).to_string())
        uniq = set(asp.dropna().unique().tolist())
        print("[PARQUET] contains BP/MF/CC:", {k: (k in uniq) for k in ["BP", "MF", "CC"]})
        print("[PARQUET] contains biological_process/molecular_function/cellular_component:",
              {k: (k in uniq) for k in ["biological_process", "molecular_function", "cellular_component"]})
    else:
        print("\n[PARQUET] NO aspect column present")

    # OBO cross-check
    import obonet

    print("\nLoading OBO (namespace mapping)...")
    graph = obonet.read_obo(str(obo_path))
    term_to_ns = {node: data.get("namespace", None) for node, data in graph.nodes(data=True)}

    uterms = pd.unique(terms)
    sample_n = min(int(args.sample_terms), len(uterms))
    sample = uterms[:sample_n]
    ns = pd.Series([term_to_ns.get(t, None) for t in sample], name="obo_namespace")
    print("[OBO] namespace coverage on first", sample_n, "unique terms:")
    print(ns.value_counts(dropna=False).head(20).to_string())

    if "aspect" in train_terms.columns:
        ns_map = {
            "biological_process": "BP",
            "molecular_function": "MF",
            "cellular_component": "CC",
        }
        tmp = train_terms[[col_term, "aspect"]].drop_duplicates().head(30).copy()
        tmp["obo_namespace"] = tmp[col_term].astype(str).map(lambda t: term_to_ns.get(t, None))
        tmp["aspect_from_obo"] = tmp["obo_namespace"].map(lambda s: ns_map.get(s, "UNK") if s is not None else None)
        print("\nSample term/aspect vs OBO:")
        print(tmp.to_string(index=False))

        # This reproduces the notebook's failure mode precisely:
        bp_ct = int((train_terms["aspect"].astype(str) == "BP").sum())
        mf_ct = int((train_terms["aspect"].astype(str) == "MF").sum())
        cc_ct = int((train_terms["aspect"].astype(str) == "CC").sum())
        print("\n[NOTEBOOK FAILURE MODE CHECK] Counts where aspect == BP/MF/CC:")
        print({"BP": bp_ct, "MF": mf_ct, "CC": cc_ct})
        if bp_ct == 0 and mf_ct == 0:
            print("=> This would trigger the notebook fallback to global Top-13,500.")
        else:
            print("=> Notebook should NOT fall back (it would find BP/MF/CC rows).")


if __name__ == "__main__":
    main()
