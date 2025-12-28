"""Verify the 'signal horizon' stable GO-term target count (expected: 1,585).

This script is intended for audit-style verification, using only Phase 1 artefacts:
- cafa6_data/parsed/train_terms.parquet (ground truth annotations)
- Train/go-basic.obo (GO namespaces: BP/MF/CC)
- cafa6_data/IA.tsv (information accretion)

It validates three properties:
1) Aspect mapping integrity from OBO namespaces
2) Hard frequency ceiling: stable iff positives >= noise_floor
3) IA coverage within the stable set (95%/99% points)

Outputs:
- artefacts_local/audits/signal_horizon_1585/summary.json

Run:
  python scripts/verify_signal_horizon_1585.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AspectAudit:
    aspect: str
    n_terms_total_in_counts: int
    n_terms_mapped: int
    n_terms_unmapped: int
    n_terms_noise_eligible: int
    noise_floor: int
    k_noise: int
    k_95_ia_eligible: Optional[int]
    k_99_ia_eligible: Optional[int]
    ia_total_all_terms: float
    ia_total_noise_eligible: float
    ia_at_k_noise: float
    ia_at_k_noise_frac_of_eligible: float
    count_at_k_noise: Optional[float]
    count_at_k_noise_plus_1: Optional[float]


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_go_obo(repo_root: Path, work_root: Path) -> Path:
    candidates = [
        repo_root / "Train" / "go-basic.obo",
        work_root / "Train" / "go-basic.obo",
        work_root / "go-basic.obo",
        repo_root / "go-basic.obo",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"go-basic.obo not found. Tried: {[str(p) for p in candidates]}"
    )


def _read_ia_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["term", "ia"]
    else:
        raise ValueError(f"Unexpected IA.tsv format: {path}")

    # Some dumps include a header row; handle that safely.
    df["ia"] = pd.to_numeric(df["ia"], errors="coerce")
    df = df.dropna(subset=["term", "ia"]).copy()
    df["term"] = df["term"].astype(str)
    return df


def _aspect_namespace_map() -> Dict[str, str]:
    return {
        "biological_process": "BP",
        "molecular_function": "MF",
        "cellular_component": "CC",
    }


def _load_term_namespace(path_go_obo: Path) -> Dict[str, str]:
    import obonet  # local import: makes failure mode obvious

    graph = obonet.read_obo(str(path_go_obo))
    term_to_ns: Dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        ns = data.get("namespace")
        if ns is None:
            continue
        term_to_ns[str(node)] = str(ns)
    return term_to_ns


def _compute_k_for_frac(values: np.ndarray, frac: float) -> Optional[int]:
    if values.size == 0:
        return None
    total = float(values.sum())
    if total <= 0:
        return None
    csum = np.cumsum(values)
    target = frac * total
    k = int(np.searchsorted(csum, target, side="left") + 1)
    return k


def _audit_aspect(
    counts_df: pd.DataFrame,
    ia_df: pd.DataFrame,
    aspect: str,
    noise_floor: int,
) -> AspectAudit:
    asp = counts_df[counts_df["aspect"] == aspect].copy()

    n_terms_total = int(asp.shape[0])
    n_terms_mapped = int((asp["is_mapped"] == True).sum())
    n_terms_unmapped = int((asp["is_mapped"] == False).sum())

    asp = asp.merge(ia_df, on="term", how="left")
    asp["ia"] = asp["ia"].fillna(0.0).astype(float)

    ia_total_all = float(asp["ia"].sum())

    eligible = asp[asp["count"] >= noise_floor].sort_values(
        "count", ascending=False
    )
    n_eligible = int(eligible.shape[0])
    ia_total_eligible = float(eligible["ia"].sum())

    ia_values = eligible["ia"].to_numpy(dtype=np.float64, copy=False)
    k95 = _compute_k_for_frac(ia_values, 0.95)
    k99 = _compute_k_for_frac(ia_values, 0.99)

    k_noise = n_eligible
    ia_at_k_noise = float(ia_values.sum())
    ia_frac_eligible = float(ia_at_k_noise / ia_total_eligible) if ia_total_eligible > 0 else 0.0

    count_at_k_noise: Optional[float] = None
    count_at_k_noise_plus_1: Optional[float] = None
    if n_eligible > 0:
        count_at_k_noise = float(eligible.iloc[n_eligible - 1]["count"])
        if asp.shape[0] > n_eligible:
            # next term just below the cut
            next_row = asp.sort_values("count", ascending=False).iloc[n_eligible]
            count_at_k_noise_plus_1 = float(next_row["count"])

    return AspectAudit(
        aspect=aspect,
        n_terms_total_in_counts=n_terms_total,
        n_terms_mapped=n_terms_mapped,
        n_terms_unmapped=n_terms_unmapped,
        n_terms_noise_eligible=n_eligible,
        noise_floor=noise_floor,
        k_noise=k_noise,
        k_95_ia_eligible=k95,
        k_99_ia_eligible=k99,
        ia_total_all_terms=ia_total_all,
        ia_total_noise_eligible=ia_total_eligible,
        ia_at_k_noise=ia_at_k_noise,
        ia_at_k_noise_frac_of_eligible=ia_frac_eligible,
        count_at_k_noise=count_at_k_noise,
        count_at_k_noise_plus_1=count_at_k_noise_plus_1,
    )


def main() -> None:
    repo_root = _find_repo_root()
    work_root = repo_root / "cafa6_data"

    train_terms_path = work_root / "parsed" / "train_terms.parquet"
    if not train_terms_path.exists():
        raise FileNotFoundError(f"Missing: {train_terms_path}")

    ia_path = work_root / "IA.tsv"
    if not ia_path.exists():
        # fallback for local layouts
        ia_path = repo_root / "IA.tsv"
    if not ia_path.exists():
        raise FileNotFoundError(f"Missing IA.tsv at {work_root / 'IA.tsv'} or {repo_root / 'IA.tsv'}")

    go_obo_path = _find_go_obo(repo_root=repo_root, work_root=work_root)

    noise_floor = 50

    train_terms = pd.read_parquet(train_terms_path)
    if "term" not in train_terms.columns:
        raise ValueError(f"Expected column 'term' in {train_terms_path}; got {list(train_terms.columns)}")

    term_counts = (
        train_terms["term"].astype(str).value_counts().rename_axis("term").reset_index(name="count")
    )

    term_to_ns = _load_term_namespace(go_obo_path)
    ns_map = _aspect_namespace_map()

    def map_aspect(term: str) -> Tuple[str, bool]:
        ns = term_to_ns.get(term)
        if ns is None:
            return "UNK", False
        asp = ns_map.get(ns)
        if asp is None:
            return "UNK", False
        return asp, True

    mapped = term_counts["term"].map(map_aspect)
    term_counts["aspect"] = mapped.map(lambda x: x[0])
    term_counts["is_mapped"] = mapped.map(lambda x: x[1])

    ia_df = _read_ia_tsv(ia_path)

    # Integrity audit (global)
    n_total_terms = int(term_counts.shape[0])
    n_mapped = int((term_counts["is_mapped"] == True).sum())
    n_unmapped = int((term_counts["is_mapped"] == False).sum())

    unmapped_top = (
        term_counts[term_counts["is_mapped"] == False]
        .sort_values("count", ascending=False)
        .head(25)
        .to_dict(orient="records")
    )

    # Aspect audits
    aspects = ["BP", "MF", "CC"]
    aspect_results: Dict[str, AspectAudit] = {}
    for asp in aspects:
        aspect_results[asp] = _audit_aspect(
            counts_df=term_counts,
            ia_df=ia_df,
            aspect=asp,
            noise_floor=noise_floor,
        )

    total_stable = int(sum(a.n_terms_noise_eligible for a in aspect_results.values()))

    out_dir = repo_root / "artefacts_local" / "audits" / "signal_horizon_1585"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "work_root": str(work_root),
        "train_terms_path": str(train_terms_path),
        "ia_path": str(ia_path),
        "go_obo_path": str(go_obo_path),
        "noise_floor": noise_floor,
        "mapping_integrity": {
            "n_total_terms_in_train_terms": n_total_terms,
            "n_mapped_terms": n_mapped,
            "n_unmapped_terms": n_unmapped,
            "unmapped_top25": unmapped_top,
        },
        "stable_targets": {
            "per_aspect": {k: asdict(v) for k, v in aspect_results.items()},
            "total_stable_targets_bp_mf_cc": total_stable,
        },
        "expected": {
            "expected_total_stable_targets": 1585,
            "matches_expected": (total_stable == 1585),
        },
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Console summary (what an auditor cares about)
    print("\n=== Signal Horizon Verification (Phase 1 artefacts) ===")
    print(f"Noise floor (positives): {noise_floor}")
    print(f"Aspect mapping: mapped={n_mapped}/{n_total_terms} | unmapped={n_unmapped}")

    for asp in aspects:
        a = aspect_results[asp]
        print(f"\n--- {asp} ---")
        print(f"Stable terms (count >= {noise_floor}): {a.n_terms_noise_eligible}")
        if a.count_at_k_noise is not None:
            print(f"Cut boundary: count@k={a.count_at_k_noise}, next={a.count_at_k_noise_plus_1}")
        if a.k_95_ia_eligible is not None and a.k_99_ia_eligible is not None:
            print(
                f"IA coverage within eligible set: k95={a.k_95_ia_eligible}, k99={a.k_99_ia_eligible} "
                f"(eligible IA total={a.ia_total_noise_eligible:.2f})"
            )
        else:
            print("IA coverage: insufficient IA data (eligible IA sum is 0)")

    print("\n--- Final ---")
    print(f"Total stable targets (BP+MF+CC): {total_stable}")
    print(f"Matches expected 1585: {total_stable == 1585}")
    print(f"Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
