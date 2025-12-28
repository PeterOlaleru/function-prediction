"""Frequency Threshold Audit + IA-weight contribution analysis for expanding GBDT targets.

Purpose
- Quantify whether expanding the Level-1 GBDT (Py-Boost) target set from 4,500 to 6,000 GO terms is viable.
- Uses Phase-1 artefact `parsed/term_counts.parquet` plus `IA.tsv`.

Outputs
- Console report (key metrics + verdict)
- `summary.json`
- Plots (`.png`) under `artefacts_local/audits/gbdt_target_scale/`

Notes
- This script is intentionally model-agnostic (no training). It focuses on signal density.
- British English output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AuditConfig:
    base_k: int
    expanded_k: int
    noise_floor: int


@dataclass(frozen=True)
class AuditSummary:
    total_terms_available: int
    base_k: int
    expanded_k: int
    tail_k: int

    # Frequency / annotation density
    base_min_count: float
    tail_max_count: float
    tail_min_count: float
    tail_mean_count: float
    tail_median_count: float
    tail_p10_count: float
    tail_p25_count: float

    sum_counts_base: float
    sum_counts_tail: float
    sum_counts_tail_over_base: float

    # Noise floor
    noise_floor: int
    tail_below_noise_n: int
    tail_below_noise_frac: float

    # IA
    missing_ia_terms_n: int
    ia_sum_base: float
    ia_sum_tail: float
    ia_tail_over_base: float

    verdict: str


def _detect_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _detect_work_root(repo_root: Path, user_work_root: Optional[str]) -> Path:
    if user_work_root:
        return Path(user_work_root).expanduser().resolve()

    candidates = [
        repo_root / 'cafa6_data',
        repo_root,
    ]
    for c in candidates:
        if (c / 'parsed' / 'term_counts.parquet').exists():
            return c

    # Fallback: still return repo_root; errors will be explicit.
    return repo_root


def _load_term_counts(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Normalise schema.
    cols = {c.lower(): c for c in df.columns}

    # Term column
    if 'term' in cols:
        term_col = cols['term']
        df = df.rename(columns={term_col: 'term'})
    else:
        # Common pattern: term stored in index.
        if df.index.name and str(df.index.name).lower() in {'term', 'go_term', 'goterm'}:
            df = df.reset_index().rename(columns={df.columns[0]: df.columns[0]})
            if 'term' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'term'})
        else:
            df = df.reset_index()
            if 'term' not in df.columns and df.shape[1] >= 1:
                df = df.rename(columns={df.columns[0]: 'term'})

    # Count column
    cols = {c.lower(): c for c in df.columns}
    if 'count' in cols:
        count_col = cols['count']
        df = df.rename(columns={count_col: 'count'})
    elif 'n' in cols:
        df = df.rename(columns={cols['n']: 'count'})
    elif 'freq' in cols:
        df = df.rename(columns={cols['freq']: 'count'})
    elif df.shape[1] >= 2:
        # Heuristic: second column is count
        df = df.rename(columns={df.columns[1]: 'count'})
    else:
        raise ValueError(f"Could not infer 'count' column from: {list(df.columns)}")

    df['term'] = df['term'].astype(str)
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(np.int64)

    return df[['term', 'count']].copy()


def _load_ia(path: Path) -> pd.DataFrame:
    # IA.tsv in this repo is sometimes headerless, sometimes has headers.
    ia_df = pd.read_csv(path, sep='\t')

    # If it looks headerless (e.g. columns like 0,1), re-read properly.
    if ia_df.shape[1] >= 2:
        lower_cols = [str(c).lower() for c in ia_df.columns]
        if 'term' in lower_cols and ('ia' in lower_cols or 'information_accretion' in lower_cols):
            term_col = ia_df.columns[lower_cols.index('term')]
            if 'ia' in lower_cols:
                ia_col = ia_df.columns[lower_cols.index('ia')]
            else:
                ia_col = ia_df.columns[lower_cols.index('information_accretion')]
            out = ia_df[[term_col, ia_col]].copy()
            out.columns = ['term', 'ia']
        else:
            # Assume no header.
            ia_df = pd.read_csv(path, sep='\t', header=None, names=['term', 'ia'])
            out = ia_df[['term', 'ia']].copy()
    else:
        ia_df = pd.read_csv(path, sep='\t', header=None, names=['term', 'ia'])
        out = ia_df[['term', 'ia']].copy()

    out['term'] = out['term'].astype(str)
    out['ia'] = pd.to_numeric(out['ia'], errors='coerce').fillna(0).astype(np.float32)
    return out


def _ensure_out_dir(repo_root: Path, out_dir: Optional[str]) -> Path:
    if out_dir:
        p = Path(out_dir).expanduser().resolve()
    else:
        p = repo_root / 'artefacts_local' / 'audits' / 'gbdt_target_scale'
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_and_save(term_counts: pd.DataFrame, base_k: int, expanded_k: int, out_dir: Path) -> None:
    # Plotting is optional at runtime import-time, so we import lazily.
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='whitegrid')

    top = term_counts.iloc[:expanded_k].copy()
    x = np.arange(1, len(top) + 1)

    # Frequency drop-off plot
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, top['count'].to_numpy(), lw=1.25, label='Annotation Frequency')
    ax.axvline(x=base_k, color='red', linestyle='--', label=f'Current GBDT Limit (Top-{base_k})')
    ax.fill_between(
        x,
        top['count'].to_numpy(),
        where=(x >= base_k) & (x <= expanded_k),
        color='orange',
        alpha=0.20,
        label=f'Expansion Zone ({base_k + 1}–{expanded_k})',
    )
    ax.set_title('GO Term Frequency Drop-off (GBDT Signal Audit)')
    ax.set_ylabel('Number of proteins annotated')
    ax.set_xlabel('Term rank (sorted by frequency)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_dir / 'frequency_dropoff_topk.png', dpi=160)
    plt.close(fig)

    # Histogram of tail counts
    tail = term_counts.iloc[base_k:expanded_k].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(tail['count'], bins=50, ax=ax)
    ax.set_title(f'Annotation Count Distribution (Ranks {base_k + 1}–{expanded_k})')
    ax.set_xlabel('Annotations per term')
    ax.set_ylabel('Number of terms')
    fig.tight_layout()
    fig.savefig(out_dir / 'tail_count_hist.png', dpi=160)
    plt.close(fig)

    # Cumulative IA plot (requires `ia` column already present)
    if 'ia' in term_counts.columns:
        top2 = term_counts.iloc[:expanded_k].copy()
        top2['ia_cumsum'] = top2['ia'].cumsum()
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(np.arange(1, len(top2) + 1), top2['ia_cumsum'].to_numpy(), lw=1.5, label='Cumulative IA')
        ax.axvline(x=base_k, color='red', linestyle='--', label=f'Current GBDT Limit (Top-{base_k})')
        ax.set_title('Cumulative IA Coverage by Term Rank')
        ax.set_xlabel('Term rank (sorted by frequency)')
        ax.set_ylabel('Cumulative Information Accretion (IA)')
        ax.legend(loc='upper left')
        fig.tight_layout()
        fig.savefig(out_dir / 'ia_cumsum_topk.png', dpi=160)
        plt.close(fig)


def run_audit(*, work_root: Path, out_dir: Path, cfg: AuditConfig) -> AuditSummary:
    term_counts_path = work_root / 'parsed' / 'term_counts.parquet'
    ia_path = work_root / 'IA.tsv'

    if not term_counts_path.exists():
        raise FileNotFoundError(f"Missing artefact: {term_counts_path}")
    if not ia_path.exists():
        raise FileNotFoundError(f"Missing artefact: {ia_path}")

    term_counts = _load_term_counts(term_counts_path)
    ia_df = _load_ia(ia_path)

    # Rank terms by frequency (highest count first)
    term_counts = term_counts.sort_values('count', ascending=False).reset_index(drop=True)
    term_counts = term_counts.merge(ia_df, on='term', how='left')
    missing_ia_terms_n = int(term_counts['ia'].isna().sum())
    term_counts['ia'] = term_counts['ia'].fillna(0).astype(np.float32)

    total_terms_available = int(len(term_counts))
    base_k = int(min(cfg.base_k, total_terms_available))
    expanded_k = int(min(cfg.expanded_k, total_terms_available))
    if expanded_k <= base_k:
        raise ValueError(f"expanded_k must be > base_k. Got base_k={base_k}, expanded_k={expanded_k}")

    base = term_counts.iloc[:base_k]
    tail = term_counts.iloc[base_k:expanded_k]

    noise_floor = int(cfg.noise_floor)
    tail_below_noise_n = int((tail['count'] < noise_floor).sum())
    tail_below_noise_frac = float(tail_below_noise_n / max(1, len(tail)))

    ia_sum_base = float(base['ia'].sum())
    ia_sum_tail = float(tail['ia'].sum())
    ia_tail_over_base = float(ia_sum_tail / ia_sum_base) if ia_sum_base > 0 else float('inf')

    sum_counts_base = float(base['count'].sum())
    sum_counts_tail = float(tail['count'].sum())
    sum_counts_tail_over_base = float(sum_counts_tail / sum_counts_base) if sum_counts_base > 0 else float('inf')

    base_min_count = float(base['count'].iloc[-1])
    tail_max_count = float(tail['count'].iloc[0])
    tail_min_count = float(tail['count'].iloc[-1])

    tail_mean_count = float(tail['count'].mean())
    tail_median_count = float(tail['count'].median())
    tail_p10_count = float(np.quantile(tail['count'].to_numpy(), 0.10))
    tail_p25_count = float(np.quantile(tail['count'].to_numpy(), 0.25))

    # Verdict logic as per your rubric.
    verdict = 'AUDIT VERDICT: STABLE FOR EXPERIMENTATION.'
    if tail_below_noise_n > (len(tail) / 2):
        verdict = 'CRITICAL AUDIT VERDICT: REJECT EXPANSION.'

    summary = AuditSummary(
        total_terms_available=total_terms_available,
        base_k=base_k,
        expanded_k=expanded_k,
        tail_k=int(len(tail)),
        base_min_count=base_min_count,
        tail_max_count=tail_max_count,
        tail_min_count=tail_min_count,
        tail_mean_count=tail_mean_count,
        tail_median_count=tail_median_count,
        tail_p10_count=tail_p10_count,
        tail_p25_count=tail_p25_count,
        sum_counts_base=sum_counts_base,
        sum_counts_tail=sum_counts_tail,
        sum_counts_tail_over_base=sum_counts_tail_over_base,
        noise_floor=noise_floor,
        tail_below_noise_n=tail_below_noise_n,
        tail_below_noise_frac=tail_below_noise_frac,
        missing_ia_terms_n=missing_ia_terms_n,
        ia_sum_base=ia_sum_base,
        ia_sum_tail=ia_sum_tail,
        ia_tail_over_base=ia_tail_over_base,
        verdict=verdict,
    )

    # Persist outputs
    (out_dir / 'summary.json').write_text(json.dumps(asdict(summary), indent=2), encoding='utf-8')

    try:
        _plot_and_save(term_counts, base_k=base_k, expanded_k=expanded_k, out_dir=out_dir)
    except Exception as e:
        # Plotting should not block audit results.
        (out_dir / 'plotting_error.txt').write_text(str(e), encoding='utf-8')

    return summary


def _print_report(summary: AuditSummary) -> None:
    base_k = summary.base_k
    expanded_k = summary.expanded_k
    tail_k = summary.tail_k

    print(f"--- GBDT Target Audit ({base_k + 1:,} to {expanded_k:,}) ---")
    print(f"Total terms available: {summary.total_terms_available:,}")
    print('')

    print('1) Annotation count drop-off')
    print(f"- Count at rank {base_k:,} (base min): {summary.base_min_count:,.0f}")
    print(f"- Count at rank {base_k + 1:,} (tail max): {summary.tail_max_count:,.0f}")
    print(f"- Count at rank {expanded_k:,} (tail min): {summary.tail_min_count:,.0f}")
    print(f"- Tail mean/median: {summary.tail_mean_count:,.2f} / {summary.tail_median_count:,.2f}")
    print(f"- Tail p10/p25: {summary.tail_p10_count:,.2f} / {summary.tail_p25_count:,.2f}")
    print(f"- Total annotations: base={summary.sum_counts_base:,.0f} tail={summary.sum_counts_tail:,.0f} (tail/base={summary.sum_counts_tail_over_base:.2%})")
    print('')

    print('2) Noise floor (GBDT stability heuristic)')
    print(f"- Noise floor: < {summary.noise_floor} annotations")
    print(f"- Tail terms below noise floor: {summary.tail_below_noise_n:,} / {tail_k:,} ({summary.tail_below_noise_frac:.1%})")
    print('')

    print('3) IA-weight contribution potential')
    if summary.missing_ia_terms_n:
        print(f"- Missing IA for {summary.missing_ia_terms_n:,} terms (filled as 0)")
    print(f"- Total IA (base Top-{base_k:,}): {summary.ia_sum_base:,.2f}")
    print(f"- Total IA (tail {tail_k:,} terms): {summary.ia_sum_tail:,.2f}")
    print(f"- IA potential increase (tail/base): {summary.ia_tail_over_base:.2%}")
    print('')

    print(summary.verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit expanding GBDT targets from 4,500 to 6,000 terms.')
    parser.add_argument('--work-root', type=str, default=None, help='Path to WORK_ROOT (contains parsed/ and IA.tsv).')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory for plots + summary.json.')
    parser.add_argument('--base-k', type=int, default=4500)
    parser.add_argument('--expanded-k', type=int, default=6000)
    parser.add_argument('--noise-floor', type=int, default=50)

    args = parser.parse_args()

    repo_root = _detect_repo_root()
    work_root = _detect_work_root(repo_root, args.work_root)
    out_dir = _ensure_out_dir(repo_root, args.out_dir)

    cfg = AuditConfig(base_k=int(args.base_k), expanded_k=int(args.expanded_k), noise_floor=int(args.noise_floor))

    summary = run_audit(work_root=work_root, out_dir=out_dir, cfg=cfg)
    _print_report(summary)

    print('')
    print(f"Wrote outputs to: {out_dir}")


if __name__ == '__main__':
    main()
