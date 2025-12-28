"""Find an evidence-based Top-K target size per GO aspect (BP/MF/CC).

Motivation
- Picking a single global K (e.g. 4,500) is a blunt instrument.
- For GBDTs, very rare labels are effectively noise. Yet IA tends to reward rarer terms.
- We therefore quantify the trade-off per aspect:
  - annotation-frequency drop-off,
  - fraction below a noise floor (default 50 positives),
  - cumulative IA coverage as K increases.

We report several 'optimal K' candidates (per aspect):
- K_noise: number of terms with count >= noise_floor (hard stability ceiling).
- K_95_ia_eligible: smallest K (within eligible terms) to reach 95% of IA among eligible terms.
- K_99_ia_eligible: smallest K (within eligible terms) to reach 99% of IA among eligible terms.

Outputs
- Console report
- JSON summary
- Plots per aspect under artefacts_local/audits/optimal_topk_by_aspect/

British English output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AspectResult:
    aspect: str
    n_terms_total: int
    n_terms_noise_eligible: int
    noise_floor: int

    # Coverage stats
    ia_total_all_terms: float
    ia_total_noise_eligible: float

    k_noise: int
    k_95_ia_eligible: int
    k_99_ia_eligible: int

    ia_at_k_noise: float
    ia_at_k_noise_frac_of_all: float
    ia_at_k_noise_frac_of_eligible: float

    # Sanity: counts at key ranks (within aspect)
    count_at_k_noise: float
    count_at_k_noise_plus_1: float


@dataclass(frozen=True)
class AuditSummary:
    work_root: str
    base_sort: str
    noise_floor: int
    aspects: Dict[str, AspectResult]


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
    return repo_root


def _load_term_counts(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    cols = {c.lower(): c for c in df.columns}

    if 'term' in cols:
        df = df.rename(columns={cols['term']: 'term'})
    else:
        df = df.reset_index()
        if 'term' not in df.columns:
            df = df.rename(columns={df.columns[0]: 'term'})

    cols = {c.lower(): c for c in df.columns}
    if 'count' in cols:
        df = df.rename(columns={cols['count']: 'count'})
    elif 'n' in cols:
        df = df.rename(columns={cols['n']: 'count'})
    elif df.shape[1] >= 2:
        df = df.rename(columns={df.columns[1]: 'count'})
    else:
        raise ValueError(f"Could not infer 'count' column from: {list(df.columns)}")

    df['term'] = df['term'].astype(str)
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(np.int64)
    return df[['term', 'count']].copy()


def _load_ia(path: Path) -> pd.DataFrame:
    ia_df = pd.read_csv(path, sep='\t')
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
            ia_df = pd.read_csv(path, sep='\t', header=None, names=['term', 'ia'])
            out = ia_df[['term', 'ia']].copy()
    else:
        ia_df = pd.read_csv(path, sep='\t', header=None, names=['term', 'ia'])
        out = ia_df[['term', 'ia']].copy()

    out['term'] = out['term'].astype(str)
    out['ia'] = pd.to_numeric(out['ia'], errors='coerce').fillna(0).astype(np.float32)
    return out


def _load_term_to_aspect(obo_path: Path) -> Dict[str, str]:
    import obonet

    graph = obonet.read_obo(obo_path)
    out: Dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        ns = data.get('namespace')
        if ns == 'biological_process':
            out[str(node)] = 'BP'
        elif ns == 'molecular_function':
            out[str(node)] = 'MF'
        elif ns == 'cellular_component':
            out[str(node)] = 'CC'
    return out


def _ensure_out_dir(repo_root: Path, out_dir: Optional[str]) -> Path:
    p = Path(out_dir).expanduser().resolve() if out_dir else (repo_root / 'artefacts_local' / 'audits' / 'optimal_topk_by_aspect')
    p.mkdir(parents=True, exist_ok=True)
    return p


def _first_k_reaching_fraction(cumsum: np.ndarray, total: float, frac: float) -> int:
    if total <= 0:
        return 0
    target = frac * total
    idx = int(np.searchsorted(cumsum, target, side='left'))
    return int(min(len(cumsum), idx + 1))


def _plot_aspect(df: pd.DataFrame, aspect: str, out_dir: Path, noise_floor: int, k_noise: int) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='whitegrid')

    # Count drop-off curve
    x = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, df['count'].to_numpy(), lw=1.25, label='Annotation Frequency')
    ax.axhline(y=noise_floor, color='grey', linestyle='--', label=f'Noise floor ({noise_floor})')
    if k_noise > 0:
        ax.axvline(x=k_noise, color='red', linestyle='--', label=f'K_noise={k_noise:,}')
    ax.set_title(f'{aspect}: Annotation Frequency Drop-off (sorted by count)')
    ax.set_xlabel('Term rank (within aspect)')
    ax.set_ylabel('Annotations per term')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_dir / f'{aspect}_count_dropoff.png', dpi=160)
    plt.close(fig)

    # IA cumulative curve
    fig, ax = plt.subplots(figsize=(11, 5))
    ia = df['ia'].to_numpy(dtype=np.float64)
    ia_cumsum = np.cumsum(ia)
    ax.plot(x, ia_cumsum, lw=1.5, label='Cumulative IA')
    if k_noise > 0:
        ax.axvline(x=k_noise, color='red', linestyle='--', label=f'K_noise={k_noise:,}')
    ax.set_title(f'{aspect}: Cumulative IA by Term Rank')
    ax.set_xlabel('Term rank (within aspect)')
    ax.set_ylabel('Cumulative IA')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_dir / f'{aspect}_ia_cumsum.png', dpi=160)
    plt.close(fig)

    # Tail histogram below noise floor (optional)
    tail = df[df['count'] < noise_floor]
    if len(tail):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.histplot(tail['count'], bins=30, ax=ax)
        ax.set_title(f'{aspect}: Below-noise terms count distribution (<{noise_floor})')
        ax.set_xlabel('Annotations per term')
        ax.set_ylabel('Number of terms')
        fig.tight_layout()
        fig.savefig(out_dir / f'{aspect}_below_noise_hist.png', dpi=160)
        plt.close(fig)


def _compute_aspect(term_counts_ia: pd.DataFrame, aspect: str, noise_floor: int, out_dir: Path) -> AspectResult:
    df = term_counts_ia[term_counts_ia['aspect'] == aspect].copy()
    df = df.sort_values('count', ascending=False).reset_index(drop=True)

    n_terms_total = int(len(df))

    eligible = df[df['count'] >= noise_floor].copy()
    n_terms_noise_eligible = int(len(eligible))

    ia_total_all_terms = float(df['ia'].sum())
    ia_total_noise_eligible = float(eligible['ia'].sum())

    # K_noise is the eligible term count.
    k_noise = n_terms_noise_eligible

    # Compute IA thresholds over eligible-only ordering (still sorted by count).
    # If eligible is empty, these Ks are 0.
    if n_terms_noise_eligible > 0:
        ia_eligible = eligible['ia'].to_numpy(dtype=np.float64)
        ia_eligible_cumsum = np.cumsum(ia_eligible)
        k_95 = _first_k_reaching_fraction(ia_eligible_cumsum, total=float(ia_eligible_cumsum[-1]), frac=0.95)
        k_99 = _first_k_reaching_fraction(ia_eligible_cumsum, total=float(ia_eligible_cumsum[-1]), frac=0.99)
        ia_at_k_noise = float(ia_eligible_cumsum[-1])
    else:
        k_95 = 0
        k_99 = 0
        ia_at_k_noise = 0.0

    ia_at_k_noise_frac_of_all = float(ia_at_k_noise / ia_total_all_terms) if ia_total_all_terms > 0 else float('inf')
    ia_at_k_noise_frac_of_eligible = float(ia_at_k_noise / ia_total_noise_eligible) if ia_total_noise_eligible > 0 else float('inf')

    # Counts at the boundary.
    if k_noise > 0:
        count_at_k_noise = float(df.iloc[k_noise - 1]['count'])
        count_at_k_noise_plus_1 = float(df.iloc[k_noise]['count']) if k_noise < n_terms_total else 0.0
    else:
        count_at_k_noise = 0.0
        count_at_k_noise_plus_1 = float(df.iloc[0]['count']) if n_terms_total else 0.0

    # Plot using full df ordering.
    _plot_aspect(df, aspect=aspect, out_dir=out_dir, noise_floor=noise_floor, k_noise=k_noise)

    return AspectResult(
        aspect=aspect,
        n_terms_total=n_terms_total,
        n_terms_noise_eligible=n_terms_noise_eligible,
        noise_floor=int(noise_floor),
        ia_total_all_terms=ia_total_all_terms,
        ia_total_noise_eligible=ia_total_noise_eligible,
        k_noise=k_noise,
        k_95_ia_eligible=int(k_95),
        k_99_ia_eligible=int(k_99),
        ia_at_k_noise=ia_at_k_noise,
        ia_at_k_noise_frac_of_all=ia_at_k_noise_frac_of_all,
        ia_at_k_noise_frac_of_eligible=ia_at_k_noise_frac_of_eligible,
        count_at_k_noise=count_at_k_noise,
        count_at_k_noise_plus_1=count_at_k_noise_plus_1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute per-aspect optimal Top-K under IA and noise-floor constraints.')
    parser.add_argument('--work-root', type=str, default=None, help='WORK_ROOT (contains parsed/term_counts.parquet and IA.tsv).')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory for plots + JSON summary.')
    parser.add_argument('--noise-floor', type=int, default=50, help='GBDT noise floor in positive labels (default 50).')
    parser.add_argument('--sort', type=str, default='count', choices=['count'], help='Sorting basis for target selection (fixed to count).')

    args = parser.parse_args()

    repo_root = _detect_repo_root()
    work_root = _detect_work_root(repo_root, args.work_root)
    out_dir = _ensure_out_dir(repo_root, args.out_dir)

    term_counts_path = work_root / 'parsed' / 'term_counts.parquet'
    ia_path = work_root / 'IA.tsv'

    # GO OBO for aspect mapping.
    obo_candidates = [
        work_root / 'Train' / 'go-basic.obo',
        work_root / 'go-basic.obo',
        repo_root / 'Train' / 'go-basic.obo',
    ]
    obo_path = next((p for p in obo_candidates if p.exists()), None)
    if obo_path is None:
        raise FileNotFoundError(f"go-basic.obo not found. Tried: {[str(p) for p in obo_candidates]}")

    term_counts = _load_term_counts(term_counts_path)
    ia_df = _load_ia(ia_path)
    term_to_aspect = _load_term_to_aspect(obo_path)

    df = term_counts.merge(ia_df, on='term', how='left')
    df['ia'] = df['ia'].fillna(0).astype(np.float32)
    df['aspect'] = df['term'].map(term_to_aspect).fillna('UNK')

    # Drop unknown-aspect terms (should be very few; they are not used for CAFA scoring).
    df = df[df['aspect'].isin(['BP', 'MF', 'CC'])].copy()

    results: Dict[str, AspectResult] = {}
    for aspect in ['BP', 'MF', 'CC']:
        results[aspect] = _compute_aspect(df, aspect=aspect, noise_floor=int(args.noise_floor), out_dir=out_dir)

    summary = AuditSummary(
        work_root=str(work_root),
        base_sort=str(args.sort),
        noise_floor=int(args.noise_floor),
        aspects=results,
    )

    # JSON serialise with nested dataclasses.
    out = {
        'work_root': summary.work_root,
        'base_sort': summary.base_sort,
        'noise_floor': summary.noise_floor,
        'aspects': {k: asdict(v) for k, v in summary.aspects.items()},
    }
    (out_dir / 'summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')

    # Console report
    print('--- Optimal Top-K by Aspect (GBDT target selection) ---')
    print(f"WORK_ROOT: {summary.work_root}")
    print(f"Noise floor: < {summary.noise_floor} positives")
    print('')

    for aspect in ['BP', 'MF', 'CC']:
        r = results[aspect]
        frac_below = 1.0 - (r.n_terms_noise_eligible / max(1, r.n_terms_total))
        print(f"{aspect}:")
        print(f"- Total terms: {r.n_terms_total:,}")
        print(f"- Noise-eligible terms (>= {r.noise_floor}): {r.n_terms_noise_eligible:,} ({(1-frac_below):.1%})")
        print(f"- K_noise (hard ceiling): {r.k_noise:,}")
        print(f"- K_95_ia_eligible: {r.k_95_ia_eligible:,}")
        print(f"- K_99_ia_eligible: {r.k_99_ia_eligible:,}")
        print(f"- IA total (all terms): {r.ia_total_all_terms:,.2f}")
        print(f"- IA total (eligible only): {r.ia_total_noise_eligible:,.2f}")
        print(f"- IA captured by eligible set: {r.ia_at_k_noise:,.2f} (eligible={r.ia_at_k_noise_frac_of_eligible:.1%}, all={r.ia_at_k_noise_frac_of_all:.1%})")
        print(f"- Count at K_noise: {r.count_at_k_noise:,.0f}; next: {r.count_at_k_noise_plus_1:,.0f}")
        print('')

    print(f"Wrote outputs to: {out_dir}")


if __name__ == '__main__':
    main()
