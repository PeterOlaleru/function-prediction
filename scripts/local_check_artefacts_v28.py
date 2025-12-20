from __future__ import annotations

import argparse
from pathlib import Path

V28_REQUIRED = [
    # external
    'external/entryid_text.tsv',
    'external/prop_test_no_kaggle.tsv',
    'external/prop_train_no_kaggle.tsv',
    # features
    'features/test_embeds_esm2.npy',
    'features/test_embeds_esm2_3b.npy',
    'features/test_embeds_t5.npy',
    'features/test_embeds_text.npy',
    'features/text_vectorizer.joblib',
    'features/train_embeds_esm2.npy',
    'features/train_embeds_esm2_3b.npy',
    'features/train_embeds_t5.npy',
    'features/train_embeds_text.npy',
    # parsed
    'parsed/test_seq.feather',
    'parsed/test_taxa.feather',
    'parsed/train_seq.feather',
    'parsed/train_taxa.feather',
    'parsed/term_counts.parquet',
    'parsed/term_priors.parquet',
    'parsed/train_terms.parquet',
]


def main() -> int:
    ap = argparse.ArgumentParser(description='Local check: v28 artefacts present + sizes')
    ap.add_argument('--work-root', type=Path, default=Path('artefacts_local') / 'artefacts', help='Path to WORK_ROOT')
    args = ap.parse_args()

    root = args.work_root
    print(f'WORK_ROOT={root.resolve()}')

    missing: list[str] = []
    total_bytes = 0

    for rel in V28_REQUIRED:
        p = root / rel
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            sz = p.stat().st_size
            total_bytes += sz
            print(f'[OK]   {rel}  ({sz/1e6:.1f} MB)')
        else:
            print(f'[MISS] {rel}')
            missing.append(rel)

    print('')
    print(f'Total size of present required files: {total_bytes/1e9:.2f} GB')

    if missing:
        print(f'Missing {len(missing)} files.')
        return 2

    print('All v28 required artefacts are present.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
