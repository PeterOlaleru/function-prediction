from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Stage the minimal Option B artefacts into a folder that can be published as a Kaggle Dataset. "
            "This is intentionally strict and will error if required files are missing."
        )
    )
    ap.add_argument(
        "--artefacts-dir",
        type=Path,
        required=True,
        help="Path to artefacts directory (contains external/, features/, parsed/ etc).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output staging directory to create/overwrite.",
    )
    args = ap.parse_args()

    artefacts_dir: Path = args.artefacts_dir
    out_dir: Path = args.out_dir

    required = [
        # TF-IDF
        artefacts_dir / "features" / "text_vectorizer.joblib",
        artefacts_dir / "features" / "train_embeds_text.npy",
        artefacts_dir / "features" / "test_embeds_text.npy",
        # External priors (Phase 1 Step 4 outputs)
        artefacts_dir / "external" / "prop_train_no_kaggle.tsv.gz",
        artefacts_dir / "external" / "prop_test_no_kaggle.tsv.gz",
        # Term space definition used downstream
        artefacts_dir / "features" / "top_terms_1500.json",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required Option B artefacts:\n" + "\n".join([f" - {p}" for p in missing])
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in required:
        rel = p.relative_to(artefacts_dir)
        _copy_file(p, out_dir / rel)

    print("Staged Option B artefacts to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
