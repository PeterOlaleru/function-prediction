from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ensure_metadata(dataset_id: str, title: str, out_dir: Path) -> Path:
    meta_path = out_dir / "dataset-metadata.json"
    if meta_path.exists():
        return meta_path

    meta = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Publish a local folder as a Kaggle Dataset using the Kaggle API. "
            "Creates the dataset if it does not exist; otherwise pushes a new version."
        )
    )
    ap.add_argument("--dataset", required=True, help="Dataset id: <username>/<dataset-slug>")
    ap.add_argument("--title", required=True, help="Dataset title shown on Kaggle")
    ap.add_argument("--dir", required=True, type=Path, help="Directory to publish")
    ap.add_argument("--message", default="Update artefacts", help="Version notes")
    ap.add_argument("--private", action="store_true", help="Create as private (first create only)")
    args = ap.parse_args()

    out_dir: Path = args.dir
    if not out_dir.exists():
        raise FileNotFoundError(f"Directory not found: {out_dir}")

    _ensure_metadata(args.dataset, args.title, out_dir)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Kaggle API client not installed. Install it with: pip install kaggle\n"
            "Then ensure your API token is available (see docs/QUICK_START.md)."
        ) from e

    api = KaggleApi()
    api.authenticate()

    # Create if missing, else version.
    try:
        api.dataset_view(args.dataset)
        api.dataset_create_version(
            str(out_dir),
            version_notes=args.message,
            quiet=False,
            delete_old_versions=False,
        )
        print("Published new dataset version:", args.dataset)
    except Exception:
        api.dataset_create_new(
            str(out_dir),
            public=not args.private,
            quiet=False,
        )
        print("Created new dataset:", args.dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
