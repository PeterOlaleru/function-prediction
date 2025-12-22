import argparse
import json
from pathlib import Path


def is_blank_source(source: object) -> bool:
    if not isinstance(source, list):
        return True
    if len(source) == 0:
        return True
    return all((not (str(line or "").strip())) for line in source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove empty cells from a Jupyter notebook JSON.")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/05_cafa_e2e.ipynb"),
        help="Path to .ipynb file",
    )
    parser.add_argument(
        "--keep-if-has-outputs",
        action="store_true",
        help="Keep blank-source cells if they contain outputs.",
    )
    args = parser.parse_args()

    notebook_path: Path = args.notebook
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    kept = []
    removed = 0
    for cell in cells:
        blank = is_blank_source(cell.get("source"))
        outputs = cell.get("outputs")
        has_outputs = isinstance(outputs, list) and len(outputs) > 0

        if blank and (not args.keep_if_has_outputs or not has_outputs):
            removed += 1
            continue

        kept.append(cell)

    nb["cells"] = kept
    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Removed {removed} empty cells; kept {len(kept)} of {len(cells)}")


if __name__ == "__main__":
    main()
