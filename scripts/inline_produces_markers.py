import argparse
import json
from pathlib import Path


def normalise_source(source: object) -> list[str]:
    if not isinstance(source, list):
        return []
    return [str(x) for x in source]


def is_produces_marker_cell(cell: dict) -> str | None:
    if cell.get("cell_type") != "code":
        return None
    src = normalise_source(cell.get("source"))
    # ignore pure whitespace lines
    nonblank = [line for line in src if line.strip()]
    if len(nonblank) != 1:
        return None
    line = nonblank[0].strip()
    if not line.startswith("# PRODUCES"):
        return None
    # Return the raw line (keep exactly as-is, but ensure newline)
    return nonblank[0] if nonblank[0].endswith("\n") else (nonblank[0] + "\n")


def insert_marker_into_cell(target_cell: dict, marker_line: str) -> None:
    src = normalise_source(target_cell.get("source"))
    if any((line.strip() == marker_line.strip()) for line in src):
        return
    # Insert after the first "# CELL" header line if present, else at top.
    insert_at = 0
    if src and src[0].lstrip().startswith("# CELL"):
        insert_at = 1
    src.insert(insert_at, marker_line)
    target_cell["source"] = src


def main() -> None:
    parser = argparse.ArgumentParser(description="Inline '# PRODUCES:' marker cells into the following cell.")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/05_cafa_e2e.ipynb"),
        help="Path to .ipynb file",
    )
    args = parser.parse_args()

    notebook_path: Path = args.notebook
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    new_cells: list[dict] = []
    moved = 0
    kept_markers = 0

    i = 0
    while i < len(cells):
        cell = cells[i]
        marker = is_produces_marker_cell(cell)
        if marker is None:
            new_cells.append(cell)
            i += 1
            continue

        # marker cell: try to inline into the next code cell
        if i + 1 < len(cells) and cells[i + 1].get("cell_type") == "code":
            insert_marker_into_cell(cells[i + 1], marker)
            moved += 1
            i += 1  # drop this marker cell
            continue

        # Can't inline (no next cell / next cell not code) -> keep it to avoid data loss.
        new_cells.append(cell)
        kept_markers += 1
        i += 1

    nb["cells"] = new_cells
    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Moved {moved} marker cells into their following cells")
    if kept_markers:
        print(f"Kept {kept_markers} marker cells (no following code cell)")


if __name__ == "__main__":
    main()
