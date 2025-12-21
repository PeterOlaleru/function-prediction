from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _cell_title(cell: dict) -> str:
    src: list[str] = list(cell.get("source", []) or [])
    for line in src[:12]:
        s = line.strip()
        if s.startswith("# CELL"):
            return s
    # fallback: first non-empty line
    for line in src:
        s = line.strip()
        if s:
            return s[:120]
    return "(empty)"


def describe(path: Path) -> list[tuple[int, str, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[int, str, str]] = []
    for i, cell in enumerate(nb.get("cells", []), start=1):
        ctype = cell.get("cell_type", "?")
        title = _cell_title(cell)
        out.append((i, ctype, title))
    return out


def _print_table(rows: Iterable[tuple[int, str, str]]) -> None:
    for i, ctype, title in rows:
        print(f"{i:02d} [{ctype}] {title}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nb05 = root / "notebooks" / "05_cafa_e2e.ipynb"
    nb04 = root / "notebooks" / "Colab_04_all_in_one.ipynb"
    nb04b = root / "notebooks" / "Colab_04b_first_submission_no_ankh.ipynb"

    for p in [nb05, nb04, nb04b]:
        print("=" * 90)
        print(p.name)
        print("=" * 90)
        _print_table(describe(p))
        print()


if __name__ == "__main__":
    main()
