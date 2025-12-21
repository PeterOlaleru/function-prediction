"""Static-ish name analysis for Jupyter notebooks.

Goals (practical, not perfect):
- Flag names used before being defined (per cell order).
- Flag names defined (assigned/def/class/import) but never used later.

Limitations:
- Jupyter is dynamic: this is heuristic; expect some false positives.
- We ignore attribute usage (e.g., np.array) as defining `np` is what matters.
- We ignore IPython magics/shell lines.

Usage:
  python scripts/analyse_notebook_names.py notebooks/05_cafa_e2e.ipynb
"""

from __future__ import annotations

import ast
import builtins
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CellReport:
    cell_number: int
    undefined_uses: list[str]
    defined_names: list[str]


PY_BUILTINS = set(dir(builtins))


def _is_magic_or_shell(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("%") or s.startswith("!")


def _strip_magics(code: str) -> str:
    lines = []
    for line in code.splitlines():
        if _is_magic_or_shell(line):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


class NameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.defined: set[str] = set()
        self.used: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Param)):
            self.defined.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        # Function parameters (incl. methods' `self`) are definitions.
        if node.arg:
            self.defined.add(node.arg)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # `except Exception as e:` defines `e`.
        if isinstance(node.name, str) and node.name:
            self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            self.defined.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined.add(node.name)
        self.generic_visit(node)


def _parse_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    out: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            code = "".join(src)
        else:
            code = str(src)
        out.append(code)
    return out


def analyse(nb_path: Path) -> tuple[list[CellReport], list[str]]:
    code_cells = _parse_cells(nb_path)

    defined_so_far: set[str] = set(PY_BUILTINS)
    defined_so_far.update({"In", "Out", "get_ipython"})

    reports: list[CellReport] = []

    # Track when a name was defined (first cell number).
    defined_at: dict[str, int] = {}

    # Track names used in any later cell.
    used_later: set[str] = set()

    per_cell_defined: list[set[str]] = []
    per_cell_used: list[set[str]] = []

    for i, raw in enumerate(code_cells, start=1):
        code = _strip_magics(raw)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # If syntax is broken, defer to the separate syntax checker.
            collector = NameCollector()
            per_cell_defined.append(set())
            per_cell_used.append(set())
            reports.append(CellReport(i, undefined_uses=[], defined_names=[]))
            continue

        collector = NameCollector()
        collector.visit(tree)

        per_cell_defined.append(set(collector.defined))
        per_cell_used.append(set(collector.used))

        # Treat names defined within the same cell as available, otherwise imports/defs
        # used later in the cell (e.g. in f-strings) become false positives.
        defined_now = defined_so_far | collector.defined
        undefined = sorted(
            n
            for n in collector.used
            if n not in defined_now and n not in {"__name__", "__file__"}
        )

        # Update defined set after checking undefined (order matters).
        for n in collector.defined:
            defined_so_far.add(n)
            defined_at.setdefault(n, i)

        reports.append(
            CellReport(cell_number=i, undefined_uses=undefined, defined_names=sorted(collector.defined))
        )

    # Compute “defined but never used later” (heuristic).
    # We only consider names defined in code cells, not builtins.
    all_defined = set().union(*per_cell_defined) if per_cell_defined else set()

    # Any name used in any cell after its first definition counts as used.
    for i in range(len(code_cells)):
        used_later.update(per_cell_used[i])

    defined_never_used = []
    for name in sorted(all_defined):
        if name.startswith("_"):
            continue
        if name in PY_BUILTINS:
            continue
        if name in {"In", "Out", "get_ipython"}:
            continue
        if name not in used_later:
            defined_never_used.append(name)

    return reports, defined_never_used


def _print_report(nb_path: Path, reports: Iterable[CellReport], defined_never_used: list[str]) -> None:
    print(f"notebook={nb_path.as_posix()}")

    any_undefined = False
    for r in reports:
        if r.undefined_uses:
            any_undefined = True
            print(f"\nCell {r.cell_number}: undefined name uses")
            for n in r.undefined_uses:
                print(f"  - {n}")

    if not any_undefined:
        print("\nNo undefined name uses detected (heuristic).")

    if defined_never_used:
        print("\nDefined but never used later (heuristic):")
        for n in defined_never_used:
            print(f"  - {n}")
    else:
        print("\nNo obviously-unused definitions detected (heuristic).")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python scripts/analyse_notebook_names.py <notebook.ipynb>")
        return 2

    nb_path = Path(argv[1])
    reports, defined_never_used = analyse(nb_path)
    _print_report(nb_path, reports, defined_never_used)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
