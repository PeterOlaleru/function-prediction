import json
import sys
from pathlib import Path


def _is_blank_line(s: str) -> bool:
    return s.strip() == ""


def _indent_level(s: str) -> int:
    # Count leading spaces/tabs. Tabs count as 4 for heuristics.
    expanded = s.replace("\t", "    ")
    return len(expanded) - len(expanded.lstrip(" "))


def _is_import(s: str) -> bool:
    st = s.lstrip()
    return st.startswith("import ") or st.startswith("from ")


def _is_def_like(s: str) -> bool:
    st = s.lstrip()
    return st.startswith("def ") or st.startswith("class ") or st.startswith("@")


def compact_code_cell_source(source: list[str]) -> list[str]:
    # Normalise line endings and trim trailing whitespace.
    lines = [str(x).replace("\r\n", "\n").replace("\r", "\n").rstrip(" \t\n") for x in source]

    # Remove leading/trailing blanks early.
    while lines and _is_blank_line(lines[0]):
        lines.pop(0)
    while lines and _is_blank_line(lines[-1]):
        lines.pop()
    if not lines:
        return []

    out: list[str] = []
    n = len(lines)
    for i, cur in enumerate(lines):
        if not _is_blank_line(cur):
            out.append(cur)
            continue

        # Decide whether to keep a single blank line here.
        # Look for prev/next nonblank.
        j = i - 1
        while j >= 0 and _is_blank_line(lines[j]):
            j -= 1
        k = i + 1
        while k < n and _is_blank_line(lines[k]):
            k += 1

        if j < 0 or k >= n:
            continue

        prev = lines[j]
        nxt = lines[k]

        prev_indent = _indent_level(prev)
        next_indent = _indent_level(nxt)

        # Keep a single blank line when we dedent to a new top-level def/class/decorator.
        # This avoids mashing top-level defs together when the last line of the previous def is indented.
        if next_indent == 0 and _is_def_like(nxt) and prev_indent > 0:
            if out and out[-1] != "":
                out.append("")
            continue

        # Drop blank lines inside indented blocks.
        if prev_indent > 0 or next_indent > 0:
            continue

        # Keep exactly one blank line after an import block.
        if _is_import(prev) and not _is_import(nxt):
            if out and out[-1] != "":
                out.append("")
            continue

        # Keep blank line between top-level defs/decorators.
        if _is_def_like(nxt) or _is_def_like(prev):
            if out and out[-1] != "":
                out.append("")
            continue

        # Otherwise: remove (this kills the "blank line between every statement" formatting).
        continue

    # Collapse any remaining consecutive blanks (defensive).
    final: list[str] = []
    prev_blank = False
    for ln in out:
        is_blank = _is_blank_line(ln)
        if is_blank and prev_blank:
            continue
        final.append("") if is_blank else final.append(ln)
        prev_blank = is_blank

    # Trim again.
    while final and _is_blank_line(final[0]):
        final.pop(0)
    while final and _is_blank_line(final[-1]):
        final.pop()

    return final


def compact_notebook_sources(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))

    changed_cells = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source")
        if not isinstance(src, list):
            continue

        new_src = compact_code_cell_source([str(x) for x in src])
        if new_src != src:
            cell["source"] = new_src
            changed_cells += 1

    if changed_cells == 0:
        print("No changes needed.")
        return

    path.write_text(json.dumps(nb, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Compacted code cell sources: {changed_cells} cells")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: compact_notebook_sources.py <path-to-ipynb>")
        return 2

    notebook_path = Path(sys.argv[1])
    if not notebook_path.exists():
        raise FileNotFoundError(notebook_path)

    compact_notebook_sources(notebook_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
