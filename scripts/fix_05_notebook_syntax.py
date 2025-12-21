from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "05_cafa_e2e.ipynb"


_RE_OOF = re.compile(r"^\s*(\w+_oof_path)\s*=")
_RE_TEST = re.compile(r"^\s*(\w+_test_path)\s*=")


def _ensure_trailing_nl(s: str) -> str:
    return s if s.endswith("\n") else s + "\n"


def _fix_empty_if_not_push_existing(lines: list[str]) -> list[str]:
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)

        if line.strip() != "if not PUSH_EXISTING_CHECKPOINTS:":
            continue

        # If the very next line is at the same indent (4 spaces) and starts another `if`,
        # this `if` has no body.
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt.startswith("    ") and (not nxt.startswith("        ")) and nxt.lstrip().startswith(
                "if PUSH_EXISTING_CHECKPOINTS"
            ):
                out.append("        pass\n")

    return out


def _fix_missing_cached_preds_guard(lines: list[str]) -> list[str]:
    """Fix a common corruption: cached-preds load lines + `else:` exist, but the `if ...exists()` line is missing.

    Looks for:
      <oof_path assignment>
      <test_path assignment>
          oof_pred_* = np.load(oof_path)
          test_pred_* = np.load(test_path)
      else:

    and inserts:
      if <oof_path>.exists() and <test_path>.exists():

    at the correct indentation.
    """

    out: list[str] = []
    last_oof: str | None = None
    last_test: str | None = None

    i = 0
    while i < len(lines):
        line = lines[i]

        if isinstance(line, str):
            mo = _RE_OOF.match(line)
            if mo:
                last_oof = mo.group(1)
            mt = _RE_TEST.match(line)
            if mt:
                last_test = mt.group(1)

        if (
            last_oof
            and last_test
            and line.startswith("        ")
            and (not line.startswith("            "))
            and (f"np.load({last_oof}" in line)
        ):
            # Look ahead for an `else:` at indent 4 (some blocks have extra cached-body lines).
            else_at: int | None = None
            for j in range(i + 1, min(i + 8, len(lines))):
                s = lines[j]
                if not isinstance(s, str):
                    continue
                if s.startswith("    ") and (not s.startswith("        ")) and s.strip() == "else:":
                    else_at = j
                    break

            # If we found the paired else, and the immediate next line loads the test path,
            # then we're missing the `if <oof>.exists() and <test>.exists():` guard.
            if else_at is not None and i + 1 < len(lines):
                nxt = lines[i + 1]
                if (
                    isinstance(nxt, str)
                    and nxt.startswith("        ")
                    and (f"np.load({last_test}" in nxt)
                ):
                    prev = out[-1] if out else ""
                    if not (prev.strip().startswith("if ") and "exists()" in prev):
                        out.append(f"    if {last_oof}.exists() and {last_test}.exists():\n")

        out.append(line)
        i += 1

    return out


def _patch_cell(nb: dict, cell_index_1based: int, fn) -> bool:
    cell = nb.get("cells", [])[cell_index_1based - 1]
    src = cell.get("source", [])
    if not isinstance(src, list):
        return False

    src2 = [_ensure_trailing_nl(str(x)) for x in src]
    new = fn(src2)

    changed = new != src2
    if changed:
        cell["source"] = new
    return changed


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    changed_any = False

    # Cell 13: missing bodies under `if not PUSH_EXISTING_CHECKPOINTS:` (multiple occurrences)
    changed_any |= _patch_cell(nb, 13, _fix_empty_if_not_push_existing)

    # Cells 15–18: cached-preds load blocks missing their `if ...exists()` guard
    for idx in [15, 16, 17, 18]:
        changed_any |= _patch_cell(nb, idx, _fix_missing_cached_preds_guard)

    if changed_any:
        NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Updated: {NB_PATH}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
