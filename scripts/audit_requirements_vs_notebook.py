from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "05_cafa_e2e.ipynb"
REQ_PATH = ROOT / "requirements.txt"


_REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def parse_requirements(path: Path) -> set[str]:
    pkgs: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _REQ_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        # Ignore editable installs / local paths if any appear later.
        if name.startswith("-"):
            continue
        pkgs.add(name)
    return pkgs


def extract_imports_from_ipynb(path: Path) -> set[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    imports: set[str] = set()

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if not isinstance(src, list):
            continue
        for line in src:
            if not isinstance(line, str):
                continue
            s = line.strip()
            if s.startswith("import "):
                rest = s[len("import ") :]
                parts = [p.strip() for p in rest.split(",")]
                for p in parts:
                    mod = p.split(" as ")[0].strip()
                    top = mod.split(".")[0].strip()
                    if top:
                        imports.add(top)
            elif s.startswith("from "):
                # from x.y import z
                rest = s[len("from ") :]
                mod = rest.split(" import ")[0].strip()
                top = mod.split(".")[0].strip()
                if top:
                    imports.add(top)

    return imports


def main() -> None:
    req_pkgs = parse_requirements(REQ_PATH)
    imports = extract_imports_from_ipynb(NB_PATH)

    # Map requirements package names -> import module name where they differ.
    # (This matches the logic we use in Cell 2.)
    pkg_to_mod = {
        "pyyaml": "yaml",
        "biopython": "Bio",
        "py-boost": "py_boost",
        "scikit-learn": "sklearn",
    }

    # Build reverse map mod->pkg for comparison.
    mod_to_pkg = {v: k for k, v in pkg_to_mod.items()}

    # Determine which requirements are needed by imports.
    needed_req_pkgs: set[str] = set()
    for mod in imports:
        if mod in mod_to_pkg:
            needed_req_pkgs.add(mod_to_pkg[mod])
        else:
            # Assume pip pkg name == module name.
            needed_req_pkgs.add(mod)

    # Intersection / diff against requirements.txt
    needed_and_listed = sorted(needed_req_pkgs & req_pkgs)
    needed_but_not_listed = sorted(needed_req_pkgs - req_pkgs)
    listed_but_not_used = sorted(req_pkgs - needed_req_pkgs)

    print("Notebook:", NB_PATH)
    print("Requirements:", REQ_PATH)
    print()

    print("Imports found (top-level):")
    print("  ", ", ".join(sorted(imports)))
    print()

    print("Needed packages that ARE in requirements.txt (OK):")
    for p in needed_and_listed:
        print("  -", p)
    print()

    print("Needed by notebook but NOT present in requirements.txt (check if builtin/implicit):")
    for p in needed_but_not_listed:
        print("  -", p)
    print()

    print("In requirements.txt but NOT imported in notebook (may still be runtime deps):")
    for p in listed_but_not_used:
        print("  -", p)


if __name__ == "__main__":
    main()
