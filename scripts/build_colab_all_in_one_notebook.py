import json
from pathlib import Path
from typing import Any


def _code_cell(source: list[str], cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {"id": cell_id, "language": "python"},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def _load_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_source_lines(text: str) -> list[str]:
    # Notebook JSON stores source as list of lines WITHOUT enforced trailing newlines.
    # We keep a reasonable convention: every line ends with \n except possibly the last.
    lines = text.splitlines(True)
    if not lines:
        return []
    if not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    return lines


def _prepend_cell_title(source: list[str], title: str) -> list[str]:
    """Prepend a single-line header to a code cell so users can 'see' the cell name."""
    # Keep titles ASCII-only (Colab/Windows consoles can choke on Unicode dashes/arrows).
    hdr = f"# CELL -- {title}\n"
    if not source:
        return [hdr]
    # Avoid double-prepending if regenerated.
    first = str(source[0])
    if first.startswith("# ") and first.strip() == hdr.strip():
        return source
    return [hdr, "\n"] + source


def _renumber_cell_titles(cells: list[dict[str, Any]]) -> None:
    """Rewrite '# CELL -- <desc>' headers to '# CELL NN - <desc>' sequentially."""
    n = 0
    for c in cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", [])
        if not isinstance(src, list) or not src:
            continue
        first = str(src[0]).strip("\n")
        if not first.startswith("# CELL"):
            continue

        # Extract description after the marker.
        desc = first
        if "--" in first:
            desc = first.split("--", 1)[1].strip()
        elif "-" in first:
            desc = first.split("-", 1)[1].strip()
        else:
            desc = first.lstrip("#").strip()

        n += 1
        src[0] = f"# CELL {n:02d} - {desc}\n"
        c["source"] = src


def _cell_contains(source: list[str], needle: str) -> bool:
    return needle in "".join(source)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    solution_nb_path = repo_root / "notebooks" / "CAFA6_Rank1_Solution.ipynb"
    out_nb_path = repo_root / "notebooks" / "Colab_04_all_in_one.ipynb"

    script_corpus_path = repo_root / "scripts" / "03_build_entryid_text_from_uniprot_pubmed.py"
    script_embed_path = repo_root / "scripts" / "02_generate_optional_embeddings.py"

    nb = _load_notebook(solution_nb_path)

    # Keep only code cells from the solution notebook ("pure code").
    base_cells: list[dict[str, Any]] = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", [])
        if isinstance(src, str):
            src = _as_source_lines(src)
        # Carry across; we will prepend visible titles for clarity.
        base_cells.append({
            "cell_type": "code",
            "metadata": {"language": "python"},
            "execution_count": None,
            "outputs": [],
            "source": src,
        })

    bootstrap = _code_cell(
        _prepend_cell_title(
            _as_source_lines(
            """
# Colab bootstrap (no external notebook/script execution)
from pathlib import Path
import os
import shutil
import subprocess

REPO_URL = os.environ.get('CAFA_REPO_GIT_URL', 'https://github.com/PeterOla/cafa-6-protein-function-prediction.git')
REPO_DIR = Path(os.environ.get('CAFA_REPO_DIR', '/content/cafa-6-protein-function-prediction'))
SAFE_CWD = Path('/content') if Path('/content').exists() else Path('/')

def run(cmd: list[str]) -> None:
    cmd_str = ' '.join(cmd)
    print('+', cmd_str)
    p = subprocess.run(cmd, text=True, capture_output=True, cwd=str(SAFE_CWD))
    if p.stdout.strip():
        print(p.stdout)
    if p.stderr.strip():
        print(p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f'Command failed (exit={p.returncode}): {cmd_str}')

os.chdir(SAFE_CWD)
if REPO_DIR.exists() and (REPO_DIR / '.git').is_dir():
    run(['git', '-C', str(REPO_DIR), 'fetch', '--depth', '1', 'origin'])
    run(['git', '-C', str(REPO_DIR), 'reset', '--hard', 'origin/HEAD'])
else:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR, ignore_errors=True)
    run(['git', 'clone', '--depth', '1', REPO_URL, str(REPO_DIR)])

os.chdir(REPO_DIR)
print('CWD:', Path.cwd())
""".lstrip()
            ),
            "Bootstrap repo (clone/update)",
        ),
        cell_id="bootstrap",
    )

    install = _code_cell(
        _prepend_cell_title(_as_source_lines("%pip -q install -r requirements.txt\n"), "Install dependencies"),
        cell_id="install",
    )

    # Inline the full text-corpus builder script as a cell.
    corpus_src = script_corpus_path.read_text(encoding="utf-8")
    corpus_cell = _code_cell(
        _prepend_cell_title(_as_source_lines(corpus_src + "\n"), "Inline: EntryID->text corpus builder"),
        cell_id="inline_corpus_script",
    )
    corpus_run = _code_cell(
        _prepend_cell_title(
            _as_source_lines(
            """
# Run corpus builder (EntryID -> text) in-process (no !python)
import sys

# Uses ARTEFACTS_DIR defined earlier in the solution notebook.
# Small smoke run; change max-ids to 0 for full.
sys.argv = [
    '03_build_entryid_text_from_uniprot_pubmed.py',
    '--artefacts-dir', str(ARTEFACTS_DIR),
    '--out-path', str(ARTEFACTS_DIR / 'external' / 'entryid_text.tsv'),
    '--cache-dir', str(ARTEFACTS_DIR / 'external' / 'uniprot_pubmed_cache'),
    '--max-ids', '1000',
    '--max-pubmed-per-protein', '3',
    '--strip-go',
    '--sleep-uniprot', '0.1',
    '--sleep-pubmed', '0.34',
]

_ = main()
""".lstrip()
            ),
            "Run: build EntryID->text corpus (UniProt+PubMed)",
        ),
        cell_id="run_corpus",
    )

    # Inline the full optional embeddings generator script.
    embed_src = script_embed_path.read_text(encoding="utf-8")
    embed_cell = _code_cell(
        _prepend_cell_title(_as_source_lines(embed_src + "\n"), "Inline: optional embeddings generator"),
        cell_id="inline_embed_script",
    )
    embed_run = _code_cell(
        _prepend_cell_title(
            _as_source_lines(
            """
# Run TF-IDF embeddings generation in-process (no !python)
import sys

sys.argv = [
    '02_generate_optional_embeddings.py',
    '--artefacts-dir', str(ARTEFACTS_DIR),
    '--mode', 'text',
    '--text-path', str(ARTEFACTS_DIR / 'external' / 'entryid_text.tsv'),
    '--text-dim', '10279',
]

_ = main()
""".lstrip()
            ),
            "Run: TF-IDF embeddings (10279D)",
        ),
        cell_id="run_tfidf",
    )

    # Insert after the FASTA->feather step within the solution notebook.
    # We detect that cell via a simple substring.
    inserted = False
    out_cells: list[dict[str, Any]] = [bootstrap, install]

    # Prepend titles to solution cells while keeping their internal comments intact.
    for idx, c in enumerate(base_cells, start=1):
        src = c.get("source", [])
        if not isinstance(src, list):
            src = _as_source_lines(str(src))

        # Derive a short label from the first meaningful comment line if present.
        label = None
        for line in src[:20]:
            s = str(line).strip()
            if s.startswith("#") and len(s) > 2:
                label = s.lstrip("#").strip()
                break
        title = "Solution" + (f": {label}" if label else "")
        c["source"] = _prepend_cell_title(src, title)

        out_cells.append(c)

        # Insert corpus+tfidf immediately after the FASTA->feather step.
        src_joined = "".join(c["source"])
        if (not inserted) and ("Parse FASTA to Feather" in src_joined or "parse_fasta(PATH_TRAIN_FASTA)" in src_joined):
            out_cells.extend([corpus_cell, corpus_run, embed_cell, embed_run])
            inserted = True

    if not inserted:
        # Fallback: just append at the end.
        out_cells.extend([corpus_cell, corpus_run, embed_cell, embed_run])

    _renumber_cell_titles(out_cells)

    out_nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": out_cells,
    }

    out_nb_path.write_text(json.dumps(out_nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {out_nb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
