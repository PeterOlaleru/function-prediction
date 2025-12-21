from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "05_cafa_e2e.ipynb"


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    if len(cells) < 3:
        raise RuntimeError(f"Expected at least 3 cells, found {len(cells)}")

    cell3 = cells[2]
    src = cell3.get("source", [])
    if not isinstance(src, list):
        raise RuntimeError("Cell 3 source is not a list")

    out: list[str] = []

    inserted_kaggle_input_root = any(
        isinstance(s, str) and s.strip().startswith("KAGGLE_INPUT_ROOT =") for s in src
    )

    # Drop known-obsolete env var use + local variable that is never used.
    # Also de-duplicate repeated blocks introduced by previous patching/formatting.
    seen_exact: dict[str, int] = {}

    inserted_path_constants = False
    inserted_manifest_name = False

    # De-duplication for broken class blocks that sometimes get repeated.
    seen_class_method_sig: set[str] = set()
    skip_method_block = False

    def _count(s: str) -> int:
        seen_exact[s] = seen_exact.get(s, 0) + 1
        return seen_exact[s]

    i = 0
    while i < len(src):
        line = src[i]
        i += 1

        if not isinstance(line, str):
            continue

        # If we are skipping a duplicated method block inside KaggleCheckpointStore,
        # keep skipping until the next method definition or class de-dent.
        if skip_method_block:
            if line.startswith("    def ") or line.startswith("    @"):
                skip_method_block = False
            else:
                continue

        # Fix previously-concatenated lines from earlier patching.
        if "Path('/kaggle/input')if IS_KAGGLE:" in line:
            out.append("KAGGLE_INPUT_ROOT = Path('/kaggle/input')\n")
            out.append("if IS_KAGGLE:\n")
            inserted_kaggle_input_root = True
            continue

        if "artefacts root# ------------------------------------------" in line:
            out.append("# Local cache roots (ephemeral) + artefacts root\n")
            out.append("# ------------------------------------------\n")
            continue

        # Repair a known-broken block that sometimes appears after patching:
        #
        #   (publish_root / 'README.md').write_text(
        #       f'# {self.dataset_title}
        #   Auto-published...
        #   Latest stage...
        #   ',
        #       encoding='utf-8',
        #   )
        #
        # The f-string is split across physical lines, producing an unterminated string literal.
        if line.strip() == "(publish_root / 'README.md').write_text(":
            out.append(line)

            # If already in the fixed, single-line string-literal form, do nothing.
            if i < len(src) and isinstance(src[i], str) and "Latest stage" in src[i] and "\\n" in src[i]:
                continue

            # Skip the old/broken string lines up to (but not including) the encoding kwarg.
            while i < len(src):
                nxt = src[i]
                if isinstance(nxt, str) and nxt.strip().startswith("encoding="):
                    break
                i += 1

            out.append('            f"# {self.dataset_title}\\n"\n')
            out.append('            "Auto-published checkpoint dataset for CAFA6.\\n"\n')
            out.append('            f"Latest stage: {stage}\\n",\n')
            continue

        # Remove unused helper (dead code today).
        if line.startswith("def _copy_merge("):
            # Skip the entire function body until the next blank line.
            while i < len(src):
                nxt = src[i]
                i += 1
                if isinstance(nxt, str) and nxt.strip() == "":
                    break
            continue

        # Collapse repeated section separators.
        if line.strip() == "# ------------------------------------------":
            if out and out[-1].strip() == "# ------------------------------------------":
                continue

        # Remove CAFA_INPUT_ROOT wiring: we never set it anywhere.
        if "CAFA_INPUT_ROOT" in line:
            continue

        # Remove INPUT_ROOT assignments entirely (05 uses fixed locations /kaggle/input when needed).
        if re.match(r"^\s*INPUT_ROOT\s*=", line):
            continue

        # Remove redundant commentary about "single source of truth" that duplicates later text.
        if line.strip().startswith("# IMPORTANT: we always write locally first"):
            continue
        if line.strip().startswith("# but the Kaggle Dataset is the *single source of truth*"):
            continue

        # De-dupe the repeated local-cache header line.
        if line.strip() == "# Local cache roots (ephemeral) + published artefacts root":
            if _count(line.strip()) > 1:
                continue
            # Keep the first, but rename to avoid implying publish-root semantics.
            out.append("# Local cache roots (ephemeral) + artefacts root\n")
            continue

        # De-dupe the repeated cache-protection comment.
        if line.strip() == "# Keep caches OUT of WORK_ROOT so we never accidentally publish them.":
            if _count(line.strip()) > 1:
                continue

        # De-dupe KAGGLE_INPUT_ROOT declarations (can be introduced by earlier patching).
        if line.strip().startswith("KAGGLE_INPUT_ROOT ="):
            if _count("KAGGLE_INPUT_ROOT_DECL") > 1:
                continue
            if not line.endswith("\n"):
                line = line + "\n"

        # De-dupe repeated cache initialisation / env wiring (these appear twice today).
        if line.strip() == "CACHE_ROOT.mkdir(parents=True, exist_ok=True)":
            if _count(line.strip()) > 1:
                continue
        if line.strip().startswith("os.environ.setdefault('HF_HOME'"):
            if _count("HF_HOME") > 1:
                continue
        if line.strip().startswith("os.environ.setdefault('TRANSFORMERS_CACHE'"):
            if _count("TRANSFORMERS_CACHE") > 1:
                continue
        if line.strip().startswith("os.environ.setdefault('HF_HUB_CACHE'"):
            if _count("HF_HUB_CACHE") > 1:
                continue
        if line.strip().startswith("os.environ.setdefault('TORCH_HOME'"):
            if _count("TORCH_HOME") > 1:
                continue

        # Ensure we declare KAGGLE_INPUT_ROOT before its first use.
        if (not inserted_kaggle_input_root) and line.strip() == "if IS_KAGGLE:":
            out.append("KAGGLE_INPUT_ROOT = Path('/kaggle/input')\n")
            inserted_kaggle_input_root = True

        # Fix invalid Python string literal: replace('\', '/') must be written as replace('\\', '/').
        # (A single backslash between quotes is a syntax error in Python source.)
        line = line.replace("replace('\\', '/')", "replace('\\\\', '/')")

        # Rewrite references to INPUT_ROOT for Kaggle listing + checkpoint mount.
        line = line.replace("if INPUT_ROOT.exists():", "if KAGGLE_INPUT_ROOT.exists():")
        line = line.replace("os.walk(str(INPUT_ROOT))", "os.walk(str(KAGGLE_INPUT_ROOT))")
        line = line.replace("input_root=INPUT_ROOT,", "input_root=KAGGLE_INPUT_ROOT,")

        # Insert canonical competition-path constants immediately after DATASET_ROOT.
        # These are used throughout the notebook (including in Cell 3 diagnostics and Cell 4 parsing).
        if (not inserted_path_constants) and line.strip() == "DATASET_ROOT = ensure_competition_data(WORK_ROOT)":
            out.append(line)
            out.append("# Canonical competition paths (always under WORK_ROOT/cafa6_data)\n")
            out.append("PATH_IA = WORK_ROOT / 'IA.tsv'\n")
            out.append("PATH_SAMPLE_SUB = WORK_ROOT / 'sample_submission.tsv'\n")
            out.append("PATH_GO_OBO = WORK_ROOT / 'Train' / 'go-basic.obo'\n")
            out.append("PATH_TRAIN_FASTA = WORK_ROOT / 'Train' / 'train_sequences.fasta'\n")
            out.append("PATH_TRAIN_TERMS = WORK_ROOT / 'Train' / 'train_terms.tsv'\n")
            out.append("PATH_TRAIN_TAXON = WORK_ROOT / 'Train' / 'train_taxonomy.tsv'\n")
            out.append("PATH_TEST_FASTA = WORK_ROOT / 'Test' / 'testsuperset.fasta'\n")
            out.append("PATH_TEST_TAXON = WORK_ROOT / 'Test' / 'testsuperset-taxon-list.tsv'\n")
            inserted_path_constants = True
            continue

        # Unify WORK_ROOT with Cell 1's DATA_ROOT if present.
        if line.strip() == "WORK_ROOT = WORKING_ROOT / 'cafa6_data'":
            out.append("# If Cell 1 ran, reuse its DATA_ROOT so we don't fork paths.\n")
            out.append("if 'DATA_ROOT' in globals():\n")
            out.append("    WORK_ROOT = Path(DATA_ROOT)\n")
            out.append("    WORKING_ROOT = WORK_ROOT.parent\n")
            out.append("else:\n")
            out.append("    WORK_ROOT = WORKING_ROOT / 'cafa6_data'\n")
            continue

        # Fix manifest copy in the publish step (self.manifest_name was sometimes missing).
        if "manifest_path = self.work_root / self.manifest_name" in line:
            out.append("        manifest_path = MANIFEST_PATH\n")
            continue

        # Insert manifest_name field once so push() can reference it if needed.
        if (not inserted_manifest_name) and line.strip() == "is_kaggle: bool":
            out.append(line)
            out.append("    manifest_name: str = 'manifest.json'\n")
            inserted_manifest_name = True
            continue

        # De-duplicate repeated KaggleCheckpointStore methods (formatter/patch artefact).
        if line.startswith("    def _can_publish(") or line.startswith("    def maybe_push("):
            sig = line.strip()
            if sig in seen_class_method_sig:
                skip_method_block = True
                continue
            seen_class_method_sig.add(sig)

        out.append(line)

    cell3["source"] = out
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated: {NB_PATH}")


if __name__ == "__main__":
    main()
