from __future__ import annotations

from pathlib import Path


def main() -> int:
    nb_path = Path('notebooks/05_cafa_e2e.ipynb')
    text = nb_path.read_text(encoding='utf-8')

    # Preserve existing newline style to avoid churn.
    nl = '\r\n' if '\r\n' in text else '\n'
    lines = text.split(nl)

    # Find the first occurrence of the JSON source line storing:
    #   REQUIRED_CHECKPOINT_FILES = [
    start_idx = None
    for i, line in enumerate(lines):
        # In .ipynb JSON, each code line is stored as a JSON string containing a literal "\\n".
        # Example: "REQUIRED_CHECKPOINT_FILES = [\\n",
        if '"REQUIRED_CHECKPOINT_FILES = [\\n"' in line:
            start_idx = i
            break

    if start_idx is None:
        raise SystemExit('Could not find REQUIRED_CHECKPOINT_FILES in notebooks/05_cafa_e2e.ipynb')

    # Find the closing bracket python line (stored as a JSON string containing "]\\n").
    end_idx = None
    for j in range(start_idx + 1, min(start_idx + 100, len(lines))):
        if lines[j].strip() in {'"]\\n",', '"]\\n"', '"]",', '"]"'}:
            end_idx = j
            break

    if end_idx is None:
        raise SystemExit('Could not locate end of REQUIRED_CHECKPOINT_FILES list (expected a line containing "]").')

    # Patch is idempotent; we always normalise the list to the current required minimal set.

    indent = lines[start_idx].split('"REQUIRED_CHECKPOINT_FILES', 1)[0]

    replacement = [
        f'{indent}"REQUIRED_CHECKPOINT_FILES = [\\n",',
        f"{indent}\"    'parsed/train_seq.feather',\\n\",",
        f"{indent}\"    'parsed/term_counts.parquet',\\n\",",
        f"{indent}\"    'parsed/term_priors.parquet',\\n\",",
        f'{indent}"]\\n",',
        f'{indent}"if PATH_TEST_FASTA.exists():\\n",',
        f"{indent}\"    REQUIRED_CHECKPOINT_FILES += ['parsed/test_seq.feather', 'parsed/test_taxa.feather']\\n\",",
    ]

    new_lines = lines[:start_idx] + replacement + lines[end_idx + 1 :]
    nb_path.write_text(nl.join(new_lines), encoding='utf-8')
    print('Patched REQUIRED_CHECKPOINT_FILES in', nb_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
