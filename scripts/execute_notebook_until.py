import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def _cell_text(cell) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "\n".join(src)
    return str(src)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Execute a notebook top-to-bottom until a stop marker is encountered. "
            "Stops BEFORE executing the matching cell."
        )
    )
    ap.add_argument("notebook", type=Path)
    ap.add_argument(
        "--stop",
        type=str,
        required=True,
        help=(
            "Regex to match within a cell's source. Execution stops BEFORE the first matching cell."
        ),
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-cell timeout seconds (default: 3600).",
    )
    ap.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="Optional Jupyter kernel name to use (default: notebook metadata / nbclient default).",
    )
    ap.add_argument(
        "--save-executed",
        type=Path,
        default=None,
        help="Optional path to write an executed copy (for debugging).",
    )
    args = ap.parse_args()

    nb_path: Path = args.notebook
    if not nb_path.exists():
        raise FileNotFoundError(nb_path)

    stop_re = re.compile(args.stop)

    nb = nbformat.read(nb_path, as_version=4)

    # Find stop cell index
    stop_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        if stop_re.search(_cell_text(cell)):
            stop_idx = i
            break

    if stop_idx is None:
        raise SystemExit(f"Stop regex did not match any code cell: {args.stop!r}")

    # Execute only cells [0, stop_idx)
    sub_nb = nbformat.v4.new_notebook(metadata=nb.metadata)
    sub_nb.cells = nb.cells[:stop_idx]

    client_kwargs = dict(
        timeout=args.timeout,
        allow_errors=False,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    if args.kernel is not None:
        client_kwargs["kernel_name"] = args.kernel

    client = NotebookClient(sub_nb, **client_kwargs)

    print(f"Notebook: {nb_path}")
    print(f"Stop regex: {args.stop!r}")
    print(f"Stop cell (1-based): {stop_idx + 1}")
    print(f"Will execute code cells up to (excluding) that cell.")

    try:
        client.execute()
    except CellExecutionError as e:
        # nbclient uses 0-based cell index in message; we convert to 1-based.
        # We avoid dumping full outputs to keep logs readable.
        msg = str(e)
        print("\nERROR: Notebook execution failed before stop marker.")
        print(msg[:4000])
        return 2

    code_cells_executed = sum(1 for c in sub_nb.cells if c.get("cell_type") == "code")
    print(f"\nOK: executed {code_cells_executed} code cells before stop marker.")

    if args.save_executed is not None:
        out_path = args.save_executed
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(sub_nb, out_path)
        print(f"Wrote executed notebook copy: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
