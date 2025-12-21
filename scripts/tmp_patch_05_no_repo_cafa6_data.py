from __future__ import annotations

import json
import re
from pathlib import Path


NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "05_cafa_e2e.ipynb"


def _is_blank_line(s: str) -> bool:
    return s.strip() == ""


def _indent_level(s: str) -> int:
    expanded = s.replace("\t", "    ")
    return len(expanded) - len(expanded.lstrip(" "))


def _is_import(s: str) -> bool:
    st = s.lstrip()
    return st.startswith("import ") or st.startswith("from ")


def _is_def_like(s: str) -> bool:
    st = s.lstrip()
    return st.startswith("def ") or st.startswith("class ") or st.startswith("@")


def _compact_code_cell_source(source: list[str]) -> list[str]:
    # Matches scripts/compact_notebook_sources.py behaviour.
    lines = [str(x).replace("\r\n", "\n").replace("\r", "\n").rstrip(" \t\n") for x in source]

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

        if next_indent == 0 and _is_def_like(nxt) and prev_indent > 0:
            if out and out[-1] != "":
                out.append("")
            continue

        if prev_indent > 0 or next_indent > 0:
            continue

        if _is_import(prev) and not _is_import(nxt):
            if out and out[-1] != "":
                out.append("")
            continue

        if _is_def_like(nxt) or _is_def_like(prev):
            if out and out[-1] != "":
                out.append("")
            continue

        continue

    final: list[str] = []
    prev_blank = False
    for ln in out:
        is_blank = _is_blank_line(ln)
        if is_blank and prev_blank:
            continue
        final.append("") if is_blank else final.append(ln)
        prev_blank = is_blank

    while final and _is_blank_line(final[0]):
        final.pop(0)
    while final and _is_blank_line(final[-1]):
        final.pop()

    return final


def _compact_notebook_in_memory(nb: dict) -> None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source")
        if not isinstance(src, list):
            continue
        cell["source"] = _compact_code_cell_source([str(x) for x in src])


def _replace_local_env_block(lines: list[str]) -> list[str]:
    # Replace the "Environment: Local" branch with a no-repo version.
    try:
        loc_i = next(i for i, s in enumerate(lines) if "print('Environment: Local Detected')" in s)
    except StopIteration:
        return lines

    # Find the next section header after the local block.
    try:
        end_i = next(
            i
            for i in range(loc_i, len(lines))
            if i > loc_i and lines[i].startswith("# ------------------------------------------")
        )
    except StopIteration:
        return lines

    # We expect the line right before loc_i to be "else:"; replace from there.
    start_i = max(0, loc_i - 1)
    local_block = [
        "else:",
        "    print('Environment: Local Detected')",
        "    # No repo assumptions: treat current working dir as runtime root.",
        "    INPUT_ROOT = Path.cwd()",
        "    WORKING_ROOT = Path.cwd()",
    ]
    return lines[:start_i] + local_block + lines[end_i:]


def _replace_work_root_block(lines: list[str]) -> list[str]:
    # Force everything under WORKING_ROOT/cafa6_data.
    try:
        wr_i = next(i for i, s in enumerate(lines) if s.strip().startswith("WORK_ROOT = "))
    except StopIteration:
        return lines

    # Find end of old WORK_ROOT mkdir block.
    try:
        end_i = next(i for i in range(wr_i, len(lines)) if "(WORK_ROOT / 'external').mkdir" in lines[i]) + 1
    except StopIteration:
        return lines

    new_block = [
        "# ------------------------------------------",
        "# Local cache roots (ephemeral) + published artefacts root",
        "# ------------------------------------------",
        "# Single source of truth for this notebook: everything lives under cafa6_data/.",
        "WORK_ROOT = WORKING_ROOT / 'cafa6_data'",
        "WORK_ROOT.mkdir(parents=True, exist_ok=True)",
        "for _d in ['parsed', 'features', 'external', 'Train', 'Test']:",
        "    (WORK_ROOT / _d).mkdir(parents=True, exist_ok=True)",
        "",
        "# Keep caches OUT of WORK_ROOT so we never accidentally publish them.",
        "CACHE_ROOT = WORKING_ROOT / 'cache'",
        "CACHE_ROOT.mkdir(parents=True, exist_ok=True)",
        "os.environ.setdefault('HF_HOME', str(CACHE_ROOT / 'hf_home'))",
        "os.environ.setdefault('TRANSFORMERS_CACHE', str(CACHE_ROOT / 'hf_home'))",
        "os.environ.setdefault('HF_HUB_CACHE', str(CACHE_ROOT / 'hf_hub'))",
        "os.environ.setdefault('TORCH_HOME', str(CACHE_ROOT / 'torch_home'))",
        "",
        "# Runtime provenance guards: we must NEVER publish downloaded artefacts.",
        "RUN_START_TS = time.time()",
        "DOWNLOADED_PATHS: set[Path] = set()",
        "def _mark_downloaded(p: Path) -> None:",
        "    DOWNLOADED_PATHS.add(Path(p).resolve())",
    ]

    return lines[:wr_i] + new_block + lines[end_i:]


def _ensure_provenance_guards(lines: list[str]) -> list[str]:
    joined = "\n".join(lines)
    if "RUN_START_TS" in joined and "DOWNLOADED_PATHS" in joined and "def _mark_downloaded" in joined:
        # Still remove CAFA_CACHE_ROOT optional override if present.
        return [s for s in lines if "CAFA_CACHE_ROOT" not in s]

    # Insert right after the first cache setup block.
    insert_after = None
    for i, s in enumerate(lines):
        if "os.environ.setdefault('TORCH_HOME'" in s:
            insert_after = i
            break

    if insert_after is None:
        # Fallback: after WORK_ROOT is created.
        for i, s in enumerate(lines):
            if s.strip().startswith("WORK_ROOT ="):
                insert_after = i
                break

    if insert_after is None:
        insert_after = 0

    block = [
        "",
        "# Runtime provenance guards: never publish downloaded artefacts.",
        "RUN_START_TS = time.time()",
        "DOWNLOADED_PATHS: set[Path] = set()",
        "def _mark_downloaded(p: Path) -> None:",
        "    DOWNLOADED_PATHS.add(Path(p).resolve())",
        "",
    ]

    out = lines[: insert_after + 1] + block + lines[insert_after + 1 :]
    out = [s for s in out if "CAFA_CACHE_ROOT" not in s]
    return out


def _replace_competition_discovery_block(lines: list[str]) -> list[str]:
    start_prefix = "# Dataset Discovery (competition data)"
    end_prefix = "# Checkpoint store (Kaggle Dataset = single source of truth)"

    try:
        s_i = next(i for i, s in enumerate(lines) if s.strip().startswith(start_prefix))
        e_i = next(i for i, s in enumerate(lines) if s.strip().startswith(end_prefix))
    except StopIteration:
        return lines

    if e_i <= s_i:
        return lines

    replacement = [
        start_prefix,
        "",
        "DATASET_SLUG = 'cafa-6-protein-function-prediction'",
        "",
        "# Required competition files (MANDATORY):",
        "REQUIRED_COMP_FILES = [",
        "    'IA.tsv',",
        "    'sample_submission.tsv',",
        "    'Train/go-basic.obo',",
        "    'Train/train_sequences.fasta',",
        "    'Train/train_terms.tsv',",
        "    'Train/train_taxonomy.tsv',",
        "    'Test/testsuperset.fasta',",
        "    'Test/testsuperset-taxon-list.tsv',",
        "]",
        "",
        "def _ensure_kaggle_cli() -> None:",
        "    try:",
        "        subprocess.run(['kaggle', '--version'], check=True, capture_output=True, text=True)",
        "    except Exception:",
        "        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'kaggle'])",
        "        subprocess.run(['kaggle', '--version'], check=True)",
        "",
        "def _kaggle_env(require: bool = True) -> dict[str, str]:",
        "    env = os.environ.copy()",
        "    if require and (not env.get('KAGGLE_USERNAME') or not env.get('KAGGLE_KEY')):",
        "        raise RuntimeError('Missing Kaggle API auth: set KAGGLE_USERNAME and KAGGLE_KEY as secrets/env vars.')",
        "    return env",
        "",
        "def _download_comp_file(rel_path: str, target_root: Path) -> None:",
        "    \"\"\"Download one competition file into target_root, preserving folders.\"\"\"",
        "    rel_path = rel_path.replace('\\\\', '/')",
        "    out_path = target_root / rel_path",
        "    out_path.parent.mkdir(parents=True, exist_ok=True)",
        "    if out_path.exists():",
        "        return",
        "    _ensure_kaggle_cli()",
        "    env = _kaggle_env(require=True)",
        "    tmp = target_root / '_tmp_download'",
        "    tmp.mkdir(parents=True, exist_ok=True)",
        "    cmd = ['kaggle', 'competitions', 'download', '-c', DATASET_SLUG, '-f', rel_path, '-p', str(tmp)]",
        "    print('+', ' '.join(cmd))",
        "    subprocess.run(cmd, check=True, env=env)",
        "    name = Path(rel_path).name",
        "    zip_path = tmp / f'{name}.zip'",
        "    if zip_path.exists():",
        "        with zipfile.ZipFile(zip_path, 'r') as zf:",
        "            zf.extractall(tmp)",
        "        zip_path.unlink()",
        "    cand1 = tmp / rel_path",
        "    cand2 = tmp / name",
        "    src = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)",
        "    if src is None:",
        "        raise FileNotFoundError(f'Downloaded file not found after unzip: {rel_path}')",
        "    out_path.parent.mkdir(parents=True, exist_ok=True)",
        "    shutil.move(str(src), str(out_path))",
        "    _mark_downloaded(out_path)",
        "",
        "def ensure_competition_data(data_root: Path) -> Path:",
        "    \"\"\"Ensures competition data exists under cafa6_data/. Returns dataset root.\"\"\"",
        "    data_root = Path(data_root)",
        "    # Kaggle: copy from mounted input if available.",
        "    if IS_KAGGLE and Path('/kaggle/input').exists():",
        "        mounted = Path('/kaggle/input') / DATASET_SLUG",
        "        if mounted.exists():",
        "            for rel in REQUIRED_COMP_FILES:",
        "                dst = data_root / rel",
        "                if dst.exists():",
        "                    continue",
        "                src = mounted / rel",
        "                dst.parent.mkdir(parents=True, exist_ok=True)",
        "                if not src.exists():",
        "                    raise FileNotFoundError(f'Missing in mounted Kaggle input: {src}')",
        "                shutil.copy2(src, dst)",
        "                _mark_downloaded(dst)",
        "            return data_root",
        "    # Colab/Local: download file-by-file via Kaggle API.",
        "    for rel in REQUIRED_COMP_FILES:",
        "        _download_comp_file(rel, data_root)",
        "    return data_root",
        "",
        "# Canonical dataset root for the entire notebook",
        "DATASET_ROOT = ensure_competition_data(WORK_ROOT)",
        "print('DATASET_ROOT:', DATASET_ROOT.resolve())",
    ]

    return lines[:s_i] + replacement + lines[e_i:]


def _remove_force_rebuild_toggle(lines: list[str]) -> list[str]:
    # Remove CAFA_FORCE_REBUILD secret/env machinery.
    try:
        rb_start = next(i for i, s in enumerate(lines) if "Resolve CAFA_FORCE_REBUILD" in s)
        rb_end = next(i for i in range(rb_start, len(lines)) if "CHECKPOINT_DATASET_ID" in lines[i])
    except StopIteration:
        return lines

    return lines[:rb_start] + ["# No optional rebuild toggles in this notebook."] + lines[rb_end:]


def _remove_cafa_force_rebuild_globally(all_cells: list[dict]) -> None:
    """Remove CAFA_FORCE_REBUILD usage and make logic purely idempotent (skip-if-exists).

    This preserves research semantics whilst eliminating env-driven optional behaviour.
    """
    for cell in all_cells:
        if cell.get("cell_type") != "code":
            continue
        src = list(cell.get("source", []))
        out: list[str] = []
        for line in src:
            if "CAFA_FORCE_REBUILD" in line:
                # Drop the env var read + any user-facing messaging about setting it.
                continue
            if line.strip().startswith("FORCE_REBUILD ="):
                continue
            # Strip any lingering messaging fragments.
            line = line.replace(" (set CAFA_FORCE_REBUILD=1 to rebuild).", ".")
            line = line.replace(" (set CAFA_FORCE_REBUILD=1 to refit).", ".")
            line = line.replace(" (set CAFA_FORCE_REBUILD=1 to force).", ".")
            # Remove conditional guards that referenced the env var.
            line = line.replace(" and CAFA_FORCE_REBUILD != '1'", "")
            line = line.replace(" and not FORCE_REBUILD", "")
            out.append(line)

        cell["source"] = out


def _make_pull_push_non_optional(lines: list[str]) -> list[str]:
    out = []
    for s in lines:
        if s.strip().startswith("CHECKPOINT_PULL ="):
            out.append("CHECKPOINT_PULL = True")
        elif s.strip().startswith("CHECKPOINT_PUSH ="):
            out.append("CHECKPOINT_PUSH = True")
            # Also make any "push existing" behaviour non-optional.
            if not any(x.strip() == "PUSH_EXISTING_CHECKPOINTS = True" for x in out):
                out.append("PUSH_EXISTING_CHECKPOINTS = True")
        else:
            out.append(s)
    return out


def _ensure_required_checkpoint_files_list(lines: list[str]) -> list[str]:
    joined = "\n".join(lines)
    if "REQUIRED_CHECKPOINT_FILES" in joined:
        return lines

    try:
        store_i = next(i for i, s in enumerate(lines) if s.strip().startswith("STORE = KaggleCheckpointStore"))
    except StopIteration:
        return lines

    insert = [
        "",
        "# Checkpoint artefacts required for this notebook (MANDATORY, pulled file-by-file)",
        "REQUIRED_CHECKPOINT_FILES = [",
        "    'features/top_terms_13500.json',",
        "    'parsed/train_terms.parquet',",
        "    'parsed/train_seq.feather',",
        "    'parsed/train_taxa.feather',",
        "]",
        "",
    ]
    return lines[:store_i] + insert + lines[store_i:]


def _call_store_pull_with_required_list(lines: list[str]) -> list[str]:
    out = []
    for s in lines:
        if s.strip() == "STORE.pull()":
            out.append("STORE.pull(required_files=REQUIRED_CHECKPOINT_FILES)")
        else:
            out.append(s)
    return out


def _remove_checkpoint_disable_messaging(lines: list[str]) -> list[str]:
    # Remove any "disabled" branches/messages that reference env toggles.
    banned_substrings = [
        "Checkpoint pull disabled",
        "Checkpoint push disabled",
        "CAFA_CHECKPOINT_PULL",
        "CAFA_CHECKPOINT_PUSH",
    ]
    out: list[str] = []
    for s in lines:
        if any(b in s for b in banned_substrings):
            continue
        out.append(s)
    return out


def _dedupe_exact(lines: list[str], exact: str) -> list[str]:
    out: list[str] = []
    seen = False
    for s in lines:
        if s.strip() == exact.strip():
            if seen:
                continue
            seen = True
            out.append(exact)
            continue
        out.append(s)
    return out


def _patch_checkpoint_store_pull_file_by_file(lines: list[str]) -> list[str]:
    """Rewrite KaggleCheckpointStore.pull to accept required_files and download/copy file-by-file."""
    text = "\n".join(lines)

    pattern = re.compile(
        r"(^\s{4}def pull\(self\) -> None:\s*$)([\s\S]*?)(^\s{4}def push\(self, stage: str, required_paths: list\[Path\], note: str = ''\) -> None:\s*$)",
        flags=re.MULTILINE,
    )

    m = pattern.search(text)
    if not m:
        return lines

    new_pull = "\n".join(
        [
            "    def pull(self, required_files: list[str] | None = None) -> None:",
            "        \"\"\"Pulls checkpoint artefacts into WORK_ROOT (cafa6_data).",
            "",
            "        Contract: required_files is mandatory for this notebook and is fetched file-by-file",
            "        (either from mounted Kaggle input, or via Kaggle API).",
            "        \"\"\"",
            "        if not self.pull_enabled:",
            "            return",
            "",
            "        checkpoint_required = True",
            "        required_files = list(required_files or [])",
            "        if not required_files:",
            "            raise ValueError('STORE.pull() requires required_files for this notebook.')",
            "",
            "        if not self.dataset_id:",
            "            raise ValueError('Missing CAFA_CHECKPOINT_DATASET_ID=<user>/<slug>; cannot resume.')",
            "",
            "        # Fast path: Kaggle-mounted dataset input",
            "        if self.mount_dir is not None:",
            "            print(f'Pulling checkpoints (file-by-file) from Kaggle mounted dataset: {self.mount_dir}')",
            "            for rel in required_files:",
            "                rel = str(rel).replace('\\\\', '/')",
            "                src = self.mount_dir / rel",
            "                dst = self.work_root / rel",
            "                dst.parent.mkdir(parents=True, exist_ok=True)",
            "                if not src.exists():",
            "                    raise FileNotFoundError(f'Missing required checkpoint file in mounted dataset: {src}')",
            "                shutil.copy2(src, dst)",
            "                _mark_downloaded(dst)",
            "            _maybe_unpack_dir_mode_zips(self.work_root)",
            "            return",
            "",
            "        print(f'Downloading checkpoints (file-by-file) from Kaggle API: {self.dataset_id}')",
            "        _ensure_kaggle_cli()",
            "        env = _kaggle_env(require=checkpoint_required)",
            "        tmp = self.work_root / '_tmp_kaggle_download'",
            "        if tmp.exists():",
            "            shutil.rmtree(tmp)",
            "        tmp.mkdir(parents=True, exist_ok=True)",
            "",
            "        for rel in required_files:",
            "            rel = str(rel).replace('\\\\', '/')",
            "            cmd = ['kaggle', 'datasets', 'download', '-d', self.dataset_id, '-f', rel, '-p', str(tmp)]",
            "            print('+', ' '.join(cmd))",
            "            subprocess.run(cmd, check=True, env=env)",
            "",
            "            name = Path(rel).name",
            "            zip_path = tmp / f'{name}.zip'",
            "            if zip_path.exists():",
            "                with zipfile.ZipFile(zip_path, 'r') as zf:",
            "                    zf.extractall(tmp)",
            "                zip_path.unlink()",
            "",
            "            cand1 = tmp / rel",
            "            cand2 = tmp / name",
            "            src = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)",
            "            if src is None:",
            "                raise FileNotFoundError(f'Downloaded checkpoint file not found after unzip: {rel}')",
            "",
            "            dst = self.work_root / rel",
            "            dst.parent.mkdir(parents=True, exist_ok=True)",
            "            shutil.move(str(src), str(dst))",
            "            _mark_downloaded(dst)",
            "",
            "        _maybe_unpack_dir_mode_zips(self.work_root)",
            "        shutil.rmtree(tmp)",
            "",
        ]
    )

    text = pattern.sub(new_pull + "\n\n" + m.group(3), text, count=1)
    return text.split("\n")


def _ensure_pull_marks_downloaded(lines: list[str]) -> list[str]:
    """Idempotently inject _mark_downloaded(dst) into the pull method body."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)

        # Mounted copy path
        if line.strip() == "shutil.copy2(src, dst)":
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if "_mark_downloaded(dst)" not in nxt:
                out.append("                _mark_downloaded(dst)")

        # API download path
        if line.strip() == "shutil.move(str(src), str(dst))":
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if "_mark_downloaded(dst)" not in nxt:
                out.append("            _mark_downloaded(dst)")

        i += 1

    return out


def _patch_checkpoint_store_push_staging_only(lines: list[str]) -> list[str]:
    """Rewrite KaggleCheckpointStore.push to publish ONLY staged freshly-built artefacts.

    Hard rules:
    - Never publish anything recorded as downloaded/pulled.
    - Only publish artefacts with mtime >= RUN_START_TS.
    - Publish from an isolated staging dir so Kaggle CLI never zips WORK_ROOT.
    """
    text = "\n".join(lines)
    pattern = re.compile(
        r"(^\s{4}def push\(self, stage: str, required_paths: list\[Path\], note: str = ''\) -> None:\s*$)([\s\S]*?)(^\s*# Checkpoint artefacts required for this notebook)",
        flags=re.MULTILINE,
    )
    m = pattern.search(text)
    if not m:
        return lines

    new_push = "\n".join(
        [
            "    def _can_publish(self, required_paths: list[Path]) -> tuple[bool, str]:",
            "        # 1) Must exist",
            "        missing = [Path(p) for p in required_paths if not Path(p).exists()]",
            "        if missing:",
            "            return False, 'missing required artefacts: ' + ', '.join([str(m) for m in missing])",
            "",
            "        # 2) Must not be downloaded/pulled",
            "        downloaded = [Path(p) for p in required_paths if Path(p).resolve() in DOWNLOADED_PATHS]",
            "        if downloaded:",
            "            return False, 'refusing to publish downloaded artefacts: ' + ', '.join([str(d) for d in downloaded])",
            "",
            "        # 3) Must be freshly built in this runtime",
            "        threshold = float(RUN_START_TS) - 1.0",
            "        not_fresh = [Path(p) for p in required_paths if Path(p).stat().st_mtime < threshold]",
            "        if not_fresh:",
            "            return False, 'refusing to publish non-fresh artefacts (not built this run): ' + ', '.join([str(n) for n in not_fresh])",
            "",
            "        return True, ''",
            "",
            "    def maybe_push(self, stage: str, required_paths: list[Path], note: str = '') -> None:",
            "        if not self.push_enabled:",
            "            return",
            "        ok, reason = self._can_publish(required_paths)",
            "        if not ok:",
            "            print(f'Skipping checkpoint publish for {stage}: {reason}')",
            "            return",
            "        self.push(stage, required_paths, note=note)",
            "",
            "    def push(self, stage: str, required_paths: list[Path], note: str = '') -> None:",
            "        if not self.push_enabled:",
            "            return",
            "        if not self.dataset_id:",
            "            raise ValueError('Missing CAFA_CHECKPOINT_DATASET_ID=<user>/<slug>; cannot checkpoint.')",
            "",
            "        ok, reason = self._can_publish(required_paths)",
            "        if not ok:",
            "            raise RuntimeError(f'Checkpoint publish blocked for {stage}: {reason}')",
            "",
            "        # Optional skip: identical stage signature",
            "        m = _load_manifest()",
            "        existing = (m.get('stages', {}) or {}).get(stage) if isinstance(m, dict) else None",
            "        if isinstance(existing, dict):",
            "            prev_files = existing.get('files', [])",
            "            if isinstance(prev_files, list):",
            "                prev_sig = sorted([{'path': f.get('path'), 'bytes': f.get('bytes')} for f in prev_files if isinstance(f, dict)], key=lambda x: str(x.get('path')))",
            "                cur_sig = _stage_files_signature(required_paths)",
            "                if prev_sig == cur_sig:",
            "                    print(f'Checkpoint stage {stage} unchanged; skipping publish')",
            "                    return",
            "",
            "        _update_manifest(stage, required_paths, note=note)",
            "",
            "        # Publish ONLY what we just built: stage to a clean directory",
            "        publish_root = self.work_root / '_publish_tmp' / stage",
            "        if publish_root.exists():",
            "            shutil.rmtree(publish_root)",
            "        publish_root.mkdir(parents=True, exist_ok=True)",
            "",
            "        work_root_resolved = self.work_root.resolve()",
            "        for p in required_paths:",
            "            p = Path(p).resolve()",
            "            try:",
            "                rel = p.relative_to(work_root_resolved)",
            "            except Exception:",
            "                raise ValueError(f'All checkpoint artefacts must live under WORK_ROOT. Got: {p}')",
            "            dst = publish_root / rel",
            "            dst.parent.mkdir(parents=True, exist_ok=True)",
            "            shutil.copy2(p, dst)",
            "",
            "        # Include the manifest (freshly written) in the publication",
            "        manifest_path = self.work_root / self.manifest_name",
            "        if manifest_path.exists():",
            "            shutil.copy2(manifest_path, publish_root / self.manifest_name)",
            "",
            "        (publish_root / 'dataset-metadata.json').write_text(",
            "            json.dumps({'title': self.dataset_title, 'id': self.dataset_id, 'licenses': [{'name': 'CC0-1.0'}]}, indent=2),",
            "            encoding='utf-8',",
            "        )",
            "        (publish_root / 'README.md').write_text(",
            "            f'# {self.dataset_title}\\n\\nAuto-published checkpoint dataset for CAFA6.\\n\\nLatest stage: {stage}\\n',",
            "            encoding='utf-8',",
            "        )",
            "",
            "        _ensure_kaggle_cli()",
            "        env = _kaggle_env(require=True)",
            "        msg = f'{stage}: {note}'.strip() if note else stage",
            "",
            "        p = subprocess.run(['kaggle', 'datasets', 'version', '-p', str(publish_root), '--dir-mode', 'zip', '-m', msg], text=True, capture_output=True, env=env)",
            "        if p.returncode != 0:",
            "            p2 = subprocess.run(['kaggle', 'datasets', 'create', '-p', str(publish_root), '--dir-mode', 'zip'], text=True, capture_output=True, env=env)",
            "            if p2.returncode != 0:",
            "                print(p.stdout); print(p.stderr); print(p2.stdout); print(p2.stderr)",
            "                raise RuntimeError('Kaggle dataset publish failed. See logs above.')",
            "            print(p2.stdout); print(p2.stderr)",
            "        else:",
            "            print(p.stdout); print(p.stderr)",
            "            print('Published new checkpoint dataset version:', self.dataset_id)",
            "",
        ]
    )

    text = pattern.sub(new_push + "\n\n" + m.group(3), text, count=1)
    return text.split("\n")


def _replace_store_push_calls_with_maybe_push(all_cells: list[dict]) -> None:
    for cell in all_cells:
        if cell.get("cell_type") != "code":
            continue
        src = list(cell.get("source", []))
        new = []
        for line in src:
            new.append(line.replace("STORE.push(", "STORE.maybe_push("))
        cell["source"] = new


def _make_push_existing_checkpoints_mandatory(all_cells: list[dict]) -> None:
    # Remove env-controlled toggles for pushing pre-existing artefacts.
    # Policy: always push when the code reaches a push stage.
    banned = [
        "CAFA_CHECKPOINT_PUSH_EXISTING",
        "Checkpoint-push for existing artefacts is opt-in",
        "Skipping checkpoint push for existing",
    ]
    for cell in all_cells:
        if cell.get("cell_type") != "code":
            continue
        src = list(cell.get("source", []))
        new: list[str] = []
        for line in src:
            if any(b in line for b in banned):
                continue
            if line.strip().startswith("PUSH_EXISTING_CHECKPOINTS ="):
                new.append("PUSH_EXISTING_CHECKPOINTS = True")
                continue
            new.append(line)
        cell["source"] = new


def _fix_remaining_repo_wording_and_paths(all_cells: list[dict]) -> None:
    # Clean up any leftover "repo root" assumptions.
    for cell in all_cells:
        if cell.get("cell_type") != "code":
            continue
        src = list(cell.get("source", []))
        new: list[str] = []
        for line in src:
            if "IA.tsv is required in repo root" in line:
                new.append("    # IA.tsv is required under DATASET_ROOT; still guard for robustness")
                continue
            if line.strip() == "ia_path = WORK_ROOT.parent / 'IA.tsv'":
                new.append("    ia_path = WORK_ROOT / 'IA.tsv'")
                continue
            # Drop the old parent fallback branch entirely.
            if line.strip() == "if not ia_path.exists():":
                # Only drop the immediate fallback that rewired ia_path to WORK_ROOT.
                # We detect it by looking ahead in the original stream.
                pass
            new.append(line)

        # Remove the specific 2-line fallback:
        #   if not ia_path.exists():
        #       ia_path = WORK_ROOT / 'IA.tsv'
        out: list[str] = []
        i = 0
        while i < len(new):
            if (
                new[i].strip() == "if not ia_path.exists():"
                and i + 1 < len(new)
                and new[i + 1].strip() == "ia_path = WORK_ROOT / 'IA.tsv'"
            ):
                i += 2
                continue
            out.append(new[i])
            i += 1

        cell["source"] = out


def _remove_checkpoint_force_push_toggle(all_cells: list[dict]) -> None:
    # Remove env-controlled force-push behaviour; keep deterministic skip-unchanged.
    for cell in all_cells:
        if cell.get("cell_type") != "code":
            continue
        src = list(cell.get("source", []))
        new: list[str] = []
        for line in src:
            if line.strip().startswith("force_push = os.environ.get('CAFA_CHECKPOINT_FORCE_PUSH'"):
                new.append("        force_push = False")
                continue
            if "CAFA_CHECKPOINT_FORCE_PUSH" in line:
                # Drop user-facing "set env var" messaging.
                continue
            new.append(line)
        cell["source"] = new


def patch_notebook() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Cell 01: no repo, set DATA_ROOT
    cells[0]["source"] = [
        "# CELL 01 - Setup (NO REPO)",
        "import os",
        "from pathlib import Path",
        "",
        "# Always run from a simple writable location; never cd into a repo.",
        "if os.path.exists('/content'):",
        "    os.chdir('/content')",
        "RUNTIME_ROOT = Path.cwd()",
        "DATA_ROOT = (RUNTIME_ROOT / 'cafa6_data')",
        "DATA_ROOT.mkdir(parents=True, exist_ok=True)",
        "print(f'CWD: {Path.cwd()}')",
        "print(f'DATA_ROOT: {DATA_ROOT.resolve()}')",
    ]

    # Cell 02: mandatory pip installs
    cells[1]["source"] = [
        "# CELL 02 - Install dependencies (mandatory, early)",
        "import importlib.util",
        "import os",
        "import subprocess",
        "import sys",
        "",
        "def _detect_kaggle() -> bool:",
        "    return bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or os.environ.get('KAGGLE_URL_BASE') or os.environ.get('KAGGLE_DATA_PROXY_URL'))",
        "def _detect_colab() -> bool:",
        "    return bool(os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_GPU') or os.environ.get('COLAB_TPU_ADDR'))",
        "IS_KAGGLE = _detect_kaggle()",
        "IS_COLAB = (not IS_KAGGLE) and _detect_colab()",
        "if IS_KAGGLE:",
        "    print('Environment: Kaggle Detected')",
        "elif IS_COLAB:",
        "    print('Environment: Colab Detected')",
        "else:",
        "    print('Environment: Local Detected')",
        "",
        "def _pip_install(pkgs: list[str]) -> None:",
        "    if not pkgs:",
        "        return",
        "    print('+', sys.executable, '-m', 'pip', 'install', *pkgs)",
        "    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', *pkgs])",
        "",
        "# Kaggle has a heavily preinstalled environment; avoid upgrading core packages by default.",
        "# We still guarantee requirements are present by installing missing ones.",
        "REQUIRED = {",
        "    # Core",
        "    'pandas': 'pandas',",
        "    'numpy': 'numpy',",
        "    'scipy': 'scipy',",
        "    'pyarrow': 'pyarrow',",
        "",
        "    # ML",
        "    'scikit-learn': 'sklearn',",
        "    'torch': 'torch',",
        "    'transformers': 'transformers',",
        "    'py-boost': 'py_boost',",
        "",
        "    # Bio / graph",
        "    'biopython': 'Bio',",
        "    'obonet': 'obonet',",
        "    'networkx': 'networkx',",
        "",
        "    # Visualisation",
        "    'matplotlib': 'matplotlib',",
        "    'seaborn': 'seaborn',",
        "",
        "    # Notebook tooling",
        "    'jupyter': 'jupyter',",
        "    'ipykernel': 'ipykernel',",
        "    'nbformat': 'nbformat',",
        "    'nbclient': 'nbclient',",
        "",
        "    # Utils",
        "    'tqdm': 'tqdm',",
        "    'requests': 'requests',",
        "    'urllib3': 'urllib3',",
        "    'joblib': 'joblib',",
        "    'psutil': 'psutil',",
        "    'fastparquet': 'fastparquet',",
        "    'pyyaml': 'yaml',",
        "    'kaggle': 'kaggle',",
        "}",
        "",
        "if IS_KAGGLE:",
        "    missing = [pkg for pkg, mod in REQUIRED.items() if importlib.util.find_spec(mod) is None]",
        "    if missing:",
        "        _pip_install(missing)",
        "    else:",
        "        print('Kaggle: skipping pip install (already satisfied).')",
        "else:",
        "    # Colab/Local: enforce dependencies up-front.",
        "    _pip_install(list(REQUIRED.keys()))",
    ]

    # Cell 03: patch key blocks, keep rest intact
    lines = list(cells[2]["source"])
    lines = _replace_local_env_block(lines)
    lines = _replace_work_root_block(lines)
    lines = _ensure_provenance_guards(lines)
    lines = _replace_competition_discovery_block(lines)
    lines = _remove_force_rebuild_toggle(lines)
    lines = _make_pull_push_non_optional(lines)
    lines = _ensure_required_checkpoint_files_list(lines)
    lines = _call_store_pull_with_required_list(lines)
    lines = _remove_checkpoint_disable_messaging(lines)
    lines = _dedupe_exact(lines, "PUSH_EXISTING_CHECKPOINTS = True")
    lines = _patch_checkpoint_store_pull_file_by_file(lines)
    lines = _ensure_pull_marks_downloaded(lines)
    lines = _patch_checkpoint_store_push_staging_only(lines)

    cells[2]["source"] = lines

    # Enforce mandatory checkpoint pushing behaviour across the whole notebook.
    _make_push_existing_checkpoints_mandatory(cells)

    # Remove CAFA_FORCE_REBUILD throughout, and avoid pushing anything "existing".
    _remove_cafa_force_rebuild_globally(cells)

    # Replace STORE.push(...) calls so the notebook never errors when artefacts are pre-existing.
    _replace_store_push_calls_with_maybe_push(cells)

    # Remove remaining "repo" assumptions in helper cells.
    _fix_remaining_repo_wording_and_paths(cells)

    # Remove env-controlled checkpoint force push toggles.
    _remove_checkpoint_force_push_toggle(cells)

    # Finally: tighten code-cell whitespace so the notebook stays readable.
    _compact_notebook_in_memory(nb)

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    patch_notebook()
    print(f"Patched: {NB_PATH}")
