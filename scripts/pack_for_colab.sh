#!/usr/bin/env bash
# Iron Law #5: Cleanly package DCPA source & configs into ./dcpa_colab_bundle.zip (< 500 KB)
# and verify notebooks/Colab_DCPA_Train.ipynb (valid JSON + ast.parse on every code cell).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/dcpa-train-py/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/dcpa-train-py/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  PYTHON_BIN="$(command -v python)"
fi

BUNDLE_PATH="${REPO_ROOT}/dcpa_colab_bundle.zip"
NOTEBOOK_PATH="${REPO_ROOT}/notebooks/Colab_DCPA_Train.ipynb"
MAX_BYTES=512000  # 500 KB strict ceiling

echo "[1/3] Validating ${NOTEBOOK_PATH} (JSON schema + ast.parse across all code cells)..."
"${PYTHON_BIN}" - "${NOTEBOOK_PATH}" <<'PYEOF'
import ast
import json
import sys
from pathlib import Path

nb_path = Path(sys.argv[1])
if not nb_path.exists():
    raise SystemExit(f"ERROR: Notebook not found: {nb_path}")

data = json.loads(nb_path.read_text(encoding="utf-8"))
if data.get("nbformat") != 4:
    raise SystemExit(f"ERROR: Expected Jupyter nbformat == 4, got {data.get('nbformat')!r}")

cells = data.get("cells")
if not isinstance(cells, list) or len(cells) != 5:
    raise SystemExit(f"ERROR: Expected exactly 5 cells, got {len(cells) if isinstance(cells, list) else type(cells)}")

code_cells = 0
for idx, cell in enumerate(cells, start=1):
    if cell.get("cell_type") == "code":
        code_cells += 1
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        try:
            ast.parse(src, filename=f"{nb_path.name}#cell_{idx}")
        except SyntaxError as exc:
            raise SystemExit(f"ERROR: ast.parse failed on Cell {idx}: {exc}") from exc

print(f"  [OK] Valid Jupyter v4 JSON ({len(cells)} cells, {code_cells} code cells passed ast.parse).")
PYEOF

echo "[2/3] Packaging pure source & config bundle -> ${BUNDLE_PATH}..."
rm -f "${BUNDLE_PATH}"

"${PYTHON_BIN}" - "${REPO_ROOT}" "${BUNDLE_PATH}" "${MAX_BYTES}" <<'PYEOF'
import os
import sys
import zipfile
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
bundle_path = Path(sys.argv[2]).resolve()
max_bytes = int(sys.argv[3])

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "target",
    "artifacts",
    "checkpoints",
    "dist",
    "build",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
}
EXCLUDED_DIR_SUFFIXES = (".egg-info",)
EXCLUDED_REL_PREFIXES = (
    "data/corpus",
    "data/smoke_corpus",
    "data/stage",
)
EXCLUDED_EXTS = {
    ".bin",
    ".idx",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".a",
    ".o",
    ".tar",
    ".gz",
    ".zip",
}
EXCLUDED_FILES = {
    ".DS_Store",
    "Cargo.lock",
    bundle_path.name,
}

included_files: list[Path] = []
for root, dirs, files in os.walk(repo_root):
    root_path = Path(root)
    rel_root = root_path.relative_to(repo_root)
    rel_root_posix = "" if str(rel_root) == "." else rel_root.as_posix()

    # Prune excluded directories in-place
    kept_dirs = []
    for d in sorted(dirs):
        if d in EXCLUDED_DIRS or any(d.endswith(suf) for suf in EXCLUDED_DIR_SUFFIXES):
            continue
        rel_d = f"{rel_root_posix}/{d}" if rel_root_posix else d
        if any(rel_d == pref or rel_d.startswith(pref + "/") or rel_d.startswith(pref) for pref in EXCLUDED_REL_PREFIXES):
            continue
        kept_dirs.append(d)
    dirs[:] = kept_dirs

    for fname in sorted(files):
        if fname in EXCLUDED_FILES:
            continue
        fpath = root_path / fname
        if fpath.suffix.lower() in EXCLUDED_EXTS:
            continue
        included_files.append(fpath)

with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for fpath in included_files:
        arcname = fpath.relative_to(repo_root).as_posix()
        zf.write(fpath, arcname=arcname)

size_bytes = bundle_path.stat().st_size
size_kb = size_bytes / 1024.0
print(f"  [OK] Packed {len(included_files)} files -> {bundle_path.name} ({size_kb:.1f} KB / {size_bytes} bytes)")

if size_bytes > max_bytes:
    raise SystemExit(
        f"ERROR: Bundle size {size_kb:.1f} KB exceeds 500 KB limit ({max_bytes} bytes)!"
    )
PYEOF

echo "[3/3] Verifying zip integrity & exclusion hygiene..."
"${PYTHON_BIN}" - "${BUNDLE_PATH}" <<'PYEOF'
import sys
import zipfile
from pathlib import Path

bundle_path = Path(sys.argv[1])
with zipfile.ZipFile(bundle_path, "r") as zf:
    bad = zf.testzip()
    if bad is not None:
        raise SystemExit(f"ERROR: Corrupt entry in zip: {bad}")
    names = zf.namelist()

forbidden_tokens = (".git/", "__pycache__/", "artifacts/", ".venv/", "target/")
forbidden_exts = (".bin", ".pt", ".safetensors", ".pyc")
for name in names:
    if any(tok in name for tok in forbidden_tokens) or name.endswith(forbidden_exts):
        raise SystemExit(f"ERROR: Forbidden artifact leaked into bundle: {name}")

required = [
    "utils/device.py",
    "data/build_corpus.py",
    "train/colab_drive_sync.py",
    "train/train_dcpa.py",
    "smoke_overfit_64.py",
    "tests/test_sanity.py",
    "tests/test_data_pipeline.py",
    "tests/test_trainer_lifecycle.py",
    "notebooks/Colab_DCPA_Train.ipynb",
    "dcpa-train-py/dcpa/model.py",
]
missing = [r for r in required if r not in names]
if missing:
    raise SystemExit(f"ERROR: Required files missing from bundle: {missing}")

print(f"  [OK] Zip integrity verified ({len(names)} clean entries, zero forbidden artifacts).")
PYEOF

echo "[DONE] Ready to upload ${BUNDLE_PATH} to /content/drive/MyDrive/DCPA_Project/code/dcpa_colab_bundle.zip"
