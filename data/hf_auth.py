"""Hugging Face Hub authentication helper shared across all DCPA data builders.

Resolution order (first non-empty wins):
  1. Explicit `cli_token` (`--hf-token`)
  2. Environment variables `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
  3. Repo-local `.env` file (`local.env` by default, ignored by git)
  4. Google Colab Secrets (`google.colab.userdata.get("HF_TOKEN")`)
  5. Existing HuggingFace CLI cached token (`huggingface_hub.get_token()`)

Once resolved, exports both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` into `os.environ` so
`datasets`, `huggingface_hub`, `hf_transfer`, `ProcessPoolExecutor` workers, and child
subprocesses all send authenticated requests (higher rate limits + gated datasets).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOKEN_ENV_KEYS: tuple[str, ...] = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _read_token_from_env_file(env_path: Path) -> Optional[str]:
    if not env_path.is_file():
        return None
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if key not in _TOKEN_ENV_KEYS:
                continue
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1].strip()
            elif " #" in val:
                val = val.split(" #", 1)[0].strip()
            if val:
                return val
    except Exception:
        return None
    return None


def _try_colab_secret(name: str) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        val = userdata.get(name)
        if val and str(val).strip():
            return str(val).strip()
    except Exception:
        return None
    return None


def _try_hf_hub_cached_token() -> Optional[str]:
    try:
        from huggingface_hub import get_token

        val = get_token()
        if val and str(val).strip():
            return str(val).strip()
    except Exception:
        return None
    return None


def resolve_hf_token(
    cli_token: Optional[str] = None,
    *,
    env_file: Optional[str | Path] = None,
) -> Optional[str]:
    """Resolve a Hugging Face token without mutating environment or leaking secrets."""
    if cli_token and str(cli_token).strip():
        return str(cli_token).strip()

    for key in _TOKEN_ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return val

    candidate_env = Path(env_file) if env_file is not None else (_REPO_ROOT / "local.env")
    tok = _read_token_from_env_file(candidate_env)
    if tok:
        return tok

    for key in _TOKEN_ENV_KEYS:
        tok = _try_colab_secret(key)
        if tok:
            return tok

    return _try_hf_hub_cached_token()


def get_hf_token() -> Optional[str]:
    """Fast lookup of an already-configured `HF_TOKEN` from `os.environ`."""
    for key in _TOKEN_ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def configure_hf_auth(
    cli_token: Optional[str] = None,
    *,
    verbose: bool = True,
    env_file: Optional[str | Path] = None,
) -> Optional[str]:
    """Resolve and export `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` into `os.environ`."""
    tok = resolve_hf_token(cli_token, env_file=env_file)
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        if verbose:
            print("[HF Auth] HF_TOKEN=SET (authenticated HF Hub requests enabled).", flush=True)
    elif verbose:
        print(
            "[HF Auth] HF_TOKEN not set (unauthenticated; set HF_TOKEN env / Colab Secrets / "
            "local.env / --hf-token for higher rate limits and gated datasets).",
            flush=True,
        )
    return tok


def hf_token_kwargs(token: Optional[str] = None) -> dict[str, str]:
    """Return `{'token': tok}` when a token is active, else `{}`."""
    tok = (token or "").strip() if token else get_hf_token()
    return {"token": tok} if tok else {}


def hf_auth_headers(
    extra: Optional[dict[str, str]] = None,
    *,
    token: Optional[str] = None,
) -> dict[str, str]:
    """Build HTTP headers for direct `hf_hub_url` requests with optional Bearer auth."""
    headers: dict[str, str] = {"User-Agent": "dcpa-corpus-builder/1.0"}
    if extra:
        headers.update(extra)
    tok = (token or "").strip() if token else get_hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers
