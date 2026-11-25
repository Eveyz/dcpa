"""Git-based cloud/Colab workspace synchronizer (`git clone` / `git pull --ff-only`).

Replaces manual `dcpa_colab_bundle.zip` uploads with clean one-command Git synchronization while:
  - Never persisting `GITHUB_TOKEN` credentials inside `.git/config` (passes HTTP Basic auth via
    ephemeral `git -c http.extraHeader=...` flags only when `token` is provided).
  - Supporting seamless branch switching (`--branch main` or `--branch feat/dcpa-turbo-parallel`).
  - Optionally falling back to Google Drive `dcpa_colab_bundle.zip` if offline or `repo_url` is empty.
"""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional


DEFAULT_GIT_REPO_URL = "https://github.com/Eveyz/dcpa.git"


def _try_colab_github_token() -> Optional[str]:
    env_tok = os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()
    if env_tok:
        return env_tok
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        val = userdata.get("GITHUB_TOKEN")
        if val and str(val).strip():
            return str(val).strip()
    except Exception:
        pass
    return None


def _git_auth_prefix(repo_url: str, token: Optional[str]) -> list[str]:
    """Build ephemeral `git -c http.extraHeader=...` args without writing secrets to `.git/config`."""
    if not token or not repo_url.startswith("https://"):
        return ["git"]
    basic = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    return ["git", f"-c", f"http.extraHeader=Authorization: Basic {basic}"]


def sync_repo_via_git(
    target_dir: str | Path,
    *,
    repo_url: str = DEFAULT_GIT_REPO_URL,
    branch: Optional[str] = None,
    token: Optional[str] = None,
    fallback_bundle_zip: Optional[str | Path] = None,
) -> dict[str, str]:
    """Clone or fast-forward pull `repo_url` into `target_dir` and return `{mode, branch, commit}`."""
    dst = Path(target_dir).resolve()
    eff_tok = token if token is not None else _try_colab_github_token()
    clean_branch = branch.strip() if branch and branch.strip() else None

    if repo_url and repo_url.strip():
        git_cmd = _git_auth_prefix(repo_url.strip(), eff_tok)
        try:
            if (dst / ".git").exists():
                subprocess.run([*git_cmd, "-C", str(dst), "remote", "set-url", "origin", repo_url.strip()], check=True)
                subprocess.run([*git_cmd, "-C", str(dst), "fetch", "--prune", "origin"], check=True)
                if clean_branch:
                    # Checkout local or remote tracking branch cleanly and fast-forward to origin/<branch>
                    rc = subprocess.run(
                        [*git_cmd, "-C", str(dst), "checkout", clean_branch],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if rc.returncode != 0:
                        subprocess.run(
                            [*git_cmd, "-C", str(dst), "checkout", "-B", clean_branch, f"origin/{clean_branch}"],
                            check=True,
                        )
                    subprocess.run([*git_cmd, "-C", str(dst), "pull", "--ff-only", "origin", clean_branch], check=True)
                else:
                    subprocess.run([*git_cmd, "-C", str(dst), "pull", "--ff-only"], check=True)
                mode = "git_pull"
            else:
                if dst.exists():
                    shutil.rmtree(dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                clone_args = [*git_cmd, "clone"]
                if clean_branch:
                    clone_args.extend(["--branch", clean_branch])
                clone_args.extend([repo_url.strip(), str(dst)])
                subprocess.run(clone_args, check=True)
                mode = "git_clone"

            rev = subprocess.run(
                ["git", "-C", str(dst), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            cur_br = subprocess.run(
                ["git", "-C", str(dst), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            return {"mode": mode, "branch": cur_br, "commit": rev, "path": str(dst)}
        except Exception as git_exc:
            if fallback_bundle_zip is None or not Path(fallback_bundle_zip).exists():
                raise RuntimeError(
                    f"Git sync failed for {repo_url!r} (branch={clean_branch!r}): {git_exc}"
                ) from git_exc
            print(
                f"[GitSync][WARN] Git sync failed ({git_exc}); falling back to Drive bundle {fallback_bundle_zip}",
                flush=True,
            )

    if fallback_bundle_zip is not None and Path(fallback_bundle_zip).exists():
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(fallback_bundle_zip, "r") as zf:
            zf.extractall(dst)
        return {"mode": "bundle_zip", "branch": "bundle", "commit": "zip", "path": str(dst)}

    raise FileNotFoundError(
        f"Neither reachable Git repository ({repo_url!r}) nor fallback bundle ({fallback_bundle_zip}) available."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Synchronize DCPA repository via git clone / git pull --ff-only")
    ap.add_argument("--target-dir", type=str, default=".", help="Local checkout directory (e.g. /content/dcpa)")
    ap.add_argument("--repo-url", type=str, default=DEFAULT_GIT_REPO_URL)
    ap.add_argument("--branch", type=str, default=None)
    ap.add_argument("--fallback-bundle", type=str, default=None)
    args = ap.parse_args()

    info = sync_repo_via_git(
        args.target_dir,
        repo_url=args.repo_url,
        branch=args.branch,
        fallback_bundle_zip=args.fallback_bundle,
    )
    print(
        f"[GitSync] mode={info['mode']} | branch={info['branch']} | commit={info['commit']} | path={info['path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
