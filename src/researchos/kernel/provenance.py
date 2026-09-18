"""Provenance capture and artifact verification.

Deterministic by construction: commit hashes, file digests and environment strings are computed,
never narrated. This is the module that lets every number answer "which code, which config, which
data, which seed?".
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..models.common import (
    ArtifactRef,
    Provenance,
    VerificationStatus,
    sha256_file,
    sha256_json,
    utcnow,
)
from .errors import VerificationError

GIT_TIMEOUT = 10


def _run_git(root: Path, *args: str) -> str | None:
    """Run git, returning ``None`` instead of raising when git is unavailable or not a repo."""
    if shutil.which("git") is None:
        return None
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_info(root: str | os.PathLike[str]) -> dict[str, object]:
    """Commit, dirty flag and remote for the repository containing ``root``."""
    path = Path(root)
    commit = _run_git(path, "rev-parse", "HEAD")
    if commit is None:
        return {}
    status = _run_git(path, "status", "--porcelain")
    remote = _run_git(path, "config", "--get", "remote.origin.url")
    branch = _run_git(path, "rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(status),
        "remote": remote,
        "branch": branch,
    }


def environment_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
    }


def config_hash(config: Mapping[str, object] | None) -> str | None:
    if not config:
        return None
    return sha256_json(dict(config))


def dataset_hashes(files: Iterable[str | os.PathLike[str]]) -> list[str]:
    out: list[str] = []
    for item in files:
        path = Path(item)
        if path.is_file():
            out.append(sha256_file(path))
    return out


def register_artifact(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    kind: str = "file",
    note: str | None = None,
) -> ArtifactRef:
    """Hash a file into an :class:`ArtifactRef` with a project-relative path."""
    root_path = Path(root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root_path / target
    target = target.resolve()
    if not target.is_file():
        raise VerificationError(f"cannot register missing artifact {target}")
    try:
        relative = target.relative_to(root_path).as_posix()
    except ValueError:
        relative = target.as_posix()
    return ArtifactRef(
        path=relative,
        sha256=sha256_file(target),
        bytes=target.stat().st_size,
        kind=kind,
        note=note,
    )


def verify_artifact(
    root: str | os.PathLike[str], ref: ArtifactRef
) -> tuple[VerificationStatus, str | None]:
    """Re-hash one artifact. Returns ``(status, detail)``; never raises for a missing file."""
    root_path = Path(root)
    target = root_path / ref.path if not Path(ref.path).is_absolute() else Path(ref.path)
    if not target.is_file():
        return VerificationStatus.MISSING_ARTIFACT, f"{ref.path} not found"
    actual = sha256_file(target)
    if not ref.sha256:
        return VerificationStatus.UNVERIFIED, "no recorded hash"
    if actual != ref.sha256:
        return (
            VerificationStatus.HASH_MISMATCH,
            f"{ref.path}: recorded {ref.sha256[:12]}…, actual {actual[:12]}… — the artifact changed "
            "after it was recorded",
        )
    return VerificationStatus.VERIFIED, None


def verify_artifacts(
    root: str | os.PathLike[str], refs: Sequence[ArtifactRef]
) -> list[tuple[ArtifactRef, VerificationStatus, str | None]]:
    return [(ref, *verify_artifact(root, ref)) for ref in refs]


def capture_provenance(
    root: str | os.PathLike[str],
    *,
    config: Mapping[str, object] | None = None,
    seeds: Sequence[int] | None = None,
    command: str | None = None,
    data_files: Iterable[str | os.PathLike[str]] = (),
    model_hash: str | None = None,
    artifact_paths: Iterable[str | os.PathLike[str]] = (),
    extra_source_files: Iterable[str] = (),
) -> Provenance:
    """Collect provenance for a record about to be created.

    Anything that cannot be determined is left ``None`` and will show up in
    :meth:`Provenance.missing` rather than being invented.
    """
    root_path = Path(root)
    info = git_info(root_path)
    env = environment_info()
    artifact_hashes: list[str] = []
    for path in artifact_paths:
        try:
            artifact_hashes.append(register_artifact(root_path, path).sha256)
        except VerificationError:
            continue
    return Provenance(
        code_commit=info.get("commit") if isinstance(info.get("commit"), str) else None,
        code_dirty=bool(info["dirty"]) if "dirty" in info else None,
        code_remote=info.get("remote") if isinstance(info.get("remote"), str) else None,
        config_hash=config_hash(config),
        dataset_hashes=dataset_hashes(data_files),
        model_hash=model_hash,
        environment=env,
        random_seeds=list(seeds or []),
        timestamp=utcnow(),
        artifact_hash=artifact_hashes[0] if artifact_hashes else None,
        command=command,
        host=env.get("host"),
        python_version=env.get("python"),
        source_files=list(extra_source_files),
    )


def short(commit: str | None, length: int = 8) -> str:
    if not commit:
        return "no-commit"
    return commit[:length]
