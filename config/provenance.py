"""Provenance recording for experiment runs."""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from config.schema import ExperimentConfig


def save_provenance(
    config: ExperimentConfig,
    output_dir: str | Path,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save resolved config and provenance metadata.

    Args:
        config: Resolved experiment configuration.
        output_dir: Directory to save provenance files.
        metrics: Optional metrics dictionary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Save manifest with provenance metadata
    manifest: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "seed": config.seed,
        "git_commit": _get_git_commit(),
        "git_dirty": _is_git_dirty(),
        "python_version": _get_python_version(),
    }

    if metrics:
        manifest["metrics"] = metrics

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _is_git_dirty() -> bool:
    """Check if git working directory is dirty."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode != 0
    except Exception:
        return False


def _get_python_version() -> str:
    """Get Python version string."""
    import sys

    return sys.version
