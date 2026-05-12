from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console


class LegacyScriptError(RuntimeError):
    pass


def module_script_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise LegacyScriptError(f"Could not locate legacy module: {module_name}")
    return Path(spec.origin)


def run_legacy_module(
    module_name: str,
    args: list[str],
    console: Console | None = None,
    env_updates: dict[str, str] | None = None,
) -> None:
    console = console or Console()
    script_path = module_script_path(module_name)
    cmd = [sys.executable, str(script_path), *args]
    console.print(f"[dim]Running:[/dim] {' '.join(cmd)}")
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    process = subprocess.run(cmd, env=env)
    if process.returncode != 0:
        raise LegacyScriptError(f"{module_name} failed with exit code {process.returncode}")
