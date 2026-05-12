from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_credentials(
    username: str | None = None,
    password: str | None = None,
    env_file: Path | None = None,
    interactive: bool = True,
    console: Console | None = None,
) -> Credentials:
    console = console or Console()
    if env_file is not None:
        load_dotenv(env_file)

    username = username or os.getenv("CORALNET_USERNAME") or os.getenv("USERNAME")
    password = password or os.getenv("CORALNET_PASSWORD") or os.getenv("PASSWORD")

    if interactive and not username:
        username = console.input("CoralNet username: ").strip()
    if interactive and not password:
        password = getpass.getpass("CoralNet password: ")

    if not username or not password:
        raise ValueError(
            "Missing CoralNet credentials. Set CORALNET_USERNAME/CORALNET_PASSWORD, "
            "create .env.coralnet, or run interactively."
        )
    return Credentials(username=username, password=password)
