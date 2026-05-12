from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

NOT_READY_KEYWORDS = ["prepar", "please wait", "queued", "try again", "processing", "not ready"]


@dataclass(frozen=True)
class RateTableChoice:
    table_id: str
    name: str


def csrf_token_from_session(session: requests.Session) -> str | None:
    return session.cookies.get("csrftoken")


def fetch_rate_table_choices(session: requests.Session, source_url: str, timeout: int = 120) -> list[RateTableChoice]:
    """Read rate table choices from the Browse Images action form."""
    response = session.get(urljoin(source_url, "browse/images/"), timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    select = soup.find("select", {"name": "rate_table_id"})
    if select is None:
        return []
    choices = []
    for option in select.find_all("option"):
        value = option.get("value")
        text = option.get_text(" ", strip=True)
        if value:
            choices.append(RateTableChoice(table_id=str(value), name=text))
    return choices


def export_calcification_rates(
    session: requests.Session,
    source_url: str,
    output_dir: Path,
    rate_table_id: str,
    timeout: int = 600,
    max_wait_seconds: int = 1800,
    poll_interval_seconds: int = 30,
    label_display: str = "code",
    optional_columns: list[str] | None = None,
) -> Path:
    """Export CoralNet calcification rates CSV for all browse results."""
    if label_display not in {"code", "name"}:
        raise ValueError("label_display must be 'code' or 'name'")
    optional_columns = optional_columns or []
    bad_optional = set(optional_columns) - {"per_label_mean", "per_label_bounds"}
    if bad_optional:
        raise ValueError(f"Unsupported calcification optional columns: {', '.join(sorted(bad_optional))}")

    csrf = csrf_token_from_session(session)
    if not csrf:
        response = session.get(source_url, timeout=min(timeout, 120))
        response.raise_for_status()
        csrf = csrf_token_from_session(session)
    if not csrf:
        raise RuntimeError("Could not determine CSRF token for calcification export")

    prep_url = urljoin(source_url, "calcification/stats_export_prep/")
    payload: list[tuple[str, str]] = [
        ("csrfmiddlewaretoken", csrf),
        ("rate_table_id", str(rate_table_id)),
        ("label_display", label_display),
        ("export_format", "csv"),
    ]
    for col in optional_columns:
        payload.append(("optional_columns", col))

    started = time.time()
    last_error = "calcification export did not finish"
    while time.time() - started <= max_wait_seconds:
        prep = session.post(
            prep_url,
            data=payload,
            headers={"Referer": urljoin(source_url, "browse/images/")},
            timeout=timeout,
        )
        if prep.status_code >= 500:
            last_error = f"calcification prep failed with HTTP {prep.status_code}"
            time.sleep(poll_interval_seconds)
            continue
        prep.raise_for_status()
        data = prep.json()
        if data.get("error"):
            raise RuntimeError(data["error"])
        timestamp = data.get("session_data_timestamp")
        if not timestamp:
            last_error = "calcification prep did not return session_data_timestamp"
            time.sleep(poll_interval_seconds)
            continue

        serve = session.get(
            urljoin(source_url, f"export/serve/?session_data_timestamp={timestamp}"),
            timeout=timeout,
        )
        if serve.headers.get("Content-Disposition") and serve.content:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "calcification_rates.csv"
            path.write_bytes(serve.content)
            return path

        text = serve.text.lower() if serve.text else ""
        if serve.status_code in (202, 429) or any(k in text for k in NOT_READY_KEYWORDS):
            last_error = "calcification export still processing"
            time.sleep(poll_interval_seconds)
            continue

        last_error = f"unexpected calcification serve response status={serve.status_code}"
        time.sleep(poll_interval_seconds)

    raise RuntimeError(last_error)


def parse_optional_columns(text: str) -> list[str]:
    if not text.strip():
        return []
    return [item.strip() for item in text.split(",") if item.strip()]
