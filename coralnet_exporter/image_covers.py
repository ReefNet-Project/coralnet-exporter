from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin

import requests

NOT_READY_KEYWORDS = ["prepar", "please wait", "queued", "try again", "processing", "not ready"]


def csrf_token_from_session(session: requests.Session) -> str | None:
    return session.cookies.get("csrftoken")


def export_image_covers(
    session: requests.Session,
    source_url: str,
    output_dir: Path,
    timeout: int = 600,
    max_wait_seconds: int = 1800,
    poll_interval_seconds: int = 30,
    label_display: str = "code",
) -> Path:
    """Export CoralNet image covers / percent cover CSV for all browse results."""
    csrf = csrf_token_from_session(session)
    if not csrf:
        # Lightweight request to establish the csrftoken cookie if login did not already do it.
        response = session.get(source_url, timeout=min(timeout, 120))
        response.raise_for_status()
        csrf = csrf_token_from_session(session)
    if not csrf:
        raise RuntimeError("Could not determine CSRF token for image covers export")

    prep_url = urljoin(source_url, "export/image_covers_prep/")
    payload = {
        "csrfmiddlewaretoken": csrf,
        "label_display": label_display,
        "export_format": "csv",
    }

    started = time.time()
    last_error = "image covers export did not finish"
    while time.time() - started <= max_wait_seconds:
        prep = session.post(
            prep_url,
            data=payload,
            headers={"Referer": urljoin(source_url, "browse/images/")},
            timeout=timeout,
        )
        if prep.status_code >= 500:
            last_error = f"image covers prep failed with HTTP {prep.status_code}"
            time.sleep(poll_interval_seconds)
            continue
        prep.raise_for_status()
        data = prep.json()
        if data.get("error"):
            raise RuntimeError(data["error"])
        timestamp = data.get("session_data_timestamp")
        if not timestamp:
            last_error = "image covers prep did not return session_data_timestamp"
            time.sleep(poll_interval_seconds)
            continue

        serve = session.get(
            urljoin(source_url, f"export/serve/?session_data_timestamp={timestamp}"),
            timeout=timeout,
        )
        if serve.headers.get("Content-Disposition") and serve.content:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "percent_cover.csv"
            path.write_bytes(serve.content)
            return path

        text = serve.text.lower() if serve.text else ""
        if serve.status_code in (202, 429) or any(k in text for k in NOT_READY_KEYWORDS):
            last_error = "image covers export still processing"
            time.sleep(poll_interval_seconds)
            continue

        last_error = f"unexpected image covers serve response status={serve.status_code}"
        time.sleep(poll_interval_seconds)

    raise RuntimeError(last_error)
