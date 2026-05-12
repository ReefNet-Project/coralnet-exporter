import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DEFAULT_INPUT_CSV = "coralnet_sources_data_04_13_2026/coralnet_sources_more_than_250_with_images.csv"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_LOG_FILE = "scrape_labelsets.log"
DEFAULT_STATE_FILE = "labelset_progress_state.json"
DEFAULT_SUMMARY_FILE = "labelset_summary.json"
DEFAULT_FAILURES_FILE = "labelset_failures.csv"

LOGIN_URL = "https://coralnet.ucsd.edu/accounts/login/"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(name: str, default: str = "unnamed") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", str(name)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .")
    return cleaned or default


def normalize_source_name(row: pd.Series) -> str:
    for col in ("Source Name Cleaned", "Source Name", "Source"):
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return f"source_row_{int(row.name)}"


def normalize_source_url(url: str | None) -> str | None:
    if not isinstance(url, str):
        return None
    u = url.strip()
    if not u:
        return None
    if not u.endswith("/"):
        u += "/"
    return u


def setup_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("crawl_labelset")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    retries: int,
    timeout: int,
    logger: logging.Logger,
    context: str,
    **kwargs,
):
    err = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.request(method=method, url=url, timeout=timeout, **kwargs)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            return resp, None
        except requests.RequestException as exc:
            err = f"{context} request failed attempt {attempt}/{retries}: {exc}"
            if attempt == retries:
                logger.error(err)
                return None, err
            logger.warning(err)
            time.sleep(min(2 ** (attempt - 1), 5))
    return None, err


def login(session: requests.Session, username: str, password: str, retries: int, timeout: int, logger: logging.Logger) -> None:
    login_page, err = request_with_retries(
        session=session,
        method="GET",
        url=LOGIN_URL,
        retries=retries,
        timeout=timeout,
        logger=logger,
        context="login-page",
    )
    if login_page is None:
        raise RuntimeError(err or "failed to retrieve login page")

    soup = BeautifulSoup(login_page.text, "html.parser")
    token_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
    if token_input is None or not token_input.get("value"):
        raise RuntimeError("CSRF token not found on login page")

    payload = {
        "username": username,
        "password": password,
        "stay_signed_in": "on",
        "csrfmiddlewaretoken": token_input["value"],
    }
    headers = {"Referer": LOGIN_URL}

    response, err = request_with_retries(
        session=session,
        method="POST",
        url=LOGIN_URL,
        retries=retries,
        timeout=timeout,
        logger=logger,
        context="login",
        data=payload,
        headers=headers,
    )
    if response is None:
        raise RuntimeError(err or "login request failed")

    # Heuristic: if still on login page and no logout button, login likely failed.
    response_text = response.text.lower()
    if "accounts/login" in str(response.url).lower() and "logout" not in response_text:
        raise RuntimeError("login failed (still on login page)")


def parse_labelset_table(html_text: str) -> tuple[pd.DataFrame | None, str | None]:
    soup = BeautifulSoup(html_text, "html.parser")

    table = soup.find("table", {"id": "label-table"})
    if table is None:
        return None, "label table not found"

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    if not headers:
        return None, "label table headers missing"

    rows: list[list[str]] = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        # keep full row shape to match headers as much as possible
        if len(cells) < len(headers):
            cells = cells + [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[: len(headers)]
        rows.append(cells)

    df = pd.DataFrame(rows, columns=headers)
    return df, None


class StateStore:
    def __init__(self, state_path: Path, resume: bool):
        self.state_path = state_path
        self.state: dict[str, Any] = {
            "meta": {
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            },
            "sources": {},
        }
        if resume and state_path.exists():
            try:
                self.state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    def get(self, source_name: str) -> dict[str, Any]:
        return dict(self.state.get("sources", {}).get(source_name, {}))

    def update(self, source_name: str, data: dict[str, Any], flush: bool = False) -> None:
        src = self.state.setdefault("sources", {})
        entry = src.setdefault(source_name, {})
        entry.update(data)
        entry["updated_at"] = utc_now_iso()
        if flush:
            self.flush()

    def flush(self) -> None:
        self.state.setdefault("meta", {})["updated_at"] = utc_now_iso()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)


def write_failures_csv(path: Path, failures: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_name", "source_url", "labelset_path", "error"],
        )
        writer.writeheader()
        for row in failures:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CoralNet labelsets into output/<source>/labelset.csv")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Input source CSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output root (contains source folders)")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="Log file path")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="State JSON file path")
    parser.add_argument("--summary-file", default=DEFAULT_SUMMARY_FILE, help="Summary JSON file path")
    parser.add_argument("--failures-file", default=DEFAULT_FAILURES_FILE, help="Failures CSV file path")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout seconds")
    parser.add_argument("--retries", type=int, default=6, help="Retries per request")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", action="store_true", help="Resume from state file and skip done sources")
    parser.add_argument("--force", action="store_true", help="Redownload even if labelset.csv exists")
    parser.add_argument("--max-sources-per-run", type=int, default=None, help="Only process next N sources")
    parser.add_argument("--username", default=os.getenv("CORALNET_USERNAME") or os.getenv("USERNAME"), help="CoralNet username")
    parser.add_argument("--password", default=os.getenv("CORALNET_PASSWORD") or os.getenv("PASSWORD"), help="CoralNet password")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.username or not args.password:
        raise ValueError("Missing credentials. Provide --username/--password or set CORALNET_USERNAME/CORALNET_PASSWORD.")

    logger = setup_logger(Path(args.log_file), args.log_level)

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    state = StateStore(Path(args.state_file), resume=args.resume)

    df = pd.read_csv(input_csv)
    if "URL" not in df.columns:
        raise ValueError("Input CSV missing required column: URL")

    session = requests.Session()
    session.headers.update({"User-Agent": "CoralNetLabelsetCrawler/2.0"})

    logger.info("Starting labelset crawl")
    logger.info("Input CSV: %s (rows=%d)", str(input_csv), len(df))

    login(session, args.username, args.password, args.retries, args.timeout, logger)
    logger.info("Login successful")

    failures: list[dict[str, Any]] = []
    done = 0
    skipped_existing = 0
    skipped_done_state = 0
    processed = 0

    iterator = tqdm(df.iterrows(), total=len(df), desc="Labelsets", unit="source", dynamic_ncols=True)
    for _, row in iterator:
        source_name = normalize_source_name(row)
        source_url = normalize_source_url(row.get("URL"))
        safe_source_name = sanitize_filename(source_name, default="source")

        if source_url is None:
            failures.append(
                {
                    "source_name": source_name,
                    "source_url": "",
                    "labelset_path": str((output_dir / safe_source_name / "labelset.csv").resolve()),
                    "error": "missing URL",
                }
            )
            state.update(source_name, {"status": "failed", "error": "missing URL"}, flush=True)
            processed += 1
            continue

        labelset_path = output_dir / safe_source_name / "labelset.csv"

        if args.resume:
            entry = state.get(source_name)
            if entry.get("status") == "done" and labelset_path.exists() and not args.force:
                skipped_done_state += 1
                continue

        if labelset_path.exists() and labelset_path.stat().st_size > 0 and not args.force:
            skipped_existing += 1
            state.update(
                source_name,
                {
                    "status": "done",
                    "source_url": source_url,
                    "labelset_path": str(labelset_path.resolve()),
                    "error": None,
                },
                flush=False,
            )
            continue

        labelset_url = urljoin(source_url, "labelset/")
        response, err = request_with_retries(
            session=session,
            method="GET",
            url=labelset_url,
            retries=args.retries,
            timeout=args.timeout,
            logger=logger,
            context=f"labelset:{source_name}",
            headers={"Referer": source_url},
        )

        if response is None:
            failures.append(
                {
                    "source_name": source_name,
                    "source_url": source_url,
                    "labelset_path": str(labelset_path.resolve()),
                    "error": err or "request failed",
                }
            )
            state.update(
                source_name,
                {
                    "status": "failed",
                    "source_url": source_url,
                    "labelset_path": str(labelset_path.resolve()),
                    "error": err or "request failed",
                },
                flush=True,
            )
            processed += 1
            continue

        table_df, parse_err = parse_labelset_table(response.text)
        if table_df is None:
            failures.append(
                {
                    "source_name": source_name,
                    "source_url": source_url,
                    "labelset_path": str(labelset_path.resolve()),
                    "error": parse_err or "failed to parse label table",
                }
            )
            state.update(
                source_name,
                {
                    "status": "failed",
                    "source_url": source_url,
                    "labelset_path": str(labelset_path.resolve()),
                    "error": parse_err or "failed to parse label table",
                },
                flush=True,
            )
            processed += 1
            continue

        labelset_path.parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(labelset_path, index=False)

        state.update(
            source_name,
            {
                "status": "done",
                "source_url": source_url,
                "labelset_path": str(labelset_path.resolve()),
                "rows": int(len(table_df)),
                "error": None,
            },
            flush=True,
        )

        done += 1
        processed += 1

        if args.max_sources_per_run is not None and processed >= args.max_sources_per_run:
            logger.info("Reached max-sources-per-run=%d", args.max_sources_per_run)
            break

    state.flush()

    write_failures_csv(Path(args.failures_file), failures)

    summary = {
        "timestamp": utc_now_iso(),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir.resolve()),
        "processed": processed,
        "downloaded": done,
        "skipped_existing": skipped_existing,
        "skipped_done_state": skipped_done_state,
        "failed": len(failures),
        "resume": bool(args.resume),
        "force": bool(args.force),
    }

    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("State file: %s", args.state_file)
    logger.info("Failures file: %s", args.failures_file)
    logger.info("Summary file: %s", args.summary_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
