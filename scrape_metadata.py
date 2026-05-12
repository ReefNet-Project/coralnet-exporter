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
DEFAULT_LOG_FILE = "scrape_metadata.log"
DEFAULT_STATE_FILE = "metadata_progress_state.json"
DEFAULT_SUMMARY_FILE = "metadata_summary.json"
DEFAULT_FAILURES_FILE = "metadata_failures.csv"

LOGIN_URL = "https://coralnet.ucsd.edu/accounts/login/"
NOT_READY_KEYWORDS = [
    "prepar",
    "please wait",
    "queued",
    "try again",
    "processing",
    "not ready",
]

CANONICAL_TYPES = {
    "all": "all",
    "on_confirmed": "on_confirmed",
    "confirmed": "on_confirmed",
}


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


def parse_metadata_types(types_text: str) -> list[str]:
    items = [x.strip().lower() for x in types_text.split(",") if x.strip()]
    if not items:
        raise ValueError("No metadata types provided")

    result: list[str] = []
    for item in items:
        if item not in CANONICAL_TYPES:
            raise ValueError(f"Unsupported metadata type '{item}'. Allowed: all,on_confirmed")
        canonical = CANONICAL_TYPES[item]
        if canonical not in result:
            result.append(canonical)
    return result


def setup_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("scrape_metadata")
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

    response_text = response.text.lower()
    if "accounts/login" in str(response.url).lower() and "logout" not in response_text:
        raise RuntimeError("login failed (still on login page)")


def fetch_source_csrf_token(
    session: requests.Session,
    source_url: str,
    retries: int,
    timeout: int,
    logger: logging.Logger,
) -> tuple[str | None, str | None]:
    browse_url = urljoin(source_url, "browse/images/")
    response, err = request_with_retries(
        session=session,
        method="GET",
        url=browse_url,
        retries=retries,
        timeout=timeout,
        logger=logger,
        context=f"browse-images:{source_url}",
    )
    if response is None:
        return None, err

    soup = BeautifulSoup(response.text, "html.parser")
    token_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
    if token_input is None or not token_input.get("value"):
        return None, "CSRF token not found on source browse/images page"

    return token_input["value"], None


def metadata_filename(metadata_type: str) -> str:
    return "metadata_all.csv" if metadata_type == "all" else "metadata_confirmed.csv"


def build_metadata_payload(csrf_token: str, metadata_type: str) -> dict[str, str]:
    payload = {"csrfmiddlewaretoken": csrf_token}
    if metadata_type == "on_confirmed":
        payload["annotation_status"] = "confirmed"
    return payload


def is_export_not_ready_response(response: requests.Response) -> bool:
    if response.status_code in (202, 429):
        return True
    text_lower = response.text.lower() if response.text else ""
    if any(k in text_lower for k in NOT_READY_KEYWORDS):
        return True
    return bool("browse images" in text_lower or "export/metadata" in str(response.url))


class StateStore:
    def __init__(self, state_path: Path, resume: bool):
        self.state_path = state_path
        self.state: dict[str, Any] = {
            "meta": {"created_at": utc_now_iso(), "updated_at": utc_now_iso()},
            "sources": {},
        }
        if resume and state_path.exists():
            try:
                self.state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    def get_type_status(self, source_name: str, metadata_type: str) -> dict[str, Any]:
        source_entry = self.state.get("sources", {}).get(source_name, {})
        return dict(source_entry.get("types", {}).get(metadata_type, {}))

    def update_type(self, source_name: str, source_url: str, metadata_type: str, data: dict[str, Any], flush: bool = False) -> None:
        sources = self.state.setdefault("sources", {})
        source_entry = sources.setdefault(source_name, {})
        source_entry["source_url"] = source_url
        types = source_entry.setdefault("types", {})
        entry = types.setdefault(metadata_type, {})
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
            fieldnames=["source_name", "source_url", "metadata_type", "metadata_path", "error"],
        )
        writer.writeheader()
        for row in failures:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CoralNet metadata files into output/<source>/metadata_all.csv and/or metadata_confirmed.csv"
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Input source CSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output root directory")
    parser.add_argument("--metadata-types", default="all", help="Comma-separated: all,on_confirmed")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="Log file path")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="State JSON path")
    parser.add_argument("--summary-file", default=DEFAULT_SUMMARY_FILE, help="Summary JSON path")
    parser.add_argument("--failures-file", default=DEFAULT_FAILURES_FILE, help="Failures CSV path")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout seconds")
    parser.add_argument("--retries", type=int, default=6, help="Retries per request")
    parser.add_argument("--export-timeout", type=int, default=300, help="Per metadata export request timeout seconds")
    parser.add_argument("--export-retries", type=int, default=3, help="Retries per metadata export request")
    parser.add_argument("--max-wait-seconds", type=int, default=1800, help="Max wait per source/type for metadata export")
    parser.add_argument("--poll-interval-seconds", type=int, default=20, help="Sleep between metadata export polls")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", action="store_true", help="Resume and skip done metadata files")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they exist")
    parser.add_argument("--max-sources-per-run", type=int, default=None, help="Only process next N sources")
    parser.add_argument("--username", default=os.getenv("CORALNET_USERNAME") or os.getenv("USERNAME"), help="CoralNet username")
    parser.add_argument("--password", default=os.getenv("CORALNET_PASSWORD") or os.getenv("PASSWORD"), help="CoralNet password")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_types = parse_metadata_types(args.metadata_types)

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
    session.headers.update({"User-Agent": "CoralNetMetadataCrawler/2.0"})

    logger.info("Starting metadata crawl")
    logger.info("Input CSV: %s (rows=%d)", str(input_csv), len(df))
    logger.info("Metadata types: %s", ",".join(metadata_types))

    login(session, args.username, args.password, args.retries, args.timeout, logger)
    logger.info("Login successful")

    failures: list[dict[str, Any]] = []
    downloaded = 0
    skipped_existing = 0
    skipped_done_state = 0
    processed_sources = 0

    iterator = tqdm(df.iterrows(), total=len(df), desc="Metadata", unit="source", dynamic_ncols=True)
    for _, row in iterator:
        source_name = normalize_source_name(row)
        source_url = normalize_source_url(row.get("URL"))
        safe_source_name = sanitize_filename(source_name, default="source")

        if source_url is None:
            for mtype in metadata_types:
                failures.append(
                    {
                        "source_name": source_name,
                        "source_url": "",
                        "metadata_type": mtype,
                        "metadata_path": str((output_dir / safe_source_name / metadata_filename(mtype)).resolve()),
                        "error": "missing URL",
                    }
                )
                state.update_type(source_name, "", mtype, {"status": "failed", "error": "missing URL"}, flush=True)
            processed_sources += 1
            continue

        csrf_token, csrf_err = fetch_source_csrf_token(
            session=session,
            source_url=source_url,
            retries=args.retries,
            timeout=args.timeout,
            logger=logger,
        )
        if csrf_token is None:
            for mtype in metadata_types:
                failures.append(
                    {
                        "source_name": source_name,
                        "source_url": source_url,
                        "metadata_type": mtype,
                        "metadata_path": str((output_dir / safe_source_name / metadata_filename(mtype)).resolve()),
                        "error": csrf_err or "failed to get csrf token",
                    }
                )
                state.update_type(
                    source_name,
                    source_url,
                    mtype,
                    {"status": "failed", "error": csrf_err or "failed to get csrf token"},
                    flush=True,
                )
            processed_sources += 1
            continue

        for mtype in metadata_types:
            metadata_path = output_dir / safe_source_name / metadata_filename(mtype)

            if args.resume:
                st = state.get_type_status(source_name, mtype)
                if st.get("status") == "done" and metadata_path.exists() and not args.force:
                    skipped_done_state += 1
                    continue

            if metadata_path.exists() and metadata_path.stat().st_size > 0 and not args.force:
                skipped_existing += 1
                state.update_type(
                    source_name,
                    source_url,
                    mtype,
                    {
                        "status": "done",
                        "metadata_path": str(metadata_path.resolve()),
                        "error": None,
                    },
                    flush=False,
                )
                continue

            payload = build_metadata_payload(csrf_token, mtype)
            headers = {"Referer": urljoin(source_url, "browse/images/")}
            export_url = urljoin(source_url, "export/metadata/")
            logger.info("[%s:%s] requesting metadata export", source_name, mtype)
            started = time.time()
            success = False
            last_error = None

            while time.time() - started <= args.max_wait_seconds:
                response, req_err = request_with_retries(
                    session=session,
                    method="POST",
                    url=export_url,
                    retries=args.export_retries,
                    timeout=args.export_timeout,
                    logger=logger,
                    context=f"metadata:{source_name}:{mtype}",
                    headers=headers,
                    data=payload,
                )

                if response is None:
                    last_error = req_err or "request failed"
                    time.sleep(args.poll_interval_seconds)
                    continue

                if not response.ok:
                    if response.status_code >= 500:
                        last_error = f"HTTP {response.status_code}"
                        time.sleep(args.poll_interval_seconds)
                        continue
                    last_error = f"HTTP {response.status_code}"
                    break

                if response.headers.get("Content-Disposition") and response.content:
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata_path.write_bytes(response.content)
                    state.update_type(
                        source_name,
                        source_url,
                        mtype,
                        {
                            "status": "done",
                            "metadata_path": str(metadata_path.resolve()),
                            "bytes": int(len(response.content)),
                            "error": None,
                        },
                        flush=True,
                    )
                    downloaded += 1
                    success = True
                    break

                if is_export_not_ready_response(response):
                    last_error = "metadata export not ready yet"
                    logger.info("[%s:%s] metadata export still preparing; retrying in %ss", source_name, mtype, args.poll_interval_seconds)
                    time.sleep(args.poll_interval_seconds)
                    continue

                last_error = "response missing Content-Disposition (not a file export)"
                break

            if not success:
                failures.append(
                    {
                        "source_name": source_name,
                        "source_url": source_url,
                        "metadata_type": mtype,
                        "metadata_path": str(metadata_path.resolve()),
                        "error": last_error or "max wait exceeded",
                    }
                )
                state.update_type(
                    source_name,
                    source_url,
                    mtype,
                    {
                        "status": "failed",
                        "metadata_path": str(metadata_path.resolve()),
                        "error": last_error or "max wait exceeded",
                    },
                    flush=True,
                )

        processed_sources += 1
        if args.max_sources_per_run is not None and processed_sources >= args.max_sources_per_run:
            logger.info("Reached max-sources-per-run=%d", args.max_sources_per_run)
            break

    state.flush()
    write_failures_csv(Path(args.failures_file), failures)

    summary = {
        "timestamp": utc_now_iso(),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir.resolve()),
        "metadata_types": metadata_types,
        "processed_sources": processed_sources,
        "downloaded_files": downloaded,
        "skipped_existing": skipped_existing,
        "skipped_done_state": skipped_done_state,
        "failed_files": len(failures),
        "resume": bool(args.resume),
        "force": bool(args.force),
        "timeout": args.timeout,
        "retries": args.retries,
        "export_timeout": args.export_timeout,
        "export_retries": args.export_retries,
        "max_wait_seconds": args.max_wait_seconds,
        "poll_interval_seconds": args.poll_interval_seconds,
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
