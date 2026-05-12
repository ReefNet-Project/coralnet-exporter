import argparse
import csv
import shutil
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
DEFAULT_LOG_FILE = "scrape_annotations.log"
DEFAULT_STATE_FILE = "annotations_progress_state.json"
DEFAULT_SUMMARY_FILE = "annotations_summary.json"
DEFAULT_FAILURES_FILE = "annotations_failures.csv"

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


def parse_annotation_types(text: str) -> list[str]:
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not items:
        raise ValueError("No annotation types provided")

    result: list[str] = []
    for item in items:
        if item not in CANONICAL_TYPES:
            raise ValueError(f"Unsupported annotation type '{item}'. Allowed: all,on_confirmed")
        canonical = CANONICAL_TYPES[item]
        if canonical not in result:
            result.append(canonical)
    return result


def annotations_filename(annotation_type: str) -> str:
    return "annotations_all.csv" if annotation_type == "all" else "annotations_confirmed.csv"


def materialize_legacy_all_annotations(source_dir: Path, target_all_path: Path) -> bool:
    """If legacy annotations.csv exists, materialize annotations_all.csv from it."""
    legacy_path = source_dir / "annotations.csv"
    if not legacy_path.exists() or not legacy_path.is_file():
        return False
    if target_all_path.exists():
        return True
    try:
        os.link(legacy_path, target_all_path)
    except OSError:
        shutil.copy2(legacy_path, target_all_path)
    return True


def setup_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("scrape_annotations")
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

    response, err = request_with_retries(
        session=session,
        method="POST",
        url=LOGIN_URL,
        retries=retries,
        timeout=timeout,
        logger=logger,
        context="login",
        data=payload,
        headers={"Referer": LOGIN_URL},
    )
    if response is None:
        raise RuntimeError(err or "login failed")

    text_lower = response.text.lower()
    if "accounts/login" in str(response.url).lower() and "logout" not in text_lower:
        raise RuntimeError("login failed (still on login page)")


def fetch_source_csrf_token(
    session: requests.Session,
    source_url: str,
    retries: int,
    timeout: int,
    logger: logging.Logger,
) -> tuple[str | None, str | None]:
    cookie_token = session.cookies.get("csrftoken")
    if cookie_token:
        return cookie_token, None

    browse_url = urljoin(source_url, "browse/images/")
    source_page, err = request_with_retries(
        session=session,
        method="GET",
        url=browse_url,
        retries=retries,
        timeout=timeout,
        logger=logger,
        context=f"browse-images-page:{source_url}",
    )
    if source_page is None:
        cookie_token = session.cookies.get("csrftoken")
        if cookie_token:
            logger.warning(
                "Falling back to csrftoken cookie after browse/images CSRF fetch failed for %s",
                source_url,
            )
            return cookie_token, None
        return None, err

    soup = BeautifulSoup(source_page.text, "html.parser")
    token_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
    if token_input is None or not token_input.get("value"):
        cookie_token = session.cookies.get("csrftoken")
        if cookie_token:
            logger.warning(
                "Falling back to csrftoken cookie because browse/images had no CSRF input for %s",
                source_url,
            )
            return cookie_token, None
        return None, "CSRF token not found on source browse/images page"
    return token_input["value"], None


def parse_optional_columns(text: str) -> list[str]:
    cols = [c.strip() for c in text.split(",") if c.strip()]
    if not cols:
        return ["annotator_info", "machine_suggestions", "metadata_date_aux", "metadata_other"]
    return cols


def build_export_payload(
    csrf_token: str,
    optional_columns: list[str],
    annotation_type: str,
    image_id_range: tuple[int, int] | None = None,
) -> list[tuple[str, str]]:
    payload: list[tuple[str, str]] = [
        ("csrfmiddlewaretoken", csrf_token),
        ("label_format", "code"),
    ]
    for col in optional_columns:
        payload.append(("optional_columns", col))
    if annotation_type == "on_confirmed":
        payload.append(("annotation_status", "confirmed"))
    if image_id_range is not None:
        payload.append(("image_id_range", f"{image_id_range[0]}_{image_id_range[1]}"))
    return payload


def export_annotations_once(
    session: requests.Session,
    source_url: str,
    csrf_token: str,
    optional_columns: list[str],
    annotation_type: str,
    export_retries: int,
    export_timeout: int,
    logger: logging.Logger,
    image_id_range: tuple[int, int] | None = None,
) -> tuple[str, bytes | None, str | None]:
    prep_url = urljoin(source_url, "annotation/export_prep/")
    payload = build_export_payload(
        csrf_token=csrf_token,
        optional_columns=optional_columns,
        annotation_type=annotation_type,
        image_id_range=image_id_range,
    )
    context_suffix = f"{annotation_type}:{image_id_range[0]}-{image_id_range[1]}" if image_id_range else annotation_type

    prep_response, err = request_with_retries(
        session=session,
        method="POST",
        url=prep_url,
        retries=export_retries,
        timeout=export_timeout,
        logger=logger,
        context=f"annotation-export-prep:{source_url}:{context_suffix}",
        data=payload,
        headers={"Referer": urljoin(source_url, "browse/images/")},
    )
    if prep_response is None:
        return "not_ready", None, err or "annotation export prep request failed"

    timestamp = None
    try:
        prep_json = prep_response.json()
        timestamp = prep_json.get("session_data_timestamp")
        if not prep_json.get("success", False):
            return "not_ready", None, "export prep returned success=false"
    except ValueError:
        return "error", None, "export prep response is not valid JSON"

    if not timestamp:
        return "error", None, "missing session_data_timestamp from export prep"

    serve_url = urljoin(source_url, f"export/serve/?session_data_timestamp={timestamp}")
    serve_response, err = request_with_retries(
        session=session,
        method="GET",
        url=serve_url,
        retries=export_retries,
        timeout=export_timeout,
        logger=logger,
        context=f"annotation-export-serve:{source_url}:{context_suffix}",
        allow_redirects=True,
    )
    if serve_response is None:
        return "not_ready", None, err or "annotation export serve request failed"

    if serve_response.headers.get("Content-Disposition") and serve_response.content:
        return "success", serve_response.content, None

    if serve_response.status_code in (202, 429):
        return "not_ready", None, f"status={serve_response.status_code}"

    text_lower = serve_response.text.lower() if serve_response.text else ""
    if any(k in text_lower for k in NOT_READY_KEYWORDS):
        return "not_ready", None, "server indicates export still processing"

    # CoralNet may redirect back to browse/images while still preparing.
    if "browse images" in text_lower or "source/" in str(serve_response.url):
        return "not_ready", None, "export file not ready yet"

    return "error", None, f"unexpected response (status={serve_response.status_code}, no file attachment)"


def image_ids_from_local_images(source_dir: Path) -> list[int]:
    image_dir = source_dir / "images"
    if not image_dir.exists():
        return []

    image_ids: set[int] = set()
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        match = re.match(r"^(\d+)__", path.name)
        if match:
            image_ids.add(int(match.group(1)))
    return sorted(image_ids)


def chunk_ranges_from_ids(image_ids: list[int], chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    ranges = []
    for start in range(0, len(image_ids), chunk_size):
        chunk = image_ids[start : start + chunk_size]
        ranges.append((chunk[0], chunk[-1]))
    return ranges


def csv_has_header(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            first_line = f.readline()
    except OSError:
        return False
    return first_line.startswith("Name,") and "Row" in first_line and "Column" in first_line


def merge_annotation_chunks(chunk_paths: list[Path], out_file: Path) -> tuple[bool, str | None, int]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    data_rows = 0
    wrote_header = False

    with tmp.open("w", encoding="utf-8", newline="") as out:
        for chunk_path in chunk_paths:
            if not chunk_path.exists() or chunk_path.stat().st_size == 0:
                return False, f"missing or empty chunk: {chunk_path}", data_rows
            if not csv_has_header(chunk_path):
                return False, f"chunk does not look like annotations CSV: {chunk_path}", data_rows

            with chunk_path.open("r", encoding="utf-8", errors="replace", newline="") as inp:
                header = inp.readline()
                if not wrote_header:
                    out.write(header)
                    wrote_header = True
                for line in inp:
                    out.write(line)
                    data_rows += 1

    if not wrote_header:
        tmp.unlink(missing_ok=True)
        return False, "no chunk headers found", data_rows

    tmp.replace(out_file)
    return True, None, data_rows


def export_annotations_chunked_by_image_id(
    session: requests.Session,
    source_name: str,
    source_url: str,
    source_dir: Path,
    out_file: Path,
    csrf_token: str,
    optional_columns: list[str],
    export_retries: int,
    export_timeout: int,
    max_wait_seconds: int,
    poll_interval_seconds: int,
    chunk_size: int,
    force: bool,
    logger: logging.Logger,
) -> tuple[bool, int, str | None]:
    image_ids = image_ids_from_local_images(source_dir)
    if not image_ids:
        return False, 0, f"no local image IDs found under {source_dir / 'images'}"

    ranges = chunk_ranges_from_ids(image_ids, chunk_size)
    chunk_dir = source_dir / ".annotation_chunks" / "all_by_image_id"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[%s:all] chunked fallback using %d local image IDs across %d chunks (chunk_size=%d)",
        source_name,
        len(image_ids),
        len(ranges),
        chunk_size,
    )

    chunk_paths: list[Path] = []
    chunk_iter = tqdm(
        list(enumerate(ranges, start=1)),
        total=len(ranges),
        desc=f"{source_name[:24]} chunks",
        unit="chunk",
        dynamic_ncols=True,
        leave=False,
    )

    for chunk_index, image_id_range in chunk_iter:
        start_id, end_id = image_id_range
        chunk_path = chunk_dir / f"chunk_{chunk_index:05d}_{start_id}_{end_id}.csv"
        chunk_paths.append(chunk_path)

        if chunk_path.exists() and chunk_path.stat().st_size > 0 and csv_has_header(chunk_path) and not force:
            continue

        started = time.time()
        last_error = None
        while time.time() - started <= max_wait_seconds:
            status, file_content, export_err = export_annotations_once(
                session=session,
                source_url=source_url,
                csrf_token=csrf_token,
                optional_columns=optional_columns,
                annotation_type="all",
                export_retries=export_retries,
                export_timeout=export_timeout,
                logger=logger,
                image_id_range=image_id_range,
            )

            if status == "success" and file_content is not None:
                tmp = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
                tmp.write_bytes(file_content)
                if not csv_has_header(tmp):
                    tmp.unlink(missing_ok=True)
                    last_error = f"chunk {chunk_index} response did not look like annotations CSV"
                    break
                tmp.replace(chunk_path)
                break

            if status == "error":
                last_error = export_err or "chunk export failed"
                break

            last_error = export_err or "chunk export not ready yet"
            logger.info(
                "[%s:all] chunk %d/%d (%d-%d) not ready; retrying in %ss",
                source_name,
                chunk_index,
                len(ranges),
                start_id,
                end_id,
                poll_interval_seconds,
            )
            time.sleep(poll_interval_seconds)

        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            return False, 0, last_error or f"chunk {chunk_index} failed or timed out"

    ok, merge_error, data_rows = merge_annotation_chunks(chunk_paths, out_file)
    if not ok:
        return False, data_rows, merge_error

    logger.info(
        "[%s:all] merged %d chunks into %s (%d data rows)",
        source_name,
        len(chunk_paths),
        out_file,
        data_rows,
    )
    return True, data_rows, None


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

    def get_type_status(self, source_name: str, annotation_type: str) -> dict[str, Any]:
        source_entry = self.state.get("sources", {}).get(source_name, {})
        return dict(source_entry.get("types", {}).get(annotation_type, {}))

    def update_type(self, source_name: str, source_url: str, annotation_type: str, data: dict[str, Any], flush: bool = False) -> None:
        sources = self.state.setdefault("sources", {})
        source_entry = sources.setdefault(source_name, {})
        source_entry["source_url"] = source_url
        types = source_entry.setdefault("types", {})
        entry = types.setdefault(annotation_type, {})
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
            fieldnames=["source_name", "source_url", "annotation_type", "annotations_path", "error"],
        )
        writer.writeheader()
        for row in failures:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CoralNet annotations into output/<source>/annotations_all.csv and/or annotations_confirmed.csv")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output root directory")
    parser.add_argument("--annotation-types", default="all", help="Comma-separated: all,on_confirmed")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="Log file path")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="State file path")
    parser.add_argument("--summary-file", default=DEFAULT_SUMMARY_FILE, help="Summary file path")
    parser.add_argument("--failures-file", default=DEFAULT_FAILURES_FILE, help="Failures CSV path")
    parser.add_argument("--timeout", type=int, default=60, help="General request timeout in seconds")
    parser.add_argument("--retries", type=int, default=6, help="General retries")
    parser.add_argument("--export-timeout", type=int, default=300, help="Per export POST/GET timeout in seconds")
    parser.add_argument("--export-retries", type=int, default=3, help="Retries for each export prep/serve request")
    parser.add_argument("--max-wait-seconds", type=int, default=1800, help="Max wait per source/type for export readiness")
    parser.add_argument("--poll-interval-seconds", type=int, default=20, help="Sleep between export polls")
    parser.add_argument(
        "--disable-chunked-fallback",
        action="store_true",
        help="Disable image_id_range chunk fallback for large annotations_all exports",
    )
    parser.add_argument(
        "--prefer-chunked-all",
        action="store_true",
        help="Use image_id_range chunks immediately for annotations_all instead of trying one full export first",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of local image IDs per image_id_range chunk when chunked fallback is used",
    )
    parser.add_argument(
        "--chunk-export-timeout",
        type=int,
        default=None,
        help="Per chunk export POST/GET timeout in seconds; defaults to --export-timeout",
    )
    parser.add_argument(
        "--chunk-max-wait-seconds",
        type=int,
        default=1200,
        help="Max wait per image_id_range chunk when chunked fallback is used",
    )
    parser.add_argument(
        "--chunk-poll-interval-seconds",
        type=int,
        default=60,
        help="Sleep between retries for image_id_range chunks",
    )
    parser.add_argument(
        "--optional-columns",
        default="annotator_info,machine_suggestions,metadata_date_aux,metadata_other",
        help="Comma-separated optional annotation columns",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", action="store_true", help="Resume and skip done outputs")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    parser.add_argument("--max-sources-per-run", type=int, default=None, help="Only process next N sources")
    parser.add_argument("--username", default=os.getenv("CORALNET_USERNAME") or os.getenv("USERNAME"), help="CoralNet username")
    parser.add_argument("--password", default=os.getenv("CORALNET_PASSWORD") or os.getenv("PASSWORD"), help="CoralNet password")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.username or not args.password:
        raise ValueError("Missing credentials. Provide --username/--password or set CORALNET_USERNAME/CORALNET_PASSWORD.")

    annotation_types = parse_annotation_types(args.annotation_types)
    optional_columns = parse_optional_columns(args.optional_columns)
    logger = setup_logger(Path(args.log_file), args.log_level)

    df = pd.read_csv(Path(args.input_csv))
    if "URL" not in df.columns:
        raise ValueError("Input CSV missing required column: URL")

    logger.info("Starting annotations crawl")
    logger.info("Input CSV: %s (rows=%d)", args.input_csv, len(df))
    logger.info("Annotation types: %s", ",".join(annotation_types))

    output_dir = Path(args.output_dir)
    state = StateStore(Path(args.state_file), resume=args.resume)

    failures: list[dict[str, Any]] = []
    downloaded_files = 0
    skipped_existing = 0
    skipped_done_state = 0
    processed_sources = 0
    pending_sources: list[dict[str, Any]] = []
    pending_files = 0

    for _, row in df.iterrows():
        source_name = normalize_source_name(row)
        source_url = normalize_source_url(row.get("URL"))
        safe_source_name = sanitize_filename(source_name, default="source")
        pending_types: list[str] = []

        if source_url is None:
            for atype in annotation_types:
                out_file = output_dir / safe_source_name / annotations_filename(atype)
                failures.append(
                    {
                        "source_name": source_name,
                        "source_url": "",
                        "annotation_type": atype,
                        "annotations_path": str(out_file.resolve()),
                        "error": "missing URL",
                    }
                )
                state.update_type(source_name, "", atype, {"status": "failed", "error": "missing URL"}, flush=True)
            continue

        for atype in annotation_types:
            out_file = output_dir / safe_source_name / annotations_filename(atype)
            source_dir = out_file.parent

            if atype == "all" and not args.force:
                # Backward compatibility for earlier runs that wrote annotations.csv.
                materialize_legacy_all_annotations(source_dir, out_file)

            if args.resume:
                st = state.get_type_status(source_name, atype)
                if st.get("status") == "done" and out_file.exists() and not args.force:
                    skipped_done_state += 1
                    continue

            if out_file.exists() and out_file.stat().st_size > 0 and not args.force:
                skipped_existing += 1
                state.update_type(
                    source_name,
                    source_url,
                    atype,
                    {
                        "status": "done",
                        "annotations_path": str(out_file.resolve()),
                        "error": None,
                    },
                    flush=False,
                )
                continue

            pending_types.append(atype)
            pending_files += 1

        if pending_types:
            pending_sources.append(
                {
                    "source_name": source_name,
                    "source_url": source_url,
                    "safe_source_name": safe_source_name,
                    "annotation_types": pending_types,
                }
            )

    state.flush()

    if args.max_sources_per_run is not None and len(pending_sources) > args.max_sources_per_run:
        pending_files = sum(len(item["annotation_types"]) for item in pending_sources[: args.max_sources_per_run])
        pending_sources = pending_sources[: args.max_sources_per_run]
        logger.info("Reached max-sources-per-run=%d during resume scan", args.max_sources_per_run)

    logger.info(
        "Resume scan selected %d pending sources (%d files); skipped done state=%d, existing files=%d",
        len(pending_sources),
        pending_files,
        skipped_done_state,
        skipped_existing,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "CoralNetAnnotationsCrawler/2.1"})

    if pending_sources:
        login(session, args.username, args.password, args.retries, args.timeout, logger)
        logger.info("Login successful")

    iterator = tqdm(pending_sources, total=len(pending_sources), desc="Annotations", unit="source", dynamic_ncols=True)
    for item in iterator:
        source_name = item["source_name"]
        source_url = item["source_url"]
        safe_source_name = item["safe_source_name"]

        for atype in item["annotation_types"]:
            started = time.time()
            success = False
            last_error = None
            out_file = output_dir / safe_source_name / annotations_filename(atype)
            csrf_token = None
            tried_preferred_chunk = False

            logger.info("[%s:%s] requesting annotations export", source_name, atype)

            if atype == "all" and args.prefer_chunked_all and not args.disable_chunked_fallback:
                tried_preferred_chunk = True
                csrf_token, csrf_err = fetch_source_csrf_token(
                    session=session,
                    source_url=source_url,
                    retries=args.retries,
                    timeout=args.timeout,
                    logger=logger,
                )
                if csrf_token is None:
                    last_error = csrf_err or "failed to get csrf token"
                else:
                    chunk_success, chunk_rows, chunk_error = export_annotations_chunked_by_image_id(
                        session=session,
                        source_name=source_name,
                        source_url=source_url,
                        source_dir=out_file.parent,
                        out_file=out_file,
                        csrf_token=csrf_token,
                        optional_columns=optional_columns,
                        export_retries=args.export_retries,
                        export_timeout=args.chunk_export_timeout or args.export_timeout,
                        max_wait_seconds=args.chunk_max_wait_seconds,
                        poll_interval_seconds=args.chunk_poll_interval_seconds,
                        chunk_size=args.chunk_size,
                        force=args.force,
                        logger=logger,
                    )
                    if chunk_success:
                        state.update_type(
                            source_name,
                            source_url,
                            atype,
                            {
                                "status": "done",
                                "annotations_path": str(out_file.resolve()),
                                "bytes": int(out_file.stat().st_size),
                                "rows": int(chunk_rows),
                                "chunked": True,
                                "error": None,
                            },
                            flush=True,
                        )
                        downloaded_files += 1
                        success = True
                    else:
                        last_error = chunk_error or "chunked export failed"

            while not success and not tried_preferred_chunk and time.time() - started <= args.max_wait_seconds:
                if csrf_token is None:
                    csrf_token, csrf_err = fetch_source_csrf_token(
                        session=session,
                        source_url=source_url,
                        retries=args.retries,
                        timeout=args.timeout,
                        logger=logger,
                    )
                    if csrf_token is None:
                        last_error = csrf_err or "failed to get csrf token"
                        logger.info(
                            "[%s:%s] source page not ready; retrying in %ss",
                            source_name,
                            atype,
                            args.poll_interval_seconds,
                        )
                        time.sleep(args.poll_interval_seconds)
                        continue

                status, file_content, export_err = export_annotations_once(
                    session=session,
                    source_url=source_url,
                    csrf_token=csrf_token,
                    optional_columns=optional_columns,
                    annotation_type=atype,
                    export_retries=args.export_retries,
                    export_timeout=args.export_timeout,
                    logger=logger,
                )

                if status == "success" and file_content is not None:
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_bytes(file_content)
                    state.update_type(
                        source_name,
                        source_url,
                        atype,
                        {
                            "status": "done",
                            "annotations_path": str(out_file.resolve()),
                            "bytes": int(len(file_content)),
                            "error": None,
                        },
                        flush=True,
                    )
                    downloaded_files += 1
                    success = True
                    break

                if status == "error":
                    last_error = export_err or "export failed"
                    break

                # status == not_ready
                last_error = export_err or "export not ready yet"
                logger.info("[%s:%s] annotations export not ready; retrying in %ss", source_name, atype, args.poll_interval_seconds)
                time.sleep(args.poll_interval_seconds)

            if not success:
                if atype == "all" and not args.disable_chunked_fallback and csrf_token is not None:
                    logger.info(
                        "[%s:%s] full export failed; trying chunked image_id_range fallback",
                        source_name,
                        atype,
                    )
                    chunk_success, chunk_rows, chunk_error = export_annotations_chunked_by_image_id(
                        session=session,
                        source_name=source_name,
                        source_url=source_url,
                        source_dir=out_file.parent,
                        out_file=out_file,
                        csrf_token=csrf_token,
                        optional_columns=optional_columns,
                        export_retries=args.export_retries,
                        export_timeout=args.chunk_export_timeout or args.export_timeout,
                        max_wait_seconds=args.chunk_max_wait_seconds,
                        poll_interval_seconds=args.chunk_poll_interval_seconds,
                        chunk_size=args.chunk_size,
                        force=args.force,
                        logger=logger,
                    )
                    if chunk_success:
                        state.update_type(
                            source_name,
                            source_url,
                            atype,
                            {
                                "status": "done",
                                "annotations_path": str(out_file.resolve()),
                                "bytes": int(out_file.stat().st_size),
                                "rows": int(chunk_rows),
                                "chunked": True,
                                "error": None,
                            },
                            flush=True,
                        )
                        downloaded_files += 1
                        success = True
                    else:
                        last_error = chunk_error or last_error

            if not success:
                failures.append(
                    {
                        "source_name": source_name,
                        "source_url": source_url,
                        "annotation_type": atype,
                        "annotations_path": str(out_file.resolve()),
                        "error": last_error or "max wait exceeded",
                    }
                )
                state.update_type(
                    source_name,
                    source_url,
                    atype,
                    {
                        "status": "failed",
                        "annotations_path": str(out_file.resolve()),
                        "error": last_error or "max wait exceeded",
                    },
                    flush=True,
                )

        processed_sources += 1

    state.flush()
    write_failures_csv(Path(args.failures_file), failures)

    summary = {
        "timestamp": utc_now_iso(),
        "input_csv": args.input_csv,
        "output_dir": str(Path(args.output_dir).resolve()),
        "annotation_types": annotation_types,
        "processed_sources": processed_sources,
        "downloaded_files": downloaded_files,
        "skipped_existing": skipped_existing,
        "skipped_done_state": skipped_done_state,
        "failed_files": len(failures),
        "resume": bool(args.resume),
        "force": bool(args.force),
        "export_timeout": args.export_timeout,
        "max_wait_seconds": args.max_wait_seconds,
        "chunked_fallback": not bool(args.disable_chunked_fallback),
        "chunk_size": args.chunk_size,
        "chunk_max_wait_seconds": args.chunk_max_wait_seconds,
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
