import argparse
import csv
import hashlib
import json
import logging
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DEFAULT_INPUT_CSV = "coralnet_sources_data_04_13_2026/coralnet_sources_more_than_250_with_images.csv"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_LOG_FILE = "scrape_logs.log"
DEFAULT_STATE_FILE = "scrape_progress_state.json"
DEFAULT_SUMMARY_FILE = "scrape_summary.json"
DEFAULT_FAILURES_FILE = "scrape_failures.csv"

BASE_URL = "https://coralnet.ucsd.edu"
LOGIN_URL = f"{BASE_URL}/accounts/login/"


@dataclass
class SourceResult:
    source_key: str
    source_name: str
    safe_source_name: str
    status: str
    expected_total: int | None
    local_count: int
    downloaded_in_run: int
    skipped_existing_in_run: int
    pages_visited: int
    last_page_url: str | None
    error: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def sanitize_filename(name: str, default: str = "unnamed") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", str(name)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .")
    return cleaned or default


def normalize_source_name(row: pd.Series) -> str:
    for col in ("Source Name Cleaned", "Source Name", "Source"):
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col].strip()
    return f"source_row_{int(row.name)}"


def stable_source_key(source_name: str, source_url: str | None) -> str:
    raw = f"{source_name}|{source_url or ''}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{sanitize_filename(source_name, default='source')}__{digest}"


def source_shard(source_key: str, shard_count: int) -> int:
    if shard_count <= 1:
        return 0
    digest = hashlib.md5(source_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % shard_count


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "CoralNetScraper/2.0"})
    return session


def request_with_retries(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int,
    logger: logging.Logger,
    source_name: str,
    kind: str,
    method: str = "GET",
    **kwargs: Any,
) -> tuple[requests.Response | None, str | None]:
    error_msg = None
    kwargs.setdefault("allow_redirects", True)
    for attempt in range(1, retries + 1):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            return resp, None
        except requests.RequestException as exc:
            error_msg = f"{kind} request failed attempt {attempt}/{retries} for {source_name}: {exc}"
            if attempt == retries:
                logger.error(error_msg)
                return None, error_msg
            logger.warning(error_msg)
            time.sleep(min(2 ** (attempt - 1), 5))
    return None, error_msg


def login(
    session: requests.Session,
    username: str,
    password: str,
    timeout: int,
    retries: int,
    logger: logging.Logger,
    source_name: str,
) -> None:
    login_page, err = request_with_retries(
        session=session,
        url=LOGIN_URL,
        timeout=timeout,
        retries=retries,
        logger=logger,
        source_name=source_name,
        kind="login-page",
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
        url=LOGIN_URL,
        timeout=timeout,
        retries=retries,
        logger=logger,
        source_name=source_name,
        kind="login",
        method="POST",
        data=payload,
        headers=headers,
    )
    if response is None:
        raise RuntimeError(err or "login request failed")

    response_text = response.text.lower()
    if "accounts/login" in str(response.url).lower() and "logout" not in response_text:
        raise RuntimeError("login failed; check CoralNet credentials")


def extract_extension_from_url(url: str | None) -> str | None:
    if not url:
        return None
    path = urlparse(unquote(url)).path
    match = re.search(r"\.([A-Za-z0-9]{1,6})$", path)
    if not match:
        return None
    return match.group(1).lower()


def extract_image_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/image/(\d+)/", url)
    return match.group(1) if match else None


def ensure_ext(filename: str, ext: str) -> str:
    if re.search(r"\.[A-Za-z0-9]{1,6}$", filename):
        return filename
    return f"{filename}.{ext}"


def build_local_image_name(
    image_name: str,
    image_url: str,
    image_id: str | None,
    content_type: str | None,
) -> str:
    base_name = sanitize_filename(image_name, default=image_id or "image")
    base_name = ensure_ext(base_name, ext=extract_extension_from_url(image_url) or "jpg")

    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            guessed = guessed.lstrip(".").lower()
            if not re.search(r"\.[A-Za-z0-9]{1,6}$", base_name):
                base_name = f"{base_name}.{guessed}"

    if image_id:
        return f"{image_id}__{base_name}"
    return base_name


def parse_image_page(html: bytes, page_url: str) -> dict[str, str | None]:
    soup = BeautifulSoup(html, "html.parser")

    header_div = soup.find("div", id="header")
    image_name = None
    if header_div:
        h2 = header_div.find("h2")
        if h2:
            anchors = h2.find_all("a")
            if anchors:
                # CoralNet commonly has [source, image_name]
                image_name = anchors[-1].get_text(strip=True)

    container = soup.find("div", id="original_image_container")
    img_element = container.find("img") if container else None
    image_url = img_element.get("src") if img_element else None
    if image_url:
        image_url = image_url.replace("&amp;", "&")
        image_url = image_url.replace('" /', "")

    detail_list = soup.find("ul", class_="detail_list")
    next_url = None
    if detail_list:
        next_anchor = detail_list.find("a", string=lambda text: text and "Next" in text)
        if next_anchor and next_anchor.get("href"):
            next_url = urljoin(BASE_URL, next_anchor.get("href"))

    image_id = extract_image_id_from_url(page_url)

    return {
        "image_name": image_name,
        "image_url": image_url,
        "next_url": next_url,
        "image_id": image_id,
    }


class ProgressStore:
    def __init__(self, state_path: Path, resume: bool, reset_state: bool):
        self.state_path = state_path
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "meta": {
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            },
            "sources": {},
        }

        if state_path.exists() and resume and not reset_state:
            try:
                self.state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Keep going with a fresh state.
                pass

    def _atomic_write_locked(self) -> None:
        self.state.setdefault("meta", {})["updated_at"] = utc_now_iso()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp_path.replace(self.state_path)

    def upsert_source(self, source_key: str, updates: dict[str, Any], flush: bool = False) -> dict[str, Any]:
        with self.lock:
            sources = self.state.setdefault("sources", {})
            entry = sources.setdefault(source_key, {})
            entry.update(updates)
            entry["updated_at"] = utc_now_iso()
            if flush:
                self._atomic_write_locked()
            return dict(entry)

    def get_source(self, source_key: str) -> dict[str, Any]:
        with self.lock:
            return dict(self.state.get("sources", {}).get(source_key, {}))

    def flush(self) -> None:
        with self.lock:
            self._atomic_write_locked()


class TqdmSlotManager:
    def __init__(self, workers: int):
        self.slots: Queue[int] = Queue()
        for pos in range(1, workers + 1):
            self.slots.put(pos)

    def acquire(self) -> int:
        return self.slots.get()

    def release(self, position: int) -> None:
        self.slots.put(position)


def setup_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("scrape")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def process_source(
    row: pd.Series,
    output_dir: Path,
    progress_store: ProgressStore,
    logger: logging.Logger,
    slot_manager: TqdmSlotManager | None,
    show_source_progress: bool,
    timeout: int,
    retries: int,
    strict_image_validation: bool,
    checkpoint_every: int,
    max_images_per_source: int | None,
    source_max_failures: int,
    username: str | None,
    password: str | None,
) -> SourceResult:
    source_name = normalize_source_name(row)
    source_url = row.get("URL") if "URL" in row else None
    source_key = stable_source_key(source_name=source_name, source_url=source_url if isinstance(source_url, str) else None)
    safe_source_name = sanitize_filename(source_name, default=source_key)

    expected_total = parse_int(row.get("Total Images"))
    first_image_number = parse_int(row.get("FirstImageNumber"))

    source_dir = output_dir / safe_source_name / "images"
    source_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [p for p in source_dir.iterdir() if p.is_file()]
    existing_count = len(existing_files)

    entry = progress_store.get_source(source_key)
    attempts = int(entry.get("attempts", 0)) + 1

    progress_store.upsert_source(
        source_key,
        {
            "source_name": source_name,
            "safe_source_name": safe_source_name,
            "source_url": source_url,
            "expected_total": expected_total,
            "first_image_number": first_image_number,
            "status": "in_progress",
            "attempts": attempts,
            "downloaded_files": existing_count,
            "last_error": None,
        },
        flush=True,
    )

    if attempts > source_max_failures:
        msg = f"source exceeded max attempts ({source_max_failures})"
        progress_store.upsert_source(
            source_key,
            {
                "status": "failed",
                "last_error": msg,
            },
            flush=True,
        )
        return SourceResult(
            source_key=source_key,
            source_name=source_name,
            safe_source_name=safe_source_name,
            status="failed",
            expected_total=expected_total,
            local_count=existing_count,
            downloaded_in_run=0,
            skipped_existing_in_run=0,
            pages_visited=0,
            last_page_url=entry.get("last_page_url"),
            error=msg,
        )

    if expected_total is not None and existing_count >= expected_total:
        progress_store.upsert_source(
            source_key,
            {
                "status": "done",
                "downloaded_files": existing_count,
                "last_page_url": None,
                "last_error": None,
            },
            flush=True,
        )
        logger.info("[%s] already complete locally (%d/%d)", source_name, existing_count, expected_total)
        return SourceResult(
            source_key=source_key,
            source_name=source_name,
            safe_source_name=safe_source_name,
            status="done",
            expected_total=expected_total,
            local_count=existing_count,
            downloaded_in_run=0,
            skipped_existing_in_run=0,
            pages_visited=0,
            last_page_url=None,
            error=None,
        )

    if entry.get("last_page_url"):
        page_url = str(entry.get("last_page_url"))
    else:
        if first_image_number is None:
            msg = "missing FirstImageNumber"
            progress_store.upsert_source(
                source_key,
                {
                    "status": "failed",
                    "last_error": msg,
                },
                flush=True,
            )
            return SourceResult(
                source_key=source_key,
                source_name=source_name,
                safe_source_name=safe_source_name,
                status="failed",
                expected_total=expected_total,
                local_count=existing_count,
                downloaded_in_run=0,
                skipped_existing_in_run=0,
                pages_visited=0,
                last_page_url=None,
                error=msg,
            )
        page_url = f"{BASE_URL}/image/{first_image_number}/view/"

    downloaded_in_run = 0
    skipped_existing_in_run = 0
    pages_visited = 0
    visited_urls: set[str] = set()

    session = make_session()
    if username and password:
        try:
            login(
                session=session,
                username=username,
                password=password,
                timeout=timeout,
                retries=retries,
                logger=logger,
                source_name=source_name,
            )
            logger.info("[%s] authenticated with CoralNet", source_name)
        except RuntimeError as exc:
            msg = f"authentication failed: {exc}"
            progress_store.upsert_source(
                source_key,
                {
                    "status": "failed",
                    "last_page_url": page_url,
                    "last_error": msg,
                    "downloaded_files": existing_count,
                },
                flush=True,
            )
            return SourceResult(
                source_key=source_key,
                source_name=source_name,
                safe_source_name=safe_source_name,
                status="failed",
                expected_total=expected_total,
                local_count=existing_count,
                downloaded_in_run=0,
                skipped_existing_in_run=0,
                pages_visited=0,
                last_page_url=page_url,
                error=msg,
            )
    else:
        logger.warning("[%s] no CoralNet credentials provided; downloading unauthenticated", source_name)

    source_bar = None
    source_bar_slot = None
    if show_source_progress and slot_manager is not None:
        source_bar_slot = slot_manager.acquire()
        total_for_bar = expected_total if expected_total is not None and expected_total > 0 else None
        initial_for_bar = min(existing_count, total_for_bar) if total_for_bar is not None else 0
        source_bar = tqdm(
            total=total_for_bar,
            initial=initial_for_bar,
            desc=f"{safe_source_name[:32]}",
            unit="img",
            position=source_bar_slot,
            leave=False,
            dynamic_ncols=True,
        )

    try:
        while page_url:
            if page_url in visited_urls:
                msg = f"detected page loop at {page_url}"
                progress_store.upsert_source(
                    source_key,
                    {
                        "status": "failed",
                        "last_page_url": page_url,
                        "last_error": msg,
                        "downloaded_files": len([p for p in source_dir.iterdir() if p.is_file()]),
                    },
                    flush=True,
                )
                return SourceResult(
                    source_key=source_key,
                    source_name=source_name,
                    safe_source_name=safe_source_name,
                    status="failed",
                    expected_total=expected_total,
                    local_count=len([p for p in source_dir.iterdir() if p.is_file()]),
                    downloaded_in_run=downloaded_in_run,
                    skipped_existing_in_run=skipped_existing_in_run,
                    pages_visited=pages_visited,
                    last_page_url=page_url,
                    error=msg,
                )
            visited_urls.add(page_url)

            page_resp, page_error = request_with_retries(
                session=session,
                url=page_url,
                timeout=timeout,
                retries=retries,
                logger=logger,
                source_name=source_name,
                kind="page",
            )
            if page_resp is None:
                progress_store.upsert_source(
                    source_key,
                    {
                        "status": "failed",
                        "last_page_url": page_url,
                        "last_error": page_error,
                        "downloaded_files": len([p for p in source_dir.iterdir() if p.is_file()]),
                    },
                    flush=True,
                )
                return SourceResult(
                    source_key=source_key,
                    source_name=source_name,
                    safe_source_name=safe_source_name,
                    status="failed",
                    expected_total=expected_total,
                    local_count=len([p for p in source_dir.iterdir() if p.is_file()]),
                    downloaded_in_run=downloaded_in_run,
                    skipped_existing_in_run=skipped_existing_in_run,
                    pages_visited=pages_visited,
                    last_page_url=page_url,
                    error=page_error,
                )

            if page_resp.status_code != 200:
                msg = f"page returned HTTP {page_resp.status_code} for {page_url}"
                progress_store.upsert_source(
                    source_key,
                    {
                        "status": "failed",
                        "last_page_url": page_url,
                        "last_error": msg,
                        "downloaded_files": len([p for p in source_dir.iterdir() if p.is_file()]),
                    },
                    flush=True,
                )
                return SourceResult(
                    source_key=source_key,
                    source_name=source_name,
                    safe_source_name=safe_source_name,
                    status="failed",
                    expected_total=expected_total,
                    local_count=len([p for p in source_dir.iterdir() if p.is_file()]),
                    downloaded_in_run=downloaded_in_run,
                    skipped_existing_in_run=skipped_existing_in_run,
                    pages_visited=pages_visited,
                    last_page_url=page_url,
                    error=msg,
                )

            parsed = parse_image_page(page_resp.content, page_url=page_url)
            image_url = parsed.get("image_url")
            image_name = parsed.get("image_name") or parsed.get("image_id") or "image"
            next_url = parsed.get("next_url")
            image_id = parsed.get("image_id")

            if not image_url:
                msg = f"missing image URL on page {page_url}"
                progress_store.upsert_source(
                    source_key,
                    {
                        "status": "failed",
                        "last_page_url": page_url,
                        "last_error": msg,
                        "downloaded_files": len([p for p in source_dir.iterdir() if p.is_file()]),
                    },
                    flush=True,
                )
                return SourceResult(
                    source_key=source_key,
                    source_name=source_name,
                    safe_source_name=safe_source_name,
                    status="failed",
                    expected_total=expected_total,
                    local_count=len([p for p in source_dir.iterdir() if p.is_file()]),
                    downloaded_in_run=downloaded_in_run,
                    skipped_existing_in_run=skipped_existing_in_run,
                    pages_visited=pages_visited,
                    last_page_url=page_url,
                    error=msg,
                )

            local_name = build_local_image_name(
                image_name=image_name,
                image_url=image_url,
                image_id=image_id,
                content_type=None,
            )
            image_path = source_dir / local_name

            if image_path.exists():
                skipped_existing_in_run += 1
            else:
                img_resp, img_error = request_with_retries(
                    session=session,
                    url=image_url,
                    timeout=timeout,
                    retries=retries,
                    logger=logger,
                    source_name=source_name,
                    kind="image",
                )
                if img_resp is None:
                    progress_store.upsert_source(
                        source_key,
                        {
                            "status": "failed",
                            "last_page_url": page_url,
                            "last_error": img_error,
                            "downloaded_files": len([p for p in source_dir.iterdir() if p.is_file()]),
                        },
                        flush=True,
                    )
                    return SourceResult(
                        source_key=source_key,
                        source_name=source_name,
                        safe_source_name=safe_source_name,
                        status="failed",
                        expected_total=expected_total,
                        local_count=len([p for p in source_dir.iterdir() if p.is_file()]),
                        downloaded_in_run=downloaded_in_run,
                        skipped_existing_in_run=skipped_existing_in_run,
                        pages_visited=pages_visited,
                        last_page_url=page_url,
                        error=img_error,
                    )

                if img_resp.status_code != 200:
                    msg = f"image returned HTTP {img_resp.status_code} for {image_url}"
                    progress_store.upsert_source(
                        source_key,
                        {
                            "status": "failed",
                            "last_page_url": page_url,
                            "last_error": msg,
                            "downloaded_files": len([p for p in source_dir.iterdir() if p.is_file()]),
                        },
                        flush=True,
                    )
                    return SourceResult(
                        source_key=source_key,
                        source_name=source_name,
                        safe_source_name=safe_source_name,
                        status="failed",
                        expected_total=expected_total,
                        local_count=len([p for p in source_dir.iterdir() if p.is_file()]),
                        downloaded_in_run=downloaded_in_run,
                        skipped_existing_in_run=skipped_existing_in_run,
                        pages_visited=pages_visited,
                        last_page_url=page_url,
                        error=msg,
                    )

                content_type = img_resp.headers.get("Content-Type", "")
                if strict_image_validation and not content_type.lower().startswith("image/"):
                    msg = f"non-image content-type '{content_type}' for {image_url}"
                    progress_store.upsert_source(
                        source_key,
                        {
                            "status": "failed",
                            "last_page_url": page_url,
                            "last_error": msg,
                            "downloaded_files": len([p for p in source_dir.iterdir() if p.is_file()]),
                        },
                        flush=True,
                    )
                    return SourceResult(
                        source_key=source_key,
                        source_name=source_name,
                        safe_source_name=safe_source_name,
                        status="failed",
                        expected_total=expected_total,
                        local_count=len([p for p in source_dir.iterdir() if p.is_file()]),
                        downloaded_in_run=downloaded_in_run,
                        skipped_existing_in_run=skipped_existing_in_run,
                        pages_visited=pages_visited,
                        last_page_url=page_url,
                        error=msg,
                    )

                local_name = build_local_image_name(
                    image_name=image_name,
                    image_url=image_url,
                    image_id=image_id,
                    content_type=content_type,
                )
                image_path = source_dir / local_name
                tmp_path = image_path.with_suffix(image_path.suffix + ".part")
                tmp_path.write_bytes(img_resp.content)
                tmp_path.replace(image_path)
                downloaded_in_run += 1

            pages_visited += 1
            current_local_count = len([p for p in source_dir.iterdir() if p.is_file()])

            if source_bar is not None:
                if source_bar.total is None:
                    source_bar.update(1)
                elif source_bar.n < source_bar.total:
                    source_bar.update(min(1, source_bar.total - source_bar.n))
                if pages_visited % 25 == 0:
                    source_bar.set_postfix(new=downloaded_in_run, skip=skipped_existing_in_run)

            should_flush = pages_visited % max(checkpoint_every, 1) == 0
            progress_store.upsert_source(
                source_key,
                {
                    "status": "in_progress",
                    "last_page_url": next_url,
                    "downloaded_files": current_local_count,
                    "downloaded_in_run": downloaded_in_run,
                    "skipped_existing_in_run": skipped_existing_in_run,
                    "pages_visited": pages_visited,
                    "last_error": None,
                },
                flush=should_flush,
            )

            if max_images_per_source is not None and pages_visited >= max_images_per_source:
                logger.info("[%s] reached max-images-per-source=%d", source_name, max_images_per_source)
                progress_store.upsert_source(
                    source_key,
                    {
                        "status": "paused",
                        "last_page_url": next_url,
                        "downloaded_files": current_local_count,
                        "last_error": None,
                    },
                    flush=True,
                )
                return SourceResult(
                    source_key=source_key,
                    source_name=source_name,
                    safe_source_name=safe_source_name,
                    status="paused",
                    expected_total=expected_total,
                    local_count=current_local_count,
                    downloaded_in_run=downloaded_in_run,
                    skipped_existing_in_run=skipped_existing_in_run,
                    pages_visited=pages_visited,
                    last_page_url=next_url,
                    error=None,
                )

            if expected_total is not None and current_local_count >= expected_total:
                progress_store.upsert_source(
                    source_key,
                    {
                        "status": "done",
                        "last_page_url": None,
                        "downloaded_files": current_local_count,
                        "last_error": None,
                    },
                    flush=True,
                )
                logger.info("[%s] complete by expected total (%d/%d)", source_name, current_local_count, expected_total)
                return SourceResult(
                    source_key=source_key,
                    source_name=source_name,
                    safe_source_name=safe_source_name,
                    status="done",
                    expected_total=expected_total,
                    local_count=current_local_count,
                    downloaded_in_run=downloaded_in_run,
                    skipped_existing_in_run=skipped_existing_in_run,
                    pages_visited=pages_visited,
                    last_page_url=None,
                    error=None,
                )

            if not next_url:
                progress_store.upsert_source(
                    source_key,
                    {
                        "status": "done",
                        "last_page_url": None,
                        "downloaded_files": current_local_count,
                        "last_error": None,
                    },
                    flush=True,
                )
                logger.info("[%s] reached final page (downloaded=%d)", source_name, current_local_count)
                return SourceResult(
                    source_key=source_key,
                    source_name=source_name,
                    safe_source_name=safe_source_name,
                    status="done",
                    expected_total=expected_total,
                    local_count=current_local_count,
                    downloaded_in_run=downloaded_in_run,
                    skipped_existing_in_run=skipped_existing_in_run,
                    pages_visited=pages_visited,
                    last_page_url=None,
                    error=None,
                )

            page_url = next_url

        # Defensive fallback.
        current_local_count = len([p for p in source_dir.iterdir() if p.is_file()])
        progress_store.upsert_source(
            source_key,
            {
                "status": "failed",
                "last_page_url": page_url,
                "downloaded_files": current_local_count,
                "last_error": "unexpected loop termination",
            },
            flush=True,
        )
        return SourceResult(
            source_key=source_key,
            source_name=source_name,
            safe_source_name=safe_source_name,
            status="failed",
            expected_total=expected_total,
            local_count=current_local_count,
            downloaded_in_run=downloaded_in_run,
            skipped_existing_in_run=skipped_existing_in_run,
            pages_visited=pages_visited,
            last_page_url=page_url,
            error="unexpected loop termination",
        )
    finally:
        if source_bar is not None:
            source_bar.close()
        if source_bar_slot is not None and slot_manager is not None:
            slot_manager.release(source_bar_slot)


def write_failures_csv(path: Path, results: list[SourceResult]) -> None:
    failures = [r for r in results if r.status == "failed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_key",
                "source_name",
                "safe_source_name",
                "status",
                "expected_total",
                "local_count",
                "downloaded_in_run",
                "skipped_existing_in_run",
                "pages_visited",
                "last_page_url",
                "error",
            ],
        )
        writer.writeheader()
        for r in failures:
            writer.writerow(
                {
                    "source_key": r.source_key,
                    "source_name": r.source_name,
                    "safe_source_name": r.safe_source_name,
                    "status": r.status,
                    "expected_total": r.expected_total,
                    "local_count": r.local_count,
                    "downloaded_in_run": r.downloaded_in_run,
                    "skipped_existing_in_run": r.skipped_existing_in_run,
                    "pages_visited": r.pages_visited,
                    "last_page_url": r.last_page_url,
                    "error": r.error,
                }
            )


def build_summary(
    results: list[SourceResult],
    args: argparse.Namespace,
    started_at: float,
    finished_at: float,
) -> dict[str, Any]:
    total_sources = len(results)
    done_sources = sum(1 for r in results if r.status == "done")
    failed_sources = sum(1 for r in results if r.status == "failed")
    paused_sources = sum(1 for r in results if r.status == "paused")

    return {
        "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "finished_at": datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat(),
        "duration_seconds": round(finished_at - started_at, 2),
        "input_csv": args.input_csv,
        "output_dir": args.output_dir,
        "workers": args.workers,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "strict_image_validation": args.strict_image_validation,
        "sources_total": total_sources,
        "sources_done": done_sources,
        "sources_failed": failed_sources,
        "sources_paused": paused_sources,
        "images_downloaded_in_run": sum(r.downloaded_in_run for r in results),
        "images_skipped_existing_in_run": sum(r.skipped_existing_in_run for r in results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CoralNet source images with resume + reporting.")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output root directory")
    parser.add_argument(
        "--username",
        default=os.getenv("CORALNET_USERNAME") or os.getenv("USERNAME"),
        help="CoralNet username. Defaults to CORALNET_USERNAME, then USERNAME.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("CORALNET_PASSWORD") or os.getenv("PASSWORD"),
        help="CoralNet password. Defaults to CORALNET_PASSWORD, then PASSWORD.",
    )
    parser.add_argument("--workers", type=int, default=5, help="Number of source worker threads")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=4, help="Retries per request")
    parser.add_argument(
        "--source-max-failures",
        type=int,
        default=8,
        help="Stop retrying a source after this many failed runs",
    )
    parser.add_argument(
        "--strict-image-validation",
        action="store_true",
        help="Fail image downloads when content-type is not image/*",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Flush progress state every N visited pages per source",
    )
    parser.add_argument(
        "--max-images-per-source",
        type=int,
        default=None,
        help="Debug mode: stop each source after N pages",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="Log file path")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="Resume state JSON path")
    parser.add_argument("--summary-file", default=DEFAULT_SUMMARY_FILE, help="Run summary JSON path")
    parser.add_argument("--failures-file", default=DEFAULT_FAILURES_FILE, help="Failed sources CSV path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing state file")
    parser.add_argument("--reset-state", action="store_true", help="Ignore existing state and start fresh")
    parser.add_argument(
        "--no-update-source-csv",
        action="store_true",
        help="Do not write DownloadedSourcePath back to a CSV file",
    )
    parser.add_argument(
        "--source-csv-output",
        default=None,
        help="Optional CSV output path for writing DownloadedSourcePath (default: input CSV; sharded runs default to shard-suffixed copy)",
    )
    parser.add_argument(
        "--no-source-progress",
        action="store_true",
        help="Disable per-source image progress bars",
    )
    parser.add_argument(
        "--max-sources-per-run",
        type=int,
        default=None,
        help="Process only the next N not-yet-done sources (in CSV order)",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of shards (for multi-machine runs)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Current shard index in [0, shard-count-1]",
    )
    return parser.parse_args()


def shard_suffix(shard_index: int, shard_count: int) -> str:
    return f"_shard{shard_index}_of_{shard_count}"


def resolve_default_sharded_path(path_value: str, default_value: str, suffix: str) -> str:
    if path_value != default_value:
        return path_value
    path = Path(path_value)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def count_local_images(source_dir: Path) -> int:
    if not source_dir.exists():
        return 0
    return sum(1 for p in source_dir.iterdir() if p.is_file())


def write_download_paths_to_csv(
    df: pd.DataFrame,
    output_csv_path: Path,
    output_dir: Path,
    shard_count: int,
    shard_index: int,
) -> int:
    out_df = df.copy()
    if "DownloadedSourcePath" not in out_df.columns:
        out_df["DownloadedSourcePath"] = pd.NA

    updated_rows = 0
    for idx, row in out_df.iterrows():
        source_name = normalize_source_name(row)
        source_url = row.get("URL") if "URL" in row else None
        source_key = stable_source_key(source_name, source_url if isinstance(source_url, str) else None)
        if source_shard(source_key, shard_count) != shard_index:
            continue

        safe_source_name = sanitize_filename(source_name, default=source_key)
        source_dir = (output_dir / safe_source_name / "images").resolve()
        if count_local_images(source_dir) > 0:
            out_df.at[idx, "DownloadedSourcePath"] = str(source_dir)
            updated_rows += 1
        else:
            out_df.at[idx, "DownloadedSourcePath"] = pd.NA

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    return updated_rows


def main() -> int:
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.max_sources_per_run is not None and args.max_sources_per_run < 1:
        raise ValueError("--max-sources-per-run must be >= 1")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")

    if args.shard_count > 1:
        suffix = shard_suffix(args.shard_index, args.shard_count)
        args.log_file = resolve_default_sharded_path(args.log_file, DEFAULT_LOG_FILE, suffix)
        args.state_file = resolve_default_sharded_path(args.state_file, DEFAULT_STATE_FILE, suffix)
        args.summary_file = resolve_default_sharded_path(args.summary_file, DEFAULT_SUMMARY_FILE, suffix)
        args.failures_file = resolve_default_sharded_path(args.failures_file, DEFAULT_FAILURES_FILE, suffix)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=Path(args.log_file), level=args.log_level)
    logger.info("Starting CoralNet scrape")
    logger.info(
        "Config: workers=%d shard=%d/%d timeout=%ds retries=%d strict_validation=%s authenticated=%s",
        args.workers,
        args.shard_index,
        args.shard_count,
        args.timeout,
        args.retries,
        args.strict_image_validation,
        bool(args.username and args.password),
    )

    df = pd.read_csv(args.input_csv)
    logger.info("Loaded %d rows from %s", len(df), args.input_csv)

    required_cols = ["FirstImageNumber"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    progress_store = ProgressStore(
        state_path=Path(args.state_file),
        resume=args.resume,
        reset_state=args.reset_state,
    )
    slot_manager = TqdmSlotManager(args.workers) if not args.no_source_progress else None

    tasks: list[pd.Series] = []
    skipped_done_by_state = 0
    skipped_done_by_disk = 0
    for _, row in df.iterrows():
        source_name = normalize_source_name(row)
        source_url = row.get("URL") if "URL" in row else None
        source_key = stable_source_key(source_name, source_url if isinstance(source_url, str) else None)
        if source_shard(source_key, args.shard_count) == args.shard_index:
            entry = progress_store.get_source(source_key)
            if entry.get("status") == "done":
                skipped_done_by_state += 1
                continue

            expected_total = parse_int(row.get("Total Images"))
            safe_source_name = sanitize_filename(source_name, default=source_key)
            source_dir = output_dir / safe_source_name / "images"
            local_count = count_local_images(source_dir)
            if expected_total is not None and local_count >= expected_total:
                skipped_done_by_disk += 1
                progress_store.upsert_source(
                    source_key,
                    {
                        "source_name": source_name,
                        "safe_source_name": safe_source_name,
                        "source_url": source_url,
                        "expected_total": expected_total,
                        "status": "done",
                        "downloaded_files": local_count,
                        "last_page_url": None,
                        "last_error": None,
                    },
                    flush=False,
                )
                continue

            tasks.append(row)
            if args.max_sources_per_run is not None and len(tasks) >= args.max_sources_per_run:
                break

    if skipped_done_by_disk > 0:
        progress_store.flush()

    logger.info(
        "Shard selected %d sources (skipped done: state=%d, disk=%d)",
        len(tasks),
        skipped_done_by_state,
        skipped_done_by_disk,
    )
    if args.max_sources_per_run is not None:
        logger.info("Applied run source limit: %d", args.max_sources_per_run)

    started_at = time.time()
    results: list[SourceResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                process_source,
                row,
                output_dir,
                progress_store,
                logger,
                slot_manager,
                not args.no_source_progress,
                args.timeout,
                args.retries,
                args.strict_image_validation,
                args.checkpoint_every,
                args.max_images_per_source,
                args.source_max_failures,
                args.username,
                args.password,
            )
            for row in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Sources",
            unit="source",
            position=0,
            dynamic_ncols=True,
        ):
            result = future.result()
            results.append(result)
            if result.status == "failed":
                logger.error("[%s] failed: %s", result.source_name, result.error)
            elif result.status == "done":
                logger.info(
                    "[%s] done (local=%d, +%d new, %d existing skipped)",
                    result.source_name,
                    result.local_count,
                    result.downloaded_in_run,
                    result.skipped_existing_in_run,
                )
            else:
                logger.info(
                    "[%s] %s (local=%d, +%d new)",
                    result.source_name,
                    result.status,
                    result.local_count,
                    result.downloaded_in_run,
                )

    progress_store.flush()

    finished_at = time.time()

    summary = build_summary(results=results, args=args, started_at=started_at, finished_at=finished_at)
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_failures_csv(Path(args.failures_file), results)

    if not args.no_update_source_csv:
        if args.source_csv_output:
            source_csv_output_path = Path(args.source_csv_output)
        else:
            source_csv_output_path = Path(args.input_csv)
            if args.shard_count > 1:
                suffix = shard_suffix(args.shard_index, args.shard_count)
                source_csv_output_path = Path(
                    resolve_default_sharded_path(
                        str(source_csv_output_path),
                        str(Path(args.input_csv)),
                        suffix,
                    )
                )

        updated_rows = write_download_paths_to_csv(
            df=df,
            output_csv_path=source_csv_output_path,
            output_dir=output_dir,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
        logger.info(
            "Updated DownloadedSourcePath for %d rows in %s",
            updated_rows,
            str(source_csv_output_path),
        )

    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("State file: %s", args.state_file)
    logger.info("Summary file: %s", args.summary_file)
    logger.info("Failures file: %s", args.failures_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
