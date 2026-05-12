from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://coralnet.ucsd.edu"


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    url: str


@dataclass
class SourceInfo:
    source_id: str
    url: str
    name: str
    total_images: int | None = None
    first_image_number: int | None = None


def normalize_source_ref(source: str) -> SourceRef:
    value = source.strip()
    if not value:
        raise ValueError("Source cannot be empty")

    if value.isdigit():
        source_id = value
    else:
        match = re.search(r"/source/(\d+)/?", value)
        if not match:
            raise ValueError("Source must be a CoralNet source ID or source URL")
        source_id = match.group(1)
    return SourceRef(source_id=source_id, url=f"{BASE_URL}/source/{source_id}/")


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"[\d,]+", text)
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def discover_source(session: requests.Session, source: str, timeout: int = 60) -> SourceInfo:
    ref = normalize_source_ref(source)
    response = session.get(ref.url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    heading = soup.find(["h1", "h2"])
    if heading and heading.get_text(strip=True):
        name = heading.get_text(" ", strip=True)
    elif soup.title:
        name = soup.title.get_text(" ", strip=True).replace("| CoralNet", "").strip()
    else:
        name = f"source_{ref.source_id}"

    total_images = None
    for text in soup.stripped_strings:
        low = text.lower()
        if low.startswith("total images"):
            total_images = _parse_int(text)
            break

    first_image_number = discover_first_image_id(session, ref.url, timeout=timeout)
    return SourceInfo(
        source_id=ref.source_id,
        url=ref.url,
        name=name,
        total_images=total_images,
        first_image_number=first_image_number,
    )


def discover_first_image_id(session: requests.Session, source_url: str, timeout: int = 60) -> int | None:
    browse_url = urljoin(source_url, "browse/images/")
    try:
        response = session.get(browse_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None
    match = re.search(r"/image/(\d+)/", response.text)
    return int(match.group(1)) if match else None


def write_single_source_csv(source_info: SourceInfo, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "Source Name": source_info.name,
        "Source Name Cleaned": source_info.name,
        "URL": source_info.url,
        "ImagesURL": urljoin(source_info.url, "browse/images/"),
        "FirstImageNumber": source_info.first_image_number,
        "Total Images": source_info.total_images,
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    return path
