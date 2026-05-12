from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def scrape_classifier_info(session: requests.Session, source_url: str, output_dir: Path, timeout: int = 60) -> dict:
    response = session.get(source_url, timeout=timeout)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    heading = soup.find(["h1", "h2"])
    source_name = heading.get_text(" ", strip=True) if heading else ""
    source_id_match = re.search(r"/source/(\d+)/", source_url)
    source_id = source_id_match.group(1) if source_id_match else ""

    backend = {}
    backend_el = soup.find(id="backend-column")
    if backend_el:
        for li in backend_el.find_all("li"):
            text = li.get_text(" ", strip=True)
            if ":" in text:
                key, value = text.split(":", 1)
                backend[key.strip()] = value.strip()

    classifiers = []
    match = re.search(r"let\s+classifierPlotData\s*=\s*(\[.*?\]);", html, flags=re.S)
    if match:
        for item in ast.literal_eval(match.group(1)):
            classifiers.append(
                {
                    "classifier_nbr": item.get("x"),
                    "accuracy_percent": item.get("y"),
                    "trained_on_images": item.get("nimages"),
                    "train_time": item.get("traintime"),
                    "date": item.get("date"),
                    "global_id": item.get("pk"),
                }
            )

    result = {
        "source_id": source_id,
        "source_name": source_name,
        "source_url": source_url,
        "backend": backend,
        "classifiers": classifiers,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classifier_info.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "classifier_info.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "source_id",
            "source_name",
            "source_url",
            "classifier_nbr",
            "accuracy_percent",
            "trained_on_images",
            "train_time",
            "date",
            "global_id",
            "last_classifier_saved",
            "last_classifier_trained",
            "feature_extractor",
            "confidence_threshold",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for clf in classifiers:
            writer.writerow(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_url": source_url,
                    **clf,
                    "last_classifier_saved": backend.get("Last classifier saved", ""),
                    "last_classifier_trained": backend.get("Last classifier trained", ""),
                    "feature_extractor": backend.get("Feature extractor", ""),
                    "confidence_threshold": backend.get("Confidence threshold", ""),
                }
            )
    return result
