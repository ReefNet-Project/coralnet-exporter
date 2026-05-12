from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .auth import resolve_credentials
from .calcification import export_calcification_rates, fetch_rate_table_choices, parse_optional_columns as parse_calcification_optional_columns
from .classifier import scrape_classifier_info
from .legacy import run_legacy_module
from .image_covers import export_image_covers
from .slurm import render_slurm_script
from .source import discover_source, normalize_source_ref, write_single_source_csv

console = Console()

DEFAULT_INCLUDE = "labelset,metadata,annotations,classifier"
ALL_EXPORTS = {
    "images",
    "labelset",
    "metadata",
    "annotations",
    "classifier",
    "covers",
    "calcification",
}


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "coralnet-exporter/0.1"})
    return session


def _login_session(session: requests.Session, username: str, password: str, timeout: int) -> None:
    from crawl_labelset import login
    import logging

    logger = logging.getLogger("coralnet_exporter_login")
    logger.addHandler(logging.NullHandler())
    login(session, username, password, retries=3, timeout=timeout, logger=logger)


def _parse_include(text: str) -> set[str]:
    values = {item.strip().lower() for item in text.split(",") if item.strip()}
    unknown = values - ALL_EXPORTS
    if unknown:
        raise click.BadParameter(f"Unknown exports: {', '.join(sorted(unknown))}")
    return values


def _source_output_dir(output_dir: Path, source_name: str) -> Path:
    import re

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", source_name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .") or "source"
    return output_dir / cleaned


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def app() -> None:
    """Export CoralNet sources and data products."""


@app.command("login")
@click.option("--username", default=None, help="CoralNet username.")
@click.option("--password", default=None, help="CoralNet password. Prefer env/prompt over CLI.")
@click.option("--env-file", type=click.Path(path_type=Path), default=Path(".env.coralnet"), help="Credentials env file.")
@click.option("--timeout", default=60, show_default=True, help="Request timeout seconds.")
def login_cmd(username: str | None, password: str | None, env_file: Path, timeout: int) -> None:
    """Validate CoralNet credentials."""
    creds = resolve_credentials(username=username, password=password, env_file=env_file, interactive=True, console=console)
    session = _session()
    _login_session(session, creds.username, creds.password, timeout)
    console.print("[green]Login successful.[/green]")


@app.command("download")
@click.argument("source", required=False)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("output"), show_default=True, help="Output directory.")
@click.option("--include", default=DEFAULT_INCLUDE, show_default=True, help=f"Comma-separated exports. Available: {', '.join(sorted(ALL_EXPORTS))}.")
@click.option("--username", default=None, help="CoralNet username.")
@click.option("--password", default=None, help="CoralNet password. Prefer env/prompt over CLI.")
@click.option("--env-file", type=click.Path(path_type=Path), default=Path(".env.coralnet"), help="Credentials env file.")
@click.option("--resume/--no-resume", default=True, show_default=True, help="Resume and skip completed files.")
@click.option("--force", is_flag=True, help="Force redownload where supported.")
@click.option("--background", is_flag=True, help="Run detached in the background.")
@click.option("--timeout", default=60, show_default=True, help="General request timeout seconds.")
@click.option("--export-timeout", default=600, show_default=True, help="Export request timeout seconds.")
@click.option("--chunk-size", default=500, show_default=True, help="Chunk size for large annotations_all exports.")
@click.option("--workers", default=4, show_default=True, help="Image downloader worker count.")
@click.option("--calcification-rate-table-id", default=None, help="Calcification rate table ID. If omitted, use the first table in CoralNet UI.")
@click.option("--calcification-label-display", type=click.Choice(["code", "name"]), default="code", show_default=True, help="Calcification label display mode.")
@click.option("--calcification-optional-columns", default="", show_default=True, help="Comma-separated: per_label_mean,per_label_bounds.")
def download_cmd(
    source: str | None,
    output_dir: Path,
    include: str,
    username: str | None,
    password: str | None,
    env_file: Path,
    resume: bool,
    force: bool,
    background: bool,
    timeout: int,
    export_timeout: int,
    chunk_size: int,
    workers: int,
    calcification_rate_table_id: str | None,
    calcification_label_display: str,
    calcification_optional_columns: str,
) -> None:
    """Download selected data products for one CoralNet source."""
    if source is None:
        source = Prompt.ask("CoralNet source ID or URL").strip()
    selected = _parse_include(include)

    if background:
        cmd = [
            sys.executable,
            "-m",
            "coralnet_exporter.cli",
            "download",
            source,
            "--output-dir",
            str(output_dir),
            "--include",
            include,
            "--env-file",
            str(env_file),
        ]
        cmd.append("--resume" if resume else "--no-resume")
        if force:
            cmd.append("--force")
        log_dir = output_dir / "_coralnet_exporter_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"source_{normalize_source_ref(source).source_id}.log"
        with log_path.open("ab") as log:
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        console.print(f"Started background job. Log: [bold]{log_path}[/bold]")
        console.print(f"Monitor with: tail -f {log_path}")
        return

    creds = resolve_credentials(username=username, password=password, env_file=env_file, interactive=True, console=console)
    session = _session()
    _login_session(session, creds.username, creds.password, timeout)
    source_info = discover_source(session, source, timeout=timeout)

    source_dir = _source_output_dir(output_dir, source_info.name)
    work_dir = source_dir / ".coralnet-exporter"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_csv = write_single_source_csv(source_info, work_dir / "source.csv")

    console.print(Panel.fit(f"Source {source_info.source_id}: {source_info.name}\nOutput: {source_dir}", title="CoralNet Exporter"))

    common_auth = ["--username", creds.username, "--password", creds.password]
    resume_flag = ["--resume"] if resume else []
    force_flag = ["--force"] if force else []

    if "classifier" in selected:
        console.print("[bold]Exporting classifier info[/bold]")
        scrape_classifier_info(session, source_info.url, source_dir, timeout=timeout)

    if "labelset" in selected:
        console.print("[bold]Exporting labelset[/bold]")
        run_legacy_module(
            "crawl_labelset",
            [
                "--input-csv", str(input_csv),
                "--output-dir", str(output_dir),
                "--state-file", str(work_dir / "labelset_state.json"),
                "--summary-file", str(work_dir / "labelset_summary.json"),
                "--failures-file", str(work_dir / "labelset_failures.csv"),
                "--log-file", str(work_dir / "labelset.log"),
                *resume_flag, *force_flag, *common_auth,
            ],
            console=console,
        )

    if "metadata" in selected:
        console.print("[bold]Exporting metadata[/bold]")
        run_legacy_module(
            "scrape_metadata",
            [
                "--input-csv", str(input_csv),
                "--output-dir", str(output_dir),
                "--metadata-types", "all,on_confirmed",
                "--state-file", str(work_dir / "metadata_state.json"),
                "--summary-file", str(work_dir / "metadata_summary.json"),
                "--failures-file", str(work_dir / "metadata_failures.csv"),
                "--log-file", str(work_dir / "metadata.log"),
                "--export-timeout", str(export_timeout),
                *resume_flag, *force_flag, *common_auth,
            ],
            console=console,
        )

    if "annotations" in selected:
        console.print("[bold]Exporting annotations[/bold]")
        run_legacy_module(
            "scrape_annotations",
            [
                "--input-csv", str(input_csv),
                "--output-dir", str(output_dir),
                "--annotation-types", "all,on_confirmed",
                "--state-file", str(work_dir / "annotations_state.json"),
                "--summary-file", str(work_dir / "annotations_summary.json"),
                "--failures-file", str(work_dir / "annotations_failures.csv"),
                "--log-file", str(work_dir / "annotations.log"),
                "--export-timeout", str(export_timeout),
                "--prefer-chunked-all",
                "--chunk-size", str(chunk_size),
                "--chunk-export-timeout", str(export_timeout),
                *resume_flag, *force_flag, *common_auth,
            ],
            console=console,
        )

    if "images" in selected:
        if source_info.first_image_number is None:
            console.print("[yellow]Skipping images: could not discover FirstImageNumber from browse/images page.[/yellow]")
        else:
            console.print("[bold]Downloading images[/bold]")
            run_legacy_module(
                "scrape",
                [
                    "--input-csv", str(input_csv),
                    "--output-dir", str(output_dir),
                    "--workers", str(workers),
                    "--state-file", str(work_dir / "images_state.json"),
                    "--summary-file", str(work_dir / "images_summary.json"),
                    "--failures-file", str(work_dir / "images_failures.csv"),
                    "--log-file", str(work_dir / "images.log"),
                    *resume_flag,
                ],
                console=console,
            )

    if "covers" in selected:
        console.print("[bold]Exporting percent cover / image covers[/bold]")
        export_image_covers(session, source_info.url, source_dir, timeout=export_timeout)

    if "calcification" in selected:
        console.print("[bold]Exporting calcification rates[/bold]")
        rate_table_id = calcification_rate_table_id
        if not rate_table_id:
            try:
                choices = fetch_rate_table_choices(session, source_info.url, timeout=timeout)
            except Exception as exc:
                raise click.ClickException(
                    "Could not discover calcification rate tables from the source page. "
                    "Rerun with --calcification-rate-table-id. "
                    f"Original error: {exc}"
                ) from exc
            if not choices:
                raise click.ClickException("No calcification rate tables were found for this source.")
            rate_table_id = choices[0].table_id
            console.print(f"[dim]Using calcification rate table {choices[0].table_id}: {choices[0].name}[/dim]")
        export_calcification_rates(
            session=session,
            source_url=source_info.url,
            output_dir=source_dir,
            rate_table_id=rate_table_id,
            timeout=export_timeout,
            label_display=calcification_label_display,
            optional_columns=parse_calcification_optional_columns(calcification_optional_columns),
        )

    console.print("[green]Export run finished.[/green]")
    console.print(f"Logs/state: {work_dir}")


@app.command("make-slurm")
@click.argument("source")
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=Path("run_coralnet_export.sbatch"), show_default=True, help="Path to write sbatch file.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("output"), show_default=True, help="Exporter output directory.")
@click.option("--include", default=DEFAULT_INCLUDE, show_default=True, help="Comma-separated exports.")
@click.option("--conda-setup", default="/home/$USER/miniconda3/etc/profile.d/conda.sh", show_default=True, help="Conda setup script path.")
@click.option("--conda-env", default=None, help="Conda env path/name.")
def make_slurm_cmd(source: str, output_path: Path, output_dir: Path, include: str, conda_setup: str | None, conda_env: str | None) -> None:
    """Generate a Slurm script for a CoralNet export."""
    script = render_slurm_script(
        source=source,
        output_dir=output_dir,
        conda_setup=conda_setup,
        conda_env=conda_env,
        include=include,
    )
    output_path.write_text(script, encoding="utf-8")
    console.print(f"Wrote [bold]{output_path}[/bold]")


if __name__ == "__main__":
    app()
