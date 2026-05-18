from __future__ import annotations

from pathlib import Path


def render_slurm_script(
    source: str,
    output_dir: Path,
    work_dir: Path | None = None,
    job_name: str = "coralnet-export",
    time: str = "47:30:00",
    cpus: int = 4,
    mem: str = "16G",
    conda_setup: str | None = None,
    conda_env: str | None = None,
    include: str = "labelset,metadata,annotations,classifier,covers,calcification",
    image_scope: str = "all",
    extra_args: str = "--resume",
) -> str:
    work_dir = work_dir or Path.cwd()
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_dir.resolve()}/coralnet_export_%j.out",
        f"#SBATCH --error={output_dir.resolve()}/coralnet_export_%j.err",
        f"#SBATCH --time={time}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        "",
        "set -euo pipefail",
        "",
        f"cd {work_dir.resolve()}",
        "",
    ]
    if conda_setup or conda_env:
        lines.extend([
            "set +u",
            f"source {conda_setup or '/home/$USER/miniconda3/etc/profile.d/conda.sh'}",
            f"conda activate {conda_env}" if conda_env else "",
            "set -u",
            "",
        ])
    lines.extend([
        "if [[ -f .env.coralnet ]]; then",
        "  set -a",
        "  source .env.coralnet",
        "  set +a",
        "fi",
        "",
        "coralnet-exporter download \\",
        f"  {source} \\",
        f"  --output-dir {output_dir.resolve()} \\",
        f"  --include {include} \\",
        f"  --image-scope {image_scope} \\",
        f"  --workers {cpus} \\",
        f"  {extra_args}",
        "",
    ])
    return "\n".join(line for line in lines if line is not None)
