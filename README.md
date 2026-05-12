# CoralNet Exporter

`coralnet-exporter` is a Python command-line tool for downloading and exporting CoralNet source data into a reproducible local folder structure.

It is designed for researchers, reef monitoring teams, machine learning engineers, and coral reef image analysis projects that need to export CoralNet images, annotations, metadata, labelsets, classifier information, percent cover, and calcification rates from public or authenticated private CoralNet sources.

## What This Tool Downloads

For a CoralNet source ID or URL, `coralnet-exporter` can download:

- Source images from public or private CoralNet sources
- Labelsets as `labelset.csv`
- Metadata for all images as `metadata_all.csv`
- Metadata for confirmed images as `metadata_confirmed.csv`
- Annotations for all images as `annotations_all.csv`
- Annotations for confirmed images as `annotations_confirmed.csv`
- Classifier/backend information as CSV and JSON
- Percent cover / image covers as `percent_cover.csv`
- Calcification rates as `calcification_rates.csv`

## Why Use It

Use `coralnet-exporter` when you need to:

- Build local CoralNet datasets for computer vision or active learning experiments
- Reproduce CoralNet source exports outside the browser
- Download private CoralNet sources using authenticated credentials
- Resume long-running CoralNet downloads after interruption
- Run CoralNet exports on a server, HPC cluster, or Slurm environment
- Prepare CoralNet image, annotation, metadata, and labelset files for downstream machine learning pipelines

Current v0.1 scope:

- Source discovery from a CoralNet source ID or URL
- Credential handling from `.env.coralnet`, environment variables, or interactive prompt
- Image download for public and private sources
- Classifier/backend info export
- Labelset export
- Metadata export: all images and confirmed images
- Annotation export: all images and confirmed images
- Percent cover / image covers export
- Calcification rates export
- Resume state and logs per source
- Chunked fallback for very large `annotations_all.csv` exports
- Slurm script generation

## Install

For normal users, install directly from GitHub:

```bash
pip install git+https://github.com/ReefNet-Project/coralnet-exporter.git
```

Upgrade to the latest GitHub version:

```bash
pip install --upgrade --force-reinstall git+https://github.com/ReefNet-Project/coralnet-exporter.git
```

For local development from a cloned repo:

```bash
git clone https://github.com/ReefNet-Project/coralnet-exporter.git
cd coralnet-exporter
pip install -e .
```

## Credentials

`coralnet-exporter` supports three credential methods:

1. Interactive login prompt, where the CLI asks for username and password.
2. A local `.env.coralnet` file.
3. Environment variables: `CORALNET_USERNAME` and `CORALNET_PASSWORD`.

The simplest option is to let the CLI ask directly:

```bash
coralnet-exporter login
coralnet-exporter download 3354 --output-dir output
```

For repeated or background runs, use a local `.env.coralnet` file. Create it in the directory where you will run `coralnet-exporter`.

For example, if you want downloads to go into `/data/coralnet_downloads`, do this:

```bash
mkdir -p /data/coralnet_downloads
cd /data/coralnet_downloads
cat > .env.coralnet <<'EOF_CREDS'
CORALNET_USERNAME=your_username
CORALNET_PASSWORD=your_password
EOF_CREDS
chmod 600 .env.coralnet
```

Then run commands from that same directory:

```bash
coralnet-exporter login --env-file .env.coralnet
coralnet-exporter download 3354 --env-file .env.coralnet --output-dir output
```

If `.env.coralnet` is in the current working directory, `--env-file .env.coralnet` is optional because it is the default.

You can also keep the credential file somewhere else and pass its path:

```bash
coralnet-exporter download 3354 --env-file /secure/path/.env.coralnet
```

Or use environment variables:

```bash
export CORALNET_USERNAME=your_username
export CORALNET_PASSWORD=your_password
```

Validate login:

```bash
coralnet-exporter login --env-file .env.coralnet
```

## Download A Source

```bash
coralnet-exporter download 3354 --output-dir output --resume
```

By default, this downloads images, labelset, metadata, annotations, and classifier information. Percent cover and calcification rates are optional because they can require source-specific choices.

Equivalent URL form:

```bash
coralnet-exporter download https://coralnet.ucsd.edu/source/3354/
```

Choose exports:

```bash
coralnet-exporter download 3354 \
  --include images,labelset,metadata,annotations,classifier,covers,calcification \
  --output-dir output \
  --resume
```

Available export names:

```text
images,labelset,metadata,annotations,classifier,covers,calcification
```

`covers` exports `percent_cover.csv`. `calcification` exports `calcification_rates.csv`; pass `--calcification-rate-table-id` to choose a specific CoralNet rate table, or omit it to use the first table shown in CoralNet's UI.

## Output Layout

```text
output/
  <source name>/
    images/
    classifier_info.csv
    classifier_info.json
    labelset.csv
    metadata_all.csv
    metadata_confirmed.csv
    annotations_all.csv
    annotations_confirmed.csv
    percent_cover.csv
    calcification_rates.csv
    .coralnet-exporter/
      source.csv
      *_state.json
      *_summary.json
      *_failures.csv
      *.log
```

Large annotation exports may create resumable chunks:

```text
output/<source name>/.annotation_chunks/all_by_image_id/
```

## Background Run

```bash
coralnet-exporter download 3354 --background --include labelset,metadata,annotations,classifier
```

The command prints a log path. Monitor it with:

```bash
tail -f output/_coralnet_exporter_logs/source_3354.log
```

## Slurm

Generate an sbatch file:

```bash
coralnet-exporter make-slurm 3354 \
  --output run_coralnet_3354.sbatch \
  --output-dir output \
  --conda-env /ibex/project/c2253/yahia_code/envs/pytoenv
```

Submit:

```bash
sbatch run_coralnet_3354.sbatch
```

## Maintainer

Maintained by [Yahia Battach](https://github.com/shakesBeardZ) for the ReefNetAI platform: https://reefnet.kaust.edu.sa/

For issues, feature requests, or contributions, use the GitHub repository issue tracker.
