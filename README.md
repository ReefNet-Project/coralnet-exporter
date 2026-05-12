# CoralNet Exporter

`coralnet-exporter` is a Python CLI for exporting CoralNet source data into a reproducible local folder structure.

Current v0.1 scope:

- Source discovery from a CoralNet source ID or URL
- Credential handling from `.env.coralnet`, environment variables, or interactive prompt
- Classifier/backend info export
- Labelset export
- Metadata export: all images and confirmed images
- Annotation export: all images and confirmed images
- Percent cover / image covers export
- Resume state and logs per source
- Chunked fallback for very large `annotations_all.csv` exports
- Slurm script generation

Planned next exporters:

- Fully integrated authenticated image download for private sources

## Install

From this repository:

```bash
pip install -e .
```

## Credentials

Preferred local credential file:

```bash
cat > .env.coralnet <<'EOF_CREDS'
CORALNET_USERNAME=your_username
CORALNET_PASSWORD=your_password
EOF_CREDS
chmod 600 .env.coralnet
```

Or use environment variables:

```bash
export CORALNET_USERNAME=your_username
export CORALNET_PASSWORD=your_password
```

Validate login:

```bash
coralnet-exporter login
```

## Download A Source

```bash
coralnet-exporter download 3354 --output-dir output --resume
```

Equivalent URL form:

```bash
coralnet-exporter download https://coralnet.ucsd.edu/source/3354/
```

Choose exports:

```bash
coralnet-exporter download 3354 \
  --include labelset,metadata,annotations,classifier,covers,calcification \
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
