# Literature Signals and Sector ETF Returns

This project tests whether scientific publication trends contain useful signal for near-term health-sector ETF returns.

The workflow is split into two parts:

1. An offline analytics pipeline builds literature features, market features, Granger test results, and rolling model comparisons.
2. A lightweight Flask dashboard serves the cached results from `artifacts/`.

## Industries

- Medical Devices: OpenAlex data in `data/raw/medical_devices/`; ETFs `IHI`, `XHE`.
- Biotech: OpenAlex data in `data/raw/biotech/`; ETFs `XBI`, `IBB`.

Market data is stored locally under `data/market/`.

## Project Layout

```text
app/                 Flask routes, HTML template, CSS, and browser JavaScript
literature_market/   Reusable data loading, feature engineering, Granger, modeling, and pipeline code
scripts/             Command-line scripts for data prep and artifact generation
artifacts/           Cached CSV/JSON outputs used by the dashboard and intended for GitHub
data/                Local raw OpenAlex and market data, ignored by git
notebooks/           Exploratory notebook work
run.py               Flask app entry point
Dockerfile           Docker image for serving cached artifacts
requirements.txt     Python dependencies
```

## Setup

Use a virtual environment so dependencies do not install into the system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data Prep

Download OpenAlex work metadata with the official [OpenAlex CLI](https://developers.openalex.org/download/openalex-cli). The CLI saves one JSON metadata file per work. Put those JSON folders under `data/raw/<industry>/<openalex-id>/`.

Install the CLI:

```bash
python -m pip install openalex-official
```

Download these OpenAlex groups:

- Biotech field IDs: `13`, `30`
- Medical Devices subfield IDs: `2204`, `2741`
- Medical Devices topic IDs: `T13690`

Example: download biotech articles for field `30`:

```bash
mkdir -p ~/Desktop/openalex_results/data/raw/biotech/30

openalex download \
  --api-key "API_KEY" \
  --output ~/Desktop/openalex_results/data/raw/biotech/30 \
  --filter "primary_topic.field.id:30,from_publication_date:2020-01-01,to_publication_date:2026-04-24,type:article,is_retracted:false,is_paratext:false,primary_location.is_published:true"
```

Example: download medical devices articles for subfield `2204`:

```bash
mkdir -p ~/Desktop/openalex_results/data/raw/medical_devices/2204

openalex download \
  --api-key "API_KEY" \
  --output ~/Desktop/openalex_results/data/raw/medical_devices/2204 \
  --filter "primary_topic.subfield.id:2204,from_publication_date:2020-01-01,to_publication_date:2026-04-24,type:article,is_retracted:false,is_paratext:false,primary_location.is_published:true"
```

Example: download medical devices articles for topic `T13690`:

```bash
mkdir -p ~/Desktop/openalex_results/data/raw/medical_devices/T13690

openalex download \
  --api-key "API_KEY" \
  --output ~/Desktop/openalex_results/data/raw/medical_devices/T13690 \
  --filter "primary_topic.id:T13690,from_publication_date:2020-01-01,to_publication_date:2026-04-24,type:article,is_retracted:false,is_paratext:false,primary_location.is_published:true"
```

Repeat the same pattern for all required IDs, changing the output folder and the relevant `primary_topic.field.id`, `primary_topic.subfield.id`, or `primary_topic.id` filter.

After downloading JSON metadata folders, export them into the CSV files used by the pipeline.

Export OpenAlex JSON folders into per-folder CSVs:

```bash
python -m scripts.export_openalex_csvs
```

Download market data from Yahoo Finance:

```bash
python -m scripts.pull_yfinance_data
```

Raw/source data under `data/` is intentionally ignored by git because it can be large or regenerated.

## Build Cached Results

Build both industries:

```bash
python -m scripts.build_results --industry all
```

Build only one industry:

```bash
python -m scripts.build_results --industry medical_devices
python -m scripts.build_results --industry biotech
```

The pipeline writes dashboard-ready outputs to:

```text
artifacts/<industry>/
├── best_models.csv
├── comparison.csv
├── daily_literature_features.csv
├── experiment_results.csv
├── fold_counts.csv
├── granger_results.csv
├── granger_significant_weekly.csv
├── jan1_downsampling.csv
├── modeling_dataset_sample.csv
└── summary.json
```

## Run Locally

Start the Flask app:

```bash
flask --app run.py run --debug
```

Then open `http://127.0.0.1:5000`.

## Docker

Build cached artifacts before building the Docker image:

```bash
python -m scripts.build_results --industry all
docker build -t literature-market-app .
docker run -p 8000:8000 literature-market-app
```

Then open `http://127.0.0.1:8000`.

## Methodology Summary

The pipeline compares two feature sets:

- Baseline: market features only, such as returns, volatility, volume, and benchmark movement.
- Literature: baseline market features plus publication volume, growth, novelty, keyword mix, subfield mix, funding, and core-journal signals.

Models are evaluated with rolling monthly train/test windows across 1-8 week prediction horizons. Regression uses RMSE, classification uses accuracy, and positive lift means the literature-augmented model improved over the market-only baseline.

