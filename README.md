# Bridge TDMS Explorer

Prototype tools for inspecting Memorial Bridge-style TDMS sensor files.

## What It Does

- Finds `.tdms` files in a folder.
- Ignores `_version_` copies and `Decimate` files for analysis.
- Detects the available time range from filename timestamps.
- Parses active float64 TDMS channels, including repeated raw chunks.
- Reconstructs wall-clock timestamps from the filename plus the TDMS `Time` channel.
- Combines all normal files selected by the dashboard time range.
- Builds a sensor catalog with active and configured-but-inactive channels.
- Computes raw channel summaries, 1-minute trend features, event candidates, and sensor health flags.
- Discovers correlated channel groups from windowed sensor features.
- Classifies event candidates into traffic/vibration, boat collision / impact, and drawbridge-operation-like families.
- Reports behavior shifts only when three or more correlated channels agree.
- Decodes BDI sensor labels into approximate physical bridge locations and highlights supporting sensors for impact candidates.
- Provides a Streamlit dashboard for exploring files, signals, traffic-like events, trends, and health.

## Run The Dashboard

Install the Python dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 -m streamlit run app.py
```

Use the same Python environment for both commands. For example, if Streamlit is
installed in Anaconda, run `/opt/anaconda3/bin/python -m pip install -r
requirements.txt` and `/opt/anaconda3/bin/python -m streamlit run app.py`.

On launch, use the sidebar folder browser or enter the local folder that contains
your TDMS files. You can select either a folder with TDMS files directly inside
it or a parent folder with daily TDMS subfolders.

On startup the app scans for new or changed TDMS files and ingests them into a
local DuckDB/Parquet cache under `cache/`. The sidebar then shows the available
cached time range. Use the `Analysis Time Range` start/end date and time inputs
to choose the interval to query. When more than a week of data is available, the
default launch range is the latest week.

TDMS timestamps are stored and queried as UTC in the cache. The dashboard range
controls, chart axes, event timelines, and added `*_eastern` table/download
columns show the same moments in Eastern bridge local time, including daylight
saving time changes.

TDMS discovery is recursive, so a folder organized like `tdms_files/` with
subfolders such as `tdms_files_2026-06-17_07-00-01/` and
`tdms_files_2026-06-18_07-00-01/` can be loaded from the parent folder.

Startup progress distinguishes scanning files, ingesting new files, updating the
index, and ready state. Once ingestion is complete, the app queries only the
selected time range and requested channels from the local cache instead of
combining every selected TDMS file in memory.

Use the sidebar `View` selector to switch between pages. Only the selected view
is computed, which keeps display-setting changes responsive on large TDMS
selections.

## Anomaly Detection

The app includes an explainable first-pass anomaly workflow:

- `Correlation Groups`: finds channels that move together over the selected time range.
- `Event Detection`: detects rolling-RMS bursts and classifies traffic/vibration, impact candidates, and operation-like events.
- `Anomaly Review`: separates urgent boat collision / impact candidates from reportable operation or behavior shifts.

Behavior shifts are gated by group support. By default, a reported shift requires
at least three channels in the same correlated group to show compatible abnormal
movement. This helps avoid false alarms from a noisy or failed single sensor.

Sensor placement maps are decoded from the BDI installation-plan naming scheme.
Locations 1-8 are on the south fixed span; locations 9-10 are on the south tower.
Impact-candidate review highlights the sensors supporting those candidate events.

## Run A CLI Summary

```bash
python3 analyze_tdms.py --cache
```

Build or update the scalable app cache:

```bash
python3 analyze_tdms.py --ingest
```

The legacy `--cache` command writes per-file Parquet files under `cache/`:

- raw samples
- sensor catalog
- channel summary
- 1-minute trend features
- sensor health flags

The scalable `--ingest` command writes:

- `cache/bridge_index.duckdb`
- partitioned raw samples under `cache/samples/`
- partitioned trend features under `cache/features/`
- per-file metadata under `cache/metadata/`

## Notes

The parser is intentionally scoped to the TDMS layout in these bridge-monitoring files:

- channel data stored as `float64`
- group name `AllNormal`
- repeated raw data chunks
- normal files named like `*_Normal_Data_AllNormal.tdms`

If later files contain strings, timestamps, or other numeric TDMS raw types as active sample channels, the parser should be extended before interpreting those channels.
