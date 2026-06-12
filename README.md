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
- Provides a Streamlit dashboard for exploring files, signals, traffic-like events, trends, and health.

## Run The Dashboard

```bash
streamlit run app.py
```

The app opens against the current folder by default:

```text
/Users/rgandhi/Downloads/tdms_files
```

On startup the sidebar shows the detected filename time range. Use the
`Analysis Time Range` start/end date and time inputs to choose the files to
combine. The plots, events, trends, summaries, and health checks are computed
over that combined selection.

When the selected file set changes, the app shows a main-page progress bar with
the current parse/combine step and an estimated time remaining. Once the initial
load is complete, switching display settings and views reuses the warmed dataset.

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

## Run A CLI Summary

```bash
python3 analyze_tdms.py --cache
```

This writes Parquet cache files under `cache/`:

- raw samples
- sensor catalog
- channel summary
- 1-minute trend features
- sensor health flags

## Notes

The parser is intentionally scoped to the TDMS layout in these bridge-monitoring files:

- channel data stored as `float64`
- group name `AllNormal`
- repeated raw data chunks
- normal files named like `*_Normal_Data_AllNormal.tdms`

If later files contain strings, timestamps, or other numeric TDMS raw types as active sample channels, the parser should be extended before interpreting those channels.
