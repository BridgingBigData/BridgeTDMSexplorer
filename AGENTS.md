# Codex Agent Handoff

## Project

Bridge TDMS Explorer is a Streamlit application for inspecting Memorial Bridge
sensor TDMS files. The app parses normal TDMS recordings, combines selected
files by filename timestamp, visualizes raw and windowed trends, discovers
correlated sensor groups, and provides an explainable first-pass anomaly review
for traffic/vibration, boat collision or impact candidates, drawbridge
operation-like events, and group-confirmed behavior shifts.

## Repository Layout

- `app.py`: Streamlit dashboard and UI wiring.
- `analyze_tdms.py`: CLI summary/cache builder for local inspection.
- `tdms_bridge/parser.py`: TDMS parser, file discovery, timestamp
  reconstruction, summaries, trend features, event detection, and health flags.
- `tdms_bridge/ml.py`: correlation grouping, event-family classification, and
  group-confirmed behavior-shift detection.
- `README.md`: user-facing run notes and workflow summary.

## Run Commands

Use the app from the repository root:

```bash
streamlit run app.py
```

Build a CLI summary and Parquet cache:

```bash
python3 analyze_tdms.py --cache
```

Basic syntax check:

```bash
python3 -m py_compile app.py analyze_tdms.py tdms_bridge/parser.py tdms_bridge/ml.py tdms_bridge/__init__.py
```

## Data Rules

Do not commit TDMS recordings or generated analysis cache files. The repository
`.gitignore` intentionally excludes:

- `*.tdms`
- `cache/`
- `__pycache__/`
- `.DS_Store`
- `.Rhistory`

The app ignores `_version_` copies and `Decimate` files during analysis. Normal
analysis files are expected to look like:

```text
*_Normal_Data_AllNormal.tdms
```

## Current Behavior

The sidebar scans the selected TDMS folder, reports the detected filename
timestamp range, and exposes explicit start/end date and time inputs. All
selected normal files are parsed and combined into one time-indexed dataset.

The main sidebar views are:

- `Files`: selected normal files, ignored files, and sensor catalog.
- `Raw Signals`: downsampled raw signal plots over the selected range.
- `Event Detection`: rolling-RMS event candidates classified as
  traffic/vibration, boat collision / impact, or drawbridge-operation-like.
- `Correlation Groups`: groups of channels that move together.
- `Anomaly Review`: reportable shifts and urgent impact candidates.
- `Trends`: windowed metrics such as RMS, peak-to-peak, mean, std, min, max.
- `Sensor Health`: inactive, flatline, and extreme-value flags.

Behavior shifts are reportable only when at least three channels in the same
correlated group show compatible abnormal behavior. This is deliberate; keep the
three-channel rule unless the user explicitly asks to change it.

## Implementation Notes

- The parser currently handles the TDMS layout observed in this data family:
  float64 active channels, repeated raw chunks, and group metadata such as
  `AllNormal`.
- Filename timestamps are used as the recording start time; the TDMS `Time`
  channel supplies elapsed seconds inside each file.
- Event detection subtracts a rolling two-minute median baseline, computes
  rolling RMS, and flags values above `background + threshold_sigma * robust
  spread`.
- Correlation groups are connected components over absolute Spearman
  correlation from windowed features.
- Boat collision / impact candidates require short, high-severity, multi-channel
  support.
- Drawbridge-operation-like events are treated as sustained coordinated group
  responses and should be separated from damage-like behavior shifts when
  possible.

## Development Guidance

- Keep the app explainable. Prefer robust baselines, correlation groups, and
  transparent thresholds before adding opaque models.
- Preserve user data privacy and repository size by keeping raw TDMS files out
  of git.
- If adding dependencies, document them in the README and keep the app runnable
  from a local folder of TDMS files.
- After edits, run the syntax check above and, when practical, open the
  Streamlit app at `http://localhost:8501`.
