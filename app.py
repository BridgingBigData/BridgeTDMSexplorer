from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tdms_bridge.ml import (
    classify_event_families_from_events,
    correlation_groups_from_features,
    detect_operation_and_behavior_shifts_from_features,
)
from tdms_bridge.locations import (
    enrich_sensor_catalog,
    event_channels,
    sensor_location_figure,
    sensor_location_table,
)
from tdms_bridge.store import (
    get_available_range,
    ingest_folder,
    query_catalog,
    query_events,
    query_features,
    query_file_index,
    query_health,
    query_ignored_files,
    query_samples,
    query_summary,
    selected_range_signature,
)

METRIC_GUIDANCE = {
    "rms": "Root mean square: signal energy in the window. Good for vibration and traffic intensity.",
    "peak_to_peak": "Maximum minus minimum in the window. Good for seeing the size of transient swings.",
    "mean": "Average value in the window. Good for slow drift, temperature response, and baseline movement.",
    "std": "Standard deviation in the window. Good for activity/noise level around the local average.",
    "min": "Lowest sample value in the window.",
    "max": "Highest sample value in the window.",
}

EVENT_COLORS = {
    "Boat collision / impact candidate": "#e11d48",
    "Drawbridge operation-like event": "#7c3aed",
    "Group-confirmed behavior shift": "#111827",
    "Group-supported traffic/vibration event": "#f59e0b",
    "Single/few-channel traffic-like event": "#fbbf24",
}

DEFAULT_EVENT_COLOR = "#64748b"

REVIEW_WORKFLOW = [
    "Files",
    "Raw Signals",
    "Event Detection",
    "Anomaly Review",
    "Sensor Health",
]

ALL_VIEWS = [
    *REVIEW_WORKFLOW,
    "Correlation Groups",
    "Trends",
]


st.set_page_config(
    page_title="Bridge TDMS Explorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def session_cached(key: tuple, compute):
    cache = st.session_state.setdefault("analysis_cache", {})
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def get_correlation_result_from_store(
    cache_dir: Path,
    signature: tuple,
    selected_start: datetime,
    selected_end: datetime,
    catalog: pd.DataFrame,
    metric: str,
    window: str,
    min_abs_corr: float,
    min_group_size: int,
):
    key = (
        "correlation_groups",
        signature,
        metric,
        window,
        float(min_abs_corr),
        int(min_group_size),
    )
    return session_cached(
        key,
        lambda: correlation_groups_from_features(
            query_features(cache_dir, selected_start, selected_end, window),
            catalog,
            metric,
            min_abs_corr,
            int(min_group_size),
        ),
    )


def get_event_families_from_store(
    cache_dir: Path,
    signature: tuple,
    selected_start: datetime,
    selected_end: datetime,
    catalog: pd.DataFrame,
    channels: list[str],
    corr_metric: str,
    corr_window: str,
    min_abs_corr: float,
    group_min_channels: int,
    window_seconds: int,
    threshold_sigma: float,
    impact_ratio: float,
) -> pd.DataFrame:
    channel_key = tuple(channels)
    key = (
        "event_families",
        signature,
        channel_key,
        corr_metric,
        corr_window,
        float(min_abs_corr),
        int(group_min_channels),
        int(window_seconds),
        float(threshold_sigma),
        float(impact_ratio),
    )

    def compute():
        groups = get_correlation_result_from_store(
            cache_dir,
            signature,
            selected_start,
            selected_end,
            catalog,
            corr_metric,
            corr_window,
            min_abs_corr,
            int(group_min_channels),
        ).groups
        events = query_events(
            cache_dir,
            selected_start,
            selected_end,
            channels,
            int(window_seconds),
            float(threshold_sigma),
        )
        return classify_event_families_from_events(
            events,
            groups,
            float(impact_ratio),
            int(group_min_channels),
        )

    return session_cached(key, compute)


def get_behavior_shifts_from_store(
    cache_dir: Path,
    signature: tuple,
    selected_start: datetime,
    selected_end: datetime,
    catalog: pd.DataFrame,
    corr_metric: str,
    corr_window: str,
    min_abs_corr: float,
    group_min_channels: int,
    shift_window: str,
    z_threshold: float,
) -> pd.DataFrame:
    key = (
        "behavior_shifts",
        signature,
        corr_metric,
        corr_window,
        float(min_abs_corr),
        int(group_min_channels),
        shift_window,
        float(z_threshold),
    )

    def compute():
        groups = get_correlation_result_from_store(
            cache_dir,
            signature,
            selected_start,
            selected_end,
            catalog,
            corr_metric,
            corr_window,
            min_abs_corr,
            int(group_min_channels),
        ).groups
        features = query_features(cache_dir, selected_start, selected_end, shift_window)
        return detect_operation_and_behavior_shifts_from_features(
            features, groups, float(z_threshold), int(group_min_channels)
        )

    return session_cached(key, compute)


def set_tdms_folder(folder: str) -> None:
    st.session_state["tdms_folder"] = folder.strip()
    st.session_state.pop("analysis_cache", None)
    st.session_state.pop("folder_browser_error", None)
    st.session_state.pop("last_load_message", None)
    st.session_state.pop("ingest_key", None)
    st.session_state.pop("range_key", None)


def set_time_range(start: datetime, end: datetime) -> None:
    st.session_state["start_date"] = start.date()
    st.session_state["start_time"] = start.time()
    st.session_state["end_date"] = end.date()
    st.session_state["end_time"] = end.time()


def browse_for_tdms_folder(initial_folder: str) -> tuple[str | None, str | None]:
    initial_path = Path(initial_folder).expanduser() if initial_folder else Path.home()
    if not initial_path.exists():
        initial_path = Path.home()

    if platform.system() == "Darwin":
        return browse_for_tdms_folder_macos(initial_path)

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            initialdir=str(initial_path),
            title="Select TDMS root folder",
            mustexist=True,
        )
    except Exception as exc:
        return None, f"Could not open the OS folder browser: {exc}"
    finally:
        if "root" in locals():
            root.destroy()

    return selected or None, None


def browse_for_tdms_folder_macos(initial_path: Path) -> tuple[str | None, str | None]:
    script = (
        'set selectedFolder to choose folder with prompt "Select TDMS root folder" '
        f'default location POSIX file "{escape_applescript_text(str(initial_path))}"\n'
        "return POSIX path of selectedFolder"
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return None, f"Could not open the macOS folder browser: {exc}"
    if result.returncode == 0:
        return result.stdout.strip() or None, None
    if result.returncode == 1 and "User canceled" in result.stderr:
        return None, None
    return None, result.stderr.strip() or "Could not open the macOS folder browser."


def escape_applescript_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    st.title("Bridge TDMS Explorer")
    startup_progress = st.empty()

    with st.sidebar:
        if "tdms_folder" not in st.session_state:
            st.session_state["tdms_folder"] = ""
        st.markdown("**Data Folder**")
        if not st.session_state["tdms_folder"]:
            st.caption(
                "Choose the parent folder that contains TDMS files or daily TDMS subfolders."
            )
        if st.button(
            "Browse for TDMS folder",
            use_container_width=True,
        ):
            st.session_state.pop("folder_browser_error", None)
            selected_folder, browser_error = browse_for_tdms_folder(
                st.session_state["tdms_folder"]
            )
            if selected_folder:
                set_tdms_folder(selected_folder)
            elif browser_error:
                st.session_state["folder_browser_error"] = browser_error
        if st.session_state.get("folder_browser_error"):
            st.warning(st.session_state["folder_browser_error"])
        with st.form("tdms_folder_form"):
            folder_entry = st.text_input(
                "Folder path",
                value=st.session_state["tdms_folder"],
                placeholder="/path/to/tdms_files",
            )
            load_folder = st.form_submit_button("Load folder", use_container_width=True)
        if load_folder:
            set_tdms_folder(folder_entry)

        folder = st.session_state["tdms_folder"]
        if not folder:
            st.info("Select a folder to scan.")
            return
        folder_path = Path(folder).expanduser().resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            st.error("The TDMS folder path does not exist or is not a folder.")
            return
        if "refresh_token" not in st.session_state:
            st.session_state.refresh_token = 0
        if st.button(
            "Rescan / ingest new files",
            use_container_width=True,
            help="Refresh the file list and ingest new or changed TDMS files into the local scalable cache.",
        ):
            st.cache_data.clear()
            st.session_state.pop("analysis_cache", None)
            st.session_state.pop("last_load_message", None)
            st.session_state.pop("ingest_key", None)
            st.session_state.pop("range_key", None)
            st.session_state.refresh_token += 1

        cache_dir = folder_path / "cache"
        ingest_key = (str(folder_path), st.session_state.refresh_token)
        if st.session_state.get("ingest_key") != ingest_key:
            progress = startup_progress.progress(0.0, text="Scanning files...")

            def ingest_progress(stage: str, current: int, total: int, message: str) -> None:
                denominator = max(total, 1)
                progress.progress(
                    min(1.0, max(0.0, current / denominator)),
                    text=f"{stage}: {message}",
                )

            try:
                report = ingest_folder(folder_path, cache_dir, ingest_progress)
            except RuntimeError as exc:
                st.error(str(exc))
                return
            progress.progress(
                1.0,
                text=(
                    "Ready: "
                    f"{report.ingested} ingested, {report.skipped} skipped, "
                    f"{report.failed} failed."
                ),
            )
            st.session_state["ingest_report"] = report
            st.session_state["ingest_key"] = ingest_key
            st.session_state["analysis_cache"] = {}

        report = st.session_state.get("ingest_report")
        if report:
            st.caption(
                f"Scanned {report.scanned} TDMS files. "
                f"Ingested {report.ingested}, skipped {report.skipped}, "
                f"failed {report.failed}, ignored {report.ignored}."
            )
            if report.messages:
                with st.expander("Ingestion warnings", expanded=False):
                    for message in report.messages:
                        st.warning(message)

        try:
            min_time, max_time = get_available_range(cache_dir)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        if min_time is None or max_time is None:
            st.warning("No successfully ingested normal TDMS files found.")
            return

        min_time = min_time.to_pydatetime()
        max_time = max_time.to_pydatetime()
        default_start = max(min_time, max_time - timedelta(days=7))
        range_key = (str(cache_dir), min_time, max_time)
        if st.session_state.get("range_key") != range_key:
            set_time_range(default_start, max_time)
            st.session_state["range_key"] = range_key

        st.markdown("**Available Cached Time Range**")
        st.caption(f"{min_time:%Y-%m-%d %H:%M:%S} to {max_time:%Y-%m-%d %H:%M:%S}")

        with st.expander("Analysis Time Range", expanded=True):
            range_preset = st.selectbox(
                "Quick range",
                ["Latest hour", "Latest day", "Latest week", "All cached"],
                index=2,
                help="Choose a common range, then apply it to the start and end controls below.",
            )
            if st.button("Apply quick range", use_container_width=True):
                if range_preset == "Latest hour":
                    set_time_range(max(min_time, max_time - timedelta(hours=1)), max_time)
                elif range_preset == "Latest day":
                    set_time_range(max(min_time, max_time - timedelta(days=1)), max_time)
                elif range_preset == "Latest week":
                    set_time_range(max(min_time, max_time - timedelta(days=7)), max_time)
                else:
                    set_time_range(min_time, max_time)

            start_date = st.date_input(
                "Start date",
                min_value=min_time.date(),
                max_value=max_time.date(),
                key="start_date",
                help="First recording start date to query from the local cache. The default is the latest week.",
            )
            start_time = st.time_input(
                "Start time",
                key="start_time",
                help="First recording start time to query from the local cache.",
            )
            end_date = st.date_input(
                "End date",
                min_value=min_time.date(),
                max_value=max_time.date(),
                key="end_date",
                help="Last recording start date to query from the local cache.",
            )
            end_time = st.time_input(
                "End time",
                key="end_time",
                help="Last recording start time to query from the local cache.",
            )

        selected_start = datetime.combine(start_date, start_time)
        selected_end = datetime.combine(end_date, end_time)
        if selected_start > selected_end:
            st.error("Start time must be before end time.")
            return
        selected_files = query_file_index(cache_dir, selected_start, selected_end)
        selected_files = selected_files[selected_files["status"].eq("ready")].copy()
        ignored_files = query_ignored_files(cache_dir)
        st.caption(f"Selected {len(selected_files)} ingested file(s).")
        st.caption(
            "The selected folder is searched recursively. Time filtering queries the "
            "local Parquet/DuckDB cache. Raw TDMS files are only parsed when new or "
            "changed files are ingested."
        )

        if selected_files.empty:
            st.warning("No ingested files fall inside the selected time range.")
            return

        selected_signature = selected_range_signature(selected_files, selected_start, selected_end)
        catalog_rows = query_catalog(cache_dir, selected_start, selected_end)
        if catalog_rows.empty:
            st.warning("No sensor catalog rows found for this selected range.")
            return
        active_catalog = _merge_catalogs(catalog_rows)
        active_catalog = active_catalog[active_catalog["active"]]
        active_catalog = enrich_sensor_catalog(active_catalog)
        traffic_candidates = active_catalog[
            active_catalog["sensor_type"].isin(["Accelerometer", "Quarterarm", "Half Bridge I"])
        ]["channel"].tolist()
        sensor_types = sorted(
            item for item in active_catalog["sensor_type"].dropna().unique() if item
        )
        selected_types = st.multiselect(
            "Sensor types",
            sensor_types,
            default=[item for item in sensor_types if item != "Time"],
            help="Filter channel pickers by sensor family. Accelerometers are usually best for vibration and traffic bursts; bridge channels are useful for strain-like response and slow trends.",
        )
        with st.expander("Anomaly Model Settings", expanded=False):
            group_min_channels = st.number_input(
                "Channels required to report a group shift",
                min_value=3,
                max_value=10,
                value=3,
                step=1,
                help="A behavior shift is reportable only when at least this many correlated channels agree. Keep this at 3 or higher to avoid single-sensor false alarms.",
            )
            corr_metric = st.selectbox(
                "Correlation basis",
                ["rms", "peak_to_peak", "mean", "std"],
                index=0,
                help="Metric used to discover sensor groups. Use RMS/peak-to-peak for traffic and vibration; use mean for slow bridge motion or thermal drift.",
            )
            corr_window = st.selectbox(
                "Correlation window",
                ["1min", "5min", "15min", "1h"],
                index=1,
                help="Time window used before computing correlations. Larger windows emphasize slow shared trends; smaller windows emphasize short activity bursts.",
            )
            min_abs_corr = st.slider(
                "Minimum absolute correlation",
                0.30,
                0.95,
                0.75,
                0.05,
                help="Channels with absolute correlation above this value are connected into groups. Lower values create larger groups; higher values create stricter groups.",
            )
        with st.expander("Event Detection Settings", expanded=False):
            event_window_seconds = st.slider(
                "RMS window seconds",
                1,
                20,
                5,
                help="Length of the rolling RMS window. Shorter windows catch sharp bursts; longer windows smooth activity and favor sustained events.",
            )
            event_threshold_sigma = st.slider(
                "Threshold sigma",
                2.0,
                8.0,
                4.0,
                0.5,
                help="Sensitivity threshold above background RMS. Lower values find more events and more false positives; higher values keep only stronger bursts.",
            )
            event_impact_ratio = st.slider(
                "Impact severity ratio",
                1.5,
                8.0,
                3.0,
                0.5,
                help="A boat collision / impact candidate needs peak RMS at least this many times the event threshold, plus correlated multi-channel support.",
            )
            st.caption(
                "These settings are shared by Event Detection, plot overlays, and "
                "urgent impact candidates in Anomaly Review."
            )
        with st.expander("Display Settings", expanded=False):
            show_data_gaps = st.checkbox(
                "Show data gaps",
                value=True,
                help="Break plotted lines and shade discontinuities between files or missing periods.",
            )
            gap_threshold_seconds = st.number_input(
                "Gap threshold seconds",
                min_value=0.1,
                max_value=3600.0,
                value=2.0,
                step=0.5,
                help="Positive time gaps above this threshold are treated as discontinuities in plots.",
            )

        st.markdown("**Review workflow**")
        st.caption("Default path: Files -> Raw Signals -> Event Detection -> Anomaly Review -> Sensor Health.")
        page = st.radio(
            "View",
            ALL_VIEWS,
            help="Only the selected view is computed. This keeps display changes from recomputing every model and plot.",
        )

        needs_gaps = page == "Files" or (
            show_data_gaps and page in {"Raw Signals", "Event Detection", "Trends"}
        )
        gaps = (
            session_cached(
                ("plot_gaps", selected_signature, float(gap_threshold_seconds)),
                lambda: detect_file_index_gaps(selected_files, float(gap_threshold_seconds)),
            )
            if needs_gaps
            else _empty_gaps()
        )

    cached_files = query_file_index(cache_dir, min_time, max_time)
    metadata_cols = st.columns(6)
    metadata_cols[0].metric("Cached files", len(cached_files))
    metadata_cols[1].metric("Selected files", len(selected_files))
    metadata_cols[2].metric("Ignored files", len(ignored_files))
    metadata_cols[3].metric("Active channels", len(active_catalog))
    metadata_cols[4].metric("Selected rows", f"{int(selected_files['sample_rows'].sum()):,}")
    metadata_cols[5].metric(
        "Selected span",
        _format_duration(
            pd.Timestamp(selected_files["sample_end"].max())
            - pd.Timestamp(selected_files["sample_start"].min())
        ),
    )

    st.caption(
        "Cached data range: "
        f"{min_time:%Y-%m-%d %H:%M:%S} to {max_time:%Y-%m-%d %H:%M:%S}. "
        "Analysis excludes version copies and decimated files; new files are added through ingestion."
    )

    if page == "Files":
        st.subheader("Selected Normal Files")
        st.caption(
            "These are the normal TDMS recordings included in the current analysis range. "
            "Version copies and decimated files are listed separately and excluded."
        )
        selected_display = add_relative_paths(selected_files, folder_path)
        display_cols = ["relative_path", "timestamp", "size_mb", "included"]
        show_dataframe(selected_display[display_cols])
        download_dataframe(
            "Download selected files CSV",
            selected_display[display_cols],
            "selected_tdms_files.csv",
        )

        st.subheader("Ignored Files")
        ignored_display = add_relative_paths(ignored_files, folder_path)
        ignored_cols = ["relative_path", "timestamp", "size_mb", "ignored_reason"]
        if ignored_display.empty:
            show_empty_state(
                "No ignored TDMS files were found.",
                "Version copies and decimated files will appear here when present.",
            )
        else:
            show_dataframe(ignored_display[ignored_cols])

        st.subheader("Sensor Catalog Across Selection")
        show_dataframe(active_catalog)

        st.subheader("Sensor Placement Map")
        st.caption(
            "Sensor locations are decoded from the BDI installation-plan naming scheme. "
            "Locations 1-8 are on the south fixed span; locations 9-10 are on the south tower."
        )
        st.plotly_chart(
            sensor_location_figure(active_catalog, title="Decoded Sensor Placement"),
            use_container_width=True,
        )

        st.subheader("Detected Gaps")
        st.caption(
            "Positive timestamp gaps above the display threshold are treated as plot discontinuities. "
            "They are shaded in charts when data-gap display is enabled. "
            f"Current display threshold: {float(gap_threshold_seconds):.1f}s."
        )
        if gaps.empty:
            show_empty_state(
                "No timestamp gaps exceed the current display threshold.",
                "Lower the gap threshold in Display Settings if you need to reveal shorter discontinuities.",
            )
        else:
            show_dataframe(gaps)

    elif page == "Raw Signals":
        st.subheader("Raw Signals Across Selected Time Range")
        st.caption(
            "Use raw signals for close inspection of waveforms. For long ranges, the plot is downsampled for speed; summaries and trend calculations still use the full selected data."
        )
        selectable = active_catalog[
            active_catalog["sensor_type"].isin(selected_types)
        ]["channel"].tolist()
        default_channels = selectable[: min(6, len(selectable))]
        channels = st.multiselect(
            "Channels",
            selectable,
            default=default_channels,
            help="Choose a small set of channels to compare. Mixing sensor types can put very different units/scales on the same axis.",
        )
        if channels:
            max_points = st.slider(
                "Max plotted points",
                1_000,
                100_000,
                12_000,
                1_000,
                help="Controls visual downsampling only. Raise it for more detail; lower it for faster plotting over long time ranges.",
            )
            overlay_raw_events = st.checkbox(
                "Overlay detected events",
                value=False,
                help="Add translucent event bands to the raw signal plot.",
            )
            raw_overlay_families = []
            raw_overlay_events = pd.DataFrame()
            if overlay_raw_events:
                raw_overlay_source = st.radio(
                    "Event overlay source",
                    ["All event channels", "Selected plotted channels"],
                    horizontal=True,
                    help=(
                        "Use all event channels for a stable event timeline, or only "
                        "the plotted channels to focus on the visible traces."
                    ),
                )
                raw_event_channels = (
                    traffic_candidates if raw_overlay_source == "All event channels" else channels
                )
                raw_overlay_events = get_event_families_from_store(
                    cache_dir,
                    selected_signature,
                    selected_start,
                    selected_end,
                    active_catalog,
                    raw_event_channels,
                    corr_metric,
                    corr_window,
                    min_abs_corr,
                    int(group_min_channels),
                    event_window_seconds,
                    event_threshold_sigma,
                    event_impact_ratio,
                )
                raw_overlay_families = st.multiselect(
                    "Raw plot event overlays",
                    sorted(raw_overlay_events["event_family"].unique())
                    if not raw_overlay_events.empty
                    else [],
                    default=sorted(raw_overlay_events["event_family"].unique())
                    if not raw_overlay_events.empty
                    else [],
                    help="Choose which event families to show as vertical bands on the raw signal plot.",
                )
            plot_data = query_samples(
                cache_dir, selected_start, selected_end, channels, max_points
            )
            if show_data_gaps:
                plot_data = insert_plot_breaks(plot_data, gaps, channels)
            long = plot_data.melt(
                id_vars=["timestamp", "source_file"],
                var_name="channel",
                value_name="value",
            )
            fig = px.line(
                long,
                x="timestamp",
                y="value",
                color="channel",
                hover_data=["source_file"],
            )
            fig.update_layout(height=560, legend_title_text="")
            if show_data_gaps:
                add_gap_bands(fig, gaps)
            if overlay_raw_events and raw_overlay_families:
                add_event_overlays(
                    fig,
                    raw_overlay_events[
                        raw_overlay_events["event_family"].isin(raw_overlay_families)
                    ],
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select one or more channels.")

        st.subheader("Combined Channel Summary")
        summary = session_cached(
            ("channel_summary", selected_signature),
            lambda: query_summary(cache_dir, selected_start, selected_end),
        )
        show_dataframe(summary)

    elif page == "Event Detection":
        st.subheader("Traffic, Impact, and Drawbridge-Operation Event Detection")
        st.caption(
            "Events start as rolling-RMS bursts, then are classified as traffic/vibration, boat collision or impact candidates, or drawbridge-operation-like events. "
            "Impact and behavior reports require support from correlated channel groups."
        )
        selected_event_channels = st.multiselect(
            "Event channels",
            traffic_candidates,
            default=[
                channel for channel in traffic_candidates if channel.startswith("A-")
            ]
            or traffic_candidates[:4],
            help="Start with accelerometers for traffic/vibration. Add strain or bridge channels to see whether structural response lines up with vibration bursts.",
        )
        event_setting_cols = st.columns(3)
        event_setting_cols[0].metric("RMS window", f"{event_window_seconds}s")
        event_setting_cols[1].metric("Threshold sigma", f"{event_threshold_sigma:.1f}")
        event_setting_cols[2].metric("Impact ratio", f"{event_impact_ratio:.1f}x")
        st.caption(
            "Detection rule: subtract a 2-minute rolling median baseline, compute rolling RMS, then flag RMS above "
            "`background + threshold sigma * robust spread`. A boat collision candidate is a short, high-severity, multi-channel response. "
            "A drawbridge-operation-like event is sustained coordinated response across a correlated group."
        )

        raw_events = session_cached(
            (
                "raw_events",
                selected_signature,
                tuple(selected_event_channels),
                int(event_window_seconds),
                float(event_threshold_sigma),
            ),
            lambda: query_events(
                cache_dir,
                selected_start,
                selected_end,
                selected_event_channels,
                event_window_seconds,
                event_threshold_sigma,
            ),
        )
        event_families = get_event_families_from_store(
            cache_dir,
            selected_signature,
            selected_start,
            selected_end,
            active_catalog,
            selected_event_channels,
            corr_metric,
            corr_window,
            min_abs_corr,
            int(group_min_channels),
            event_window_seconds,
            event_threshold_sigma,
            event_impact_ratio,
        )
        st.subheader("Classified Event Families")
        st.caption(
            f"Using {event_window_seconds}s RMS windows, threshold sigma "
            f"{event_threshold_sigma:.1f}, impact ratio {event_impact_ratio:.1f}x, "
            f"{corr_window} {corr_metric} correlation groups, and "
            f"{int(group_min_channels)}-channel group support."
        )
        if event_families.empty:
            show_empty_state(
                "No classified events were found for the selected range and channels.",
                "Try widening the time range, adding accelerometer channels, or lowering Threshold sigma in Event Detection Settings.",
            )
        else:
            show_dataframe(event_families)
            download_dataframe(
                "Download event families CSV",
                event_families,
                "classified_event_families.csv",
            )
        if not event_families.empty:
            st.subheader("Event Timeline")
            st.caption(
                "Each lane groups events by family. Bars span event duration and hover text shows support and severity details."
            )
            st.plotly_chart(
                event_timeline_figure(event_families),
                use_container_width=True,
            )

            family_counts = event_families.groupby("event_family", as_index=False).size()
            fig = px.bar(
                family_counts,
                x="event_family",
                y="size",
                labels={"size": "events"},
            )
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

            selected_event = st.selectbox(
                "Inspect classified event",
                event_families.index,
                format_func=lambda idx: (
                    f"{event_families.loc[idx, 'start']} - {event_families.loc[idx, 'event_family']}"
                ),
            )
            event = event_families.loc[selected_event]
            span_start = event["start"] - pd.Timedelta(seconds=10)
            span_end = event["end"] + pd.Timedelta(seconds=10)
            event_plot_channels = [
                channel
                for channel in event_channels(event)
                if channel in active_catalog["channel"].tolist()
            ][:8]
            if event["event_family"] == "Boat collision / impact candidate":
                st.warning(
                    "Boat collision / impact candidate: the highlighted sensors show "
                    "where the strongest supporting channels are physically located."
                )
            st.subheader("Selected Event Sensor Locations")
            st.plotly_chart(
                sensor_location_figure(
                    active_catalog,
                    highlight_channels=event_plot_channels,
                    title=f"Supporting Sensor Locations - {event['event_family']}",
                ),
                use_container_width=True,
            )
            show_dataframe(sensor_location_table(active_catalog, event_plot_channels))
            if event_plot_channels:
                event_data = query_samples(
                    cache_dir, span_start, span_end, event_plot_channels
                )
                long_event_data = event_data.melt(
                    id_vars=["timestamp", "source_file"],
                    var_name="channel",
                    value_name="value",
                )
                fig = px.line(
                    long_event_data,
                    x="timestamp",
                    y="value",
                    color="channel",
                    hover_data=["source_file"],
                )
                fig.add_vrect(x0=event["start"], x1=event["end"], fillcolor="red", opacity=0.18)
                if show_data_gaps:
                    add_gap_bands(fig, gaps)
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                show_empty_state(
                    "No supporting channels from this event are available in the active catalog.",
                    "Select another event or broaden the selected time range so supporting sensor metadata is present.",
                )

        with st.expander("Raw channel-level event detections"):
            if raw_events.empty:
                show_empty_state(
                    "No raw channel-level event detections were found.",
                    "Try lowering Threshold sigma, widening the range, or selecting more event channels.",
                )
            else:
                show_dataframe(raw_events)

    elif page == "Correlation Groups":
        st.subheader("Correlated Sensor Channel Groups")
        st.caption(
            "Groups are discovered from channels whose selected metric moves together over the selected time range. "
            f"These groups are used to validate reported shifts, requiring at least {int(group_min_channels)} agreeing channels."
        )
        corr_result = get_correlation_result_from_store(
            cache_dir,
            selected_signature,
            selected_start,
            selected_end,
            active_catalog,
            corr_metric,
            corr_window,
            min_abs_corr,
            int(group_min_channels),
        )
        group_cols = st.columns(4)
        group_cols[0].metric("Groups", len(corr_result.groups))
        group_cols[1].metric("Basis", corr_metric)
        group_cols[2].metric("Window", corr_window)
        group_cols[3].metric("Min |corr|", f"{min_abs_corr:.2f}")
        st.caption(
            "Lower minimum correlation values create broader groups that are more sensitive but less specific. "
            "Higher values require channels to move together more tightly before they can support the same anomaly."
        )
        if corr_result.groups.empty:
            show_empty_state(
                "No correlated groups met the current settings.",
                "Try widening the time range, lowering Minimum absolute correlation, using a larger Correlation window, or selecting sensor types with more active channels.",
            )
        else:
            show_dataframe(corr_result.groups)
        if not corr_result.matrix.empty:
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_result.matrix.to_numpy(),
                    x=corr_result.matrix.columns,
                    y=corr_result.matrix.index,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                    colorbar={"title": "corr"},
                )
            )
            fig.update_layout(height=680)
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Strongest channel pairs"):
            if corr_result.pairs.empty:
                show_empty_state(
                    "No channel-pair correlations are available.",
                    "The selected range may not contain enough non-flat channel data to compute pairwise correlations.",
                )
            else:
                show_dataframe(corr_result.pairs.head(100))

    elif page == "Anomaly Review":
        st.subheader("Anomaly Review and Reportable Shifts")
        st.caption(
            "This view separates raw event candidates from reportable behavior shifts. "
            "A shift is reported only when at least three channels in the same correlated group show compatible abnormal movement."
        )
        anomaly_cols = st.columns(2)
        shift_window = anomaly_cols[0].selectbox(
            "Shift detection window",
            ["1min", "5min", "15min", "1h"],
            index=1,
            help="Window used to look for sustained operation-like or behavior-shift changes. Larger windows suppress traffic bursts and emphasize slow behavior.",
        )
        z_threshold = anomaly_cols[1].slider(
            "Shift robust z-threshold",
            2.0,
            8.0,
            3.5,
            0.5,
            help="How far a channel-window must move from its normal distribution before it counts as abnormal. Higher values are stricter.",
        )
        shifts = get_behavior_shifts_from_store(
            cache_dir,
            selected_signature,
            selected_start,
            selected_end,
            active_catalog,
            corr_metric,
            corr_window,
            min_abs_corr,
            int(group_min_channels),
            shift_window,
            z_threshold,
        )
        impact_candidates = get_event_families_from_store(
            cache_dir,
            selected_signature,
            selected_start,
            selected_end,
            active_catalog,
            traffic_candidates,
            corr_metric,
            corr_window,
            min_abs_corr,
            int(group_min_channels),
            event_window_seconds,
            event_threshold_sigma,
            event_impact_ratio,
        )
        impact_candidates = impact_candidates[
            impact_candidates["event_family"].eq("Boat collision / impact candidate")
        ]
        review_cols = st.columns(3)
        review_cols[0].metric("Reportable shifts", len(shifts))
        review_cols[1].metric("Impact candidates", len(impact_candidates))
        review_cols[2].metric("Required channel support", int(group_min_channels))
        st.caption(
            f"Shift review uses {shift_window} windows and robust z >= {z_threshold:.1f}. "
            f"Impact review uses the shared event settings: {event_window_seconds}s RMS, "
            f"sigma {event_threshold_sigma:.1f}, impact ratio {event_impact_ratio:.1f}x."
        )

        st.subheader("Reportable Operation / Behavior Shifts")
        if shifts.empty:
            show_empty_state(
                "No reportable operation or behavior shifts met the group-support rule.",
                "Try widening the time range, lowering the shift z-threshold, or reviewing Correlation Groups to confirm enough channels are grouped together.",
            )
        else:
            show_dataframe(shifts)
            download_dataframe(
                "Download reportable shifts CSV",
                shifts,
                "reportable_behavior_shifts.csv",
            )
        st.subheader("Urgent Boat Collision / Impact Candidates")
        if impact_candidates.empty:
            show_empty_state(
                "No urgent boat collision or impact candidates were found.",
                "Try widening the time range, lowering Threshold sigma, or lowering the Impact severity ratio if you are reviewing a known incident window.",
            )
        else:
            show_dataframe(impact_candidates)
            download_dataframe(
                "Download impact candidates CSV",
                impact_candidates,
                "urgent_impact_candidates.csv",
            )
        if not impact_candidates.empty:
            impact_channels = sorted(
                {
                    channel
                    for _, row in impact_candidates.iterrows()
                    for channel in event_channels(row)
                }
            )
            st.subheader("Impact Candidate Sensor Locations")
            st.caption(
                "Highlighted sensors are channels that support one or more urgent "
                "boat collision / impact candidates in the selected range."
            )
            st.plotly_chart(
                sensor_location_figure(
                    active_catalog,
                    highlight_channels=impact_channels,
                    title="Urgent Impact Candidate Sensor Locations",
                ),
                use_container_width=True,
            )
            show_dataframe(sensor_location_table(active_catalog, impact_channels))
        anomaly_timeline = anomaly_timeline_events(shifts, impact_candidates, shift_window)
        if not anomaly_timeline.empty:
            st.subheader("Anomaly Timeline")
            st.caption(
                "Reportable shifts and urgent impact candidates are shown together so multiple events can be reviewed in context."
            )
            st.plotly_chart(
                event_timeline_figure(anomaly_timeline),
                use_container_width=True,
            )

    elif page == "Trends":
        st.subheader("Windowed Trends Across Selected Time Range")
        st.caption(
            "Trends aggregate each channel into fixed time windows. Use RMS, standard deviation, or peak-to-peak for traffic/vibration intensity; use mean for baseline drift or thermal patterns."
        )
        trend_window = st.selectbox(
            "Window",
            ["30s", "1min", "5min", "15min", "1h"],
            index=1,
            help="Aggregation interval. Smaller windows show short events; larger windows reveal hourly or daily patterns.",
        )
        features = session_cached(
            ("features", selected_signature, trend_window),
            lambda: query_features(cache_dir, selected_start, selected_end, trend_window),
        )
        trend_channels = active_catalog[
            active_catalog["sensor_type"].isin(selected_types)
        ]["channel"].tolist()
        chosen_trends = st.multiselect(
            "Trend channels",
            trend_channels,
            default=trend_channels[: min(6, len(trend_channels))],
            help="Choose channels to trend over the selected time range. Select related channels together, such as east/west or matching orientations.",
        )
        metric = st.selectbox(
            "Metric",
            ["rms", "peak_to_peak", "mean", "std", "min", "max"],
            help="Pick how each time window is summarized. The caption below updates with guidance for the selected metric.",
        )
        st.caption(METRIC_GUIDANCE[metric])
        st.caption(
            f"Trend rows are computed with {trend_window} windows over the full selected data. "
            "Changing chart overlays only changes display bands; it does not change the trend calculations."
        )
        overlay_trend_events = st.checkbox(
            "Overlay detected events on trend",
            value=False,
            help="Add event-family bands to the trend line plot.",
        )
        trend_overlay_events = pd.DataFrame()
        trend_overlay_families = []
        if overlay_trend_events:
            trend_overlay_source = st.radio(
                "Trend overlay source",
                ["All event channels", "Selected trend channels"],
                horizontal=True,
                help=(
                    "Use all event channels for a stable event timeline, or only "
                    "the selected trend channels to focus overlays on those sensors."
                ),
            )
            trend_event_channels = (
                traffic_candidates
                if trend_overlay_source == "All event channels"
                else chosen_trends
            )
            trend_overlay_events = get_event_families_from_store(
                cache_dir,
                selected_signature,
                selected_start,
                selected_end,
                active_catalog,
                trend_event_channels,
                corr_metric,
                corr_window,
                min_abs_corr,
                int(group_min_channels),
                event_window_seconds,
                event_threshold_sigma,
                event_impact_ratio,
            )
            trend_overlay_families = st.multiselect(
                "Trend plot event overlays",
                sorted(trend_overlay_events["event_family"].unique())
                if not trend_overlay_events.empty
                else [],
                default=sorted(trend_overlay_events["event_family"].unique())
                if not trend_overlay_events.empty
                else [],
                help="Choose which event families to show as vertical bands on the trend plot.",
            )
        if chosen_trends and not features.empty:
            chart_data = features[features["channel"].isin(chosen_trends)]
            if show_data_gaps:
                chart_data = insert_metric_plot_breaks(chart_data, gaps, metric)
            fig = px.line(chart_data, x="timestamp", y=metric, color="channel")
            fig.update_layout(height=520, legend_title_text="")
            if show_data_gaps:
                add_gap_bands(fig, gaps)
            if overlay_trend_events and trend_overlay_families:
                add_event_overlays(
                    fig,
                    trend_overlay_events[
                        trend_overlay_events["event_family"].isin(trend_overlay_families)
                    ],
                )
            st.plotly_chart(fig, use_container_width=True)

            heatmap_data = features[features["channel"].isin(chosen_trends)].pivot_table(
                index="channel", columns="timestamp", values=metric
            )
            fig = go.Figure(
                data=go.Heatmap(
                    z=heatmap_data.to_numpy(),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale="Viridis",
                    colorbar={"title": metric},
                )
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        elif not chosen_trends:
            show_empty_state(
                "No trend channels are selected.",
                "Select one or more channels, or choose additional sensor types in the sidebar.",
            )
        else:
            show_empty_state(
                "No trend feature rows were found for the selected range.",
                "Try widening the time range or choosing a larger aggregation window.",
            )

    elif page == "Sensor Health":
        st.subheader("Sensor Health Across Selected Files")
        st.caption(
            "Health flags highlight channels that may need skepticism before interpretation, such as inactive sensors, flatlines, or extreme bridge values."
        )
        health = session_cached(
            ("sensor_health", selected_signature),
            lambda: query_health(cache_dir, selected_start, selected_end),
        )
        if health.empty or "flags" not in health:
            show_empty_state(
                "No sensor health rows were found for the selected range.",
                "Try widening the time range or confirm that the selected files ingested successfully.",
            )
        else:
            flag_options = sorted(health["flags"].dropna().unique())
            flag_filter = st.multiselect(
                "Flags",
                flag_options,
                default=flag_options,
                help="Filter the health table to focus on suspicious channels or confirm which channels are okay.",
            )
            filtered_health = health[health["flags"].isin(flag_filter)]
            if filtered_health.empty:
                show_empty_state(
                    "No sensor health rows match the selected flags.",
                    "Select additional flags or clear the filter to restore the health table.",
                )
            else:
                show_dataframe(filtered_health)
                download_dataframe(
                    "Download sensor health CSV",
                    filtered_health,
                    "sensor_health.csv",
                )


def detect_plot_gaps(samples: pd.DataFrame, threshold_seconds: float) -> pd.DataFrame:
    if samples.empty or "timestamp" not in samples:
        return _empty_gaps()
    frame = samples[["timestamp", "source_file"]].dropna(subset=["timestamp"]).copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["duration_s"] = frame["timestamp"].diff().dt.total_seconds()
    positive_deltas = frame.loc[frame["duration_s"] > 0, "duration_s"]
    expected_interval = positive_deltas.quantile(0.10) if not positive_deltas.empty else 0.0
    expected_threshold = expected_interval * 10 if expected_interval else 0.0
    gap_threshold = max(float(threshold_seconds), float(expected_threshold))

    gaps = frame.loc[
        frame["duration_s"] > gap_threshold,
        ["timestamp", "source_file", "duration_s"],
    ].copy()
    if gaps.empty:
        return _empty_gaps()
    gaps["gap_start"] = frame["timestamp"].shift().loc[gaps.index]
    gaps["gap_end"] = gaps["timestamp"]
    gaps["previous_file"] = frame["source_file"].shift().loc[gaps.index]
    gaps["next_file"] = gaps["source_file"]
    return gaps[
        ["gap_start", "gap_end", "duration_s", "previous_file", "next_file"]
    ].reset_index(drop=True)


def detect_file_index_gaps(files: pd.DataFrame, threshold_seconds: float) -> pd.DataFrame:
    if files.empty or "sample_start" not in files or "sample_end" not in files:
        return _empty_gaps()
    frame = files[["file", "sample_start", "sample_end"]].dropna().copy()
    if frame.empty:
        return _empty_gaps()
    frame["sample_start"] = pd.to_datetime(frame["sample_start"])
    frame["sample_end"] = pd.to_datetime(frame["sample_end"])
    frame = frame.sort_values("sample_start").reset_index(drop=True)
    frame["previous_end"] = frame["sample_end"].shift()
    frame["previous_file"] = frame["file"].shift()
    frame["duration_s"] = (
        frame["sample_start"] - frame["previous_end"]
    ).dt.total_seconds()
    gaps = frame.loc[
        frame["duration_s"] > float(threshold_seconds),
        ["previous_end", "sample_start", "duration_s", "previous_file", "file"],
    ].copy()
    if gaps.empty:
        return _empty_gaps()
    gaps = gaps.rename(
        columns={
            "previous_end": "gap_start",
            "sample_start": "gap_end",
            "file": "next_file",
        }
    )
    return gaps[
        ["gap_start", "gap_end", "duration_s", "previous_file", "next_file"]
    ].reset_index(drop=True)


def insert_plot_breaks(
    frame: pd.DataFrame, gaps: pd.DataFrame, value_columns: list[str]
) -> pd.DataFrame:
    if frame.empty or gaps.empty:
        return frame
    break_rows = []
    for _, gap in gaps.iterrows():
        row = {column: np.nan for column in value_columns}
        row["timestamp"] = gap["gap_start"] + pd.Timedelta(microseconds=1)
        row["source_file"] = "data gap"
        break_rows.append(row)
    if not break_rows:
        return frame
    return (
        pd.concat([frame, pd.DataFrame(break_rows)], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def insert_metric_plot_breaks(
    frame: pd.DataFrame, gaps: pd.DataFrame, metric: str
) -> pd.DataFrame:
    if frame.empty or gaps.empty:
        return frame
    break_rows = []
    for channel in sorted(frame["channel"].dropna().unique()):
        for _, gap in gaps.iterrows():
            break_rows.append(
                {
                    "timestamp": gap["gap_start"] + pd.Timedelta(microseconds=1),
                    "channel": channel,
                    metric: np.nan,
                }
            )
    if not break_rows:
        return frame
    return (
        pd.concat([frame, pd.DataFrame(break_rows)], ignore_index=True)
        .sort_values(["channel", "timestamp"])
        .reset_index(drop=True)
    )


def add_gap_bands(fig: go.Figure, gaps: pd.DataFrame) -> None:
    if gaps.empty:
        return
    for _, gap in gaps.iterrows():
        fig.add_vrect(
            x0=gap["gap_start"],
            x1=gap["gap_end"],
            fillcolor="#94a3b8",
            opacity=0.16,
            line_width=0,
            layer="below",
        )


def event_timeline_figure(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        fig = go.Figure()
        fig.update_layout(height=260)
        return fig
    timeline = normalize_event_timeline(events)
    fig = px.timeline(
        timeline,
        x_start="start",
        x_end="end",
        y="event_family",
        color="event_family",
        color_discrete_map=EVENT_COLORS,
        hover_data=[
            "priority",
            "duration_s",
            "supporting_channels",
            "same_group_channels",
            "peak_ratio",
            "channels",
            "rationale",
        ],
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(
        height=max(320, 78 * timeline["event_family"].nunique()),
        legend_title_text="",
        margin={"l": 16, "r": 16, "t": 24, "b": 24},
    )
    return fig


def normalize_event_timeline(events: pd.DataFrame) -> pd.DataFrame:
    timeline = events.copy()
    if "start" not in timeline and "timestamp" in timeline:
        timeline["start"] = timeline["timestamp"]
    if "end" not in timeline:
        timeline["end"] = timeline["start"] + pd.Timedelta(seconds=1)
    timeline["start"] = pd.to_datetime(timeline["start"])
    timeline["end"] = pd.to_datetime(timeline["end"])
    zero_duration = timeline["end"] <= timeline["start"]
    timeline.loc[zero_duration, "end"] = timeline.loc[zero_duration, "start"] + pd.Timedelta(seconds=1)
    defaults = {
        "priority": "",
        "duration_s": (timeline["end"] - timeline["start"]).dt.total_seconds(),
        "supporting_channels": np.nan,
        "same_group_channels": np.nan,
        "peak_ratio": np.nan,
        "channels": "",
        "rationale": "",
    }
    for column, default in defaults.items():
        if column not in timeline:
            timeline[column] = default
    return timeline


def add_event_overlays(fig: go.Figure, events: pd.DataFrame) -> None:
    if events.empty:
        return
    for _, event in normalize_event_timeline(events).iterrows():
        family = event["event_family"]
        fig.add_vrect(
            x0=event["start"],
            x1=event["end"],
            fillcolor=EVENT_COLORS.get(family, DEFAULT_EVENT_COLOR),
            opacity=event_overlay_opacity(family),
            line_width=1 if family == "Group-confirmed behavior shift" else 0,
            line_color="#111827",
            layer="below",
        )


def event_overlay_opacity(family: str) -> float:
    if family == "Boat collision / impact candidate":
        return 0.28
    if family == "Group-confirmed behavior shift":
        return 0.22
    if family == "Drawbridge operation-like event":
        return 0.18
    return 0.14


def anomaly_timeline_events(
    shifts: pd.DataFrame, impact_candidates: pd.DataFrame, shift_window: str
) -> pd.DataFrame:
    frames = []
    if not shifts.empty:
        shift_events = shifts.copy()
        shift_events["start"] = pd.to_datetime(shift_events["timestamp"])
        shift_events["end"] = shift_events["start"] + pd.to_timedelta(shift_window)
        shift_events["priority"] = "reportable"
        shift_events["duration_s"] = (
            shift_events["end"] - shift_events["start"]
        ).dt.total_seconds()
        shift_events["same_group_channels"] = shift_events["supporting_channels"]
        shift_events["peak_ratio"] = np.nan
        frames.append(shift_events)
    if not impact_candidates.empty:
        frames.append(impact_candidates.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _empty_gaps() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["gap_start", "gap_end", "duration_s", "previous_file", "next_file"]
    )


def add_relative_paths(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    if frame.empty or "path" not in frame:
        return frame
    display = frame.copy()

    def relative_path(value: str) -> str:
        try:
            return str(Path(value).resolve().relative_to(root))
        except (OSError, ValueError):
            return Path(value).name

    display["relative_path"] = display["path"].map(relative_path)
    return display


def dataframe_config(frame: pd.DataFrame) -> dict:
    config = {}
    for column in frame.columns:
        if column in {"timestamp", "start", "end", "sample_start", "sample_end"}:
            config[column] = st.column_config.DatetimeColumn(column, format="YYYY-MM-DD HH:mm:ss")
        elif column in {"size_mb"}:
            config[column] = st.column_config.NumberColumn(column, format="%.2f MB")
        elif column in {"duration_s"}:
            config[column] = st.column_config.NumberColumn(column, format="%.1f s")
        elif column in {"peak_ratio", "z_score", "max_abs_z", "avg_abs_correlation", "correlation", "abs_correlation"}:
            config[column] = st.column_config.NumberColumn(column, format="%.2f")
        elif column in {"sample_rows", "samples", "supporting_channels", "same_group_channels", "channel_count", "files_present"}:
            config[column] = st.column_config.NumberColumn(column, format="%d")
    return config


def show_dataframe(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config=dataframe_config(frame),
        **kwargs,
    )


def download_dataframe(label: str, frame: pd.DataFrame, filename: str) -> None:
    if frame.empty:
        return
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def show_empty_state(message: str, suggestions: str) -> None:
    st.info(message)
    st.caption(suggestions)


def _merge_catalogs(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    sort_cols = ["active", "array_column", "channel"]
    grouped = (
        catalog.sort_values(sort_cols, ascending=[False, True, True])
        .groupby("channel", as_index=False)
        .agg(
            active=("active", "any"),
            sensor_type=("sensor_type", "first"),
            sensor_name=("sensor_name", "first"),
            hardware_location=("hardware_location", "first"),
            gain=("gain", "first"),
            offset=("offset", "first"),
            cal_factor=("cal_factor", "first"),
            excitation_mode=("excitation_mode", "first"),
            array_column=("array_column", "first"),
            dtype=("dtype", "first"),
            samples=("samples", "sum"),
            files_present=("source_file", "nunique"),
        )
    )
    return grouped.sort_values(["active", "array_column", "channel"], ascending=[False, True, True])


def _downsample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    return frame.iloc[::step].copy()


def _format_duration(delta: pd.Timedelta) -> str:
    seconds = int(delta.total_seconds())
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


if __name__ == "__main__":
    main()
