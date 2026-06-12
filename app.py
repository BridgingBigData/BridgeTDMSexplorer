from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tdms_bridge.parser import (
    channel_summary,
    detect_events,
    discover_analysis_files,
    feature_windows,
    parse_tdms,
    sensor_health,
)
from tdms_bridge.ml import (
    classify_event_families,
    correlation_groups,
    detect_operation_and_behavior_shifts,
)


ROOT = Path(__file__).resolve().parent

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


st.set_page_config(
    page_title="Bridge TDMS Explorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_file_table(folder: str, refresh_token: int) -> pd.DataFrame:
    return discover_analysis_files(Path(folder))


@st.cache_data(show_spinner="Parsing TDMS file...")
def load_parsed(path: str):
    return parse_tdms(Path(path))


@st.cache_data(show_spinner="Combining selected TDMS files...")
def load_combined(paths: tuple[str, ...]):
    parsed_files = [load_parsed(path) for path in paths]
    if not parsed_files:
        return None

    samples = []
    catalogs = []
    for parsed in parsed_files:
        sample_frame = parsed.samples.copy()
        sample_frame["source_file"] = parsed.path.name
        samples.append(sample_frame)

        catalog_frame = parsed.sensor_catalog.copy()
        catalog_frame["source_file"] = parsed.path.name
        catalogs.append(catalog_frame)

    combined_samples = (
        pd.concat(samples, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined_catalog = _merge_catalogs(pd.concat(catalogs, ignore_index=True))
    file_properties = parsed_files[0].file_properties
    group_properties = parsed_files[0].group_properties
    return SimpleNamespace(
        path=Path(paths[0]),
        sha256="",
        file_properties=file_properties,
        group_properties=group_properties,
        sensor_catalog=combined_catalog,
        samples=combined_samples,
        file_count=len(parsed_files),
    )


@st.cache_data(show_spinner=False)
def cached_summary(paths: tuple[str, ...]) -> pd.DataFrame:
    return channel_summary(load_combined(paths))


@st.cache_data(show_spinner="Computing trend windows...")
def cached_features(paths: tuple[str, ...], window: str) -> pd.DataFrame:
    return feature_windows(load_combined(paths), window)


@st.cache_data(show_spinner="Checking sensor health...")
def cached_health(paths: tuple[str, ...]) -> pd.DataFrame:
    return sensor_health(load_combined(paths))


@st.cache_data(show_spinner="Detecting traffic-like events...")
def cached_events(
    paths: tuple[str, ...], channels: list[str], window_seconds: int, threshold_sigma: float
) -> pd.DataFrame:
    return detect_events(load_combined(paths), channels, window_seconds, threshold_sigma)


@st.cache_data(show_spinner="Finding correlated sensor groups...")
def cached_correlation_groups(
    paths: tuple[str, ...],
    metric: str,
    window: str,
    min_abs_corr: float,
    min_group_size: int,
):
    return correlation_groups(
        load_combined(paths), metric, window, min_abs_corr, min_group_size
    )


@st.cache_data(show_spinner="Classifying traffic, impact, and operation candidates...")
def cached_event_families(
    paths: tuple[str, ...],
    channels: list[str],
    corr_metric: str,
    corr_window: str,
    min_abs_corr: float,
    group_min_channels: int,
    window_seconds: int,
    threshold_sigma: float,
    impact_ratio: float,
) -> pd.DataFrame:
    combined = load_combined(paths)
    groups = correlation_groups(
        combined, corr_metric, corr_window, min_abs_corr, group_min_channels
    ).groups
    return classify_event_families(
        combined,
        groups,
        channels,
        window_seconds,
        threshold_sigma,
        impact_ratio,
        group_min_channels,
    )


@st.cache_data(show_spinner="Detecting group-confirmed operation and behavior shifts...")
def cached_behavior_shifts(
    paths: tuple[str, ...],
    corr_metric: str,
    corr_window: str,
    min_abs_corr: float,
    group_min_channels: int,
    shift_window: str,
    z_threshold: float,
) -> pd.DataFrame:
    combined = load_combined(paths)
    groups = correlation_groups(
        combined, corr_metric, corr_window, min_abs_corr, group_min_channels
    ).groups
    return detect_operation_and_behavior_shifts(
        combined, groups, shift_window, z_threshold, group_min_channels
    )


@st.cache_data(show_spinner=False)
def cached_plot_gaps(paths: tuple[str, ...], threshold_seconds: float) -> pd.DataFrame:
    return detect_plot_gaps(load_combined(paths).samples, threshold_seconds)


def main() -> None:
    st.title("Bridge TDMS Explorer")

    with st.sidebar:
        folder = st.text_input(
            "TDMS folder",
            value=str(ROOT),
            help="Folder containing TDMS files. The app uses normal data files and ignores version copies and decimated files.",
        )
        if "refresh_token" not in st.session_state:
            st.session_state.refresh_token = 0
        if st.button(
            "Rescan folder",
            use_container_width=True,
            help="Refresh the file list after adding, removing, or moving TDMS files.",
        ):
            st.cache_data.clear()
            st.session_state.refresh_token += 1

        file_table = load_file_table(folder, st.session_state.refresh_token)
        if file_table.empty:
            st.warning("No .tdms files found.")
            return

        included_files = file_table[file_table["included"]].copy()
        ignored_files = file_table[~file_table["included"]].copy()
        st.caption(
            f"Found {len(file_table)} TDMS files. "
            f"Using {len(included_files)} normal files; ignoring "
            f"{len(ignored_files)} version/decimate files."
        )
        if included_files.empty:
            st.warning("No non-version, non-decimate TDMS files with filename timestamps found.")
            return

        min_time = included_files["timestamp"].min().to_pydatetime()
        max_time = included_files["timestamp"].max().to_pydatetime()
        st.markdown("**Detected Filename Time Range**")
        st.caption(f"{min_time:%Y-%m-%d %H:%M:%S} to {max_time:%Y-%m-%d %H:%M:%S}")

        with st.expander("Analysis Time Range", expanded=True):
            start_date = st.date_input(
                "Start date",
                value=min_time.date(),
                min_value=min_time.date(),
                max_value=max_time.date(),
                help="First recording start date to include, based on timestamps in file names.",
            )
            start_time = st.time_input(
                "Start time",
                value=min_time.time(),
                help="First recording start time to include. Files starting before this time are excluded.",
            )
            end_date = st.date_input(
                "End date",
                value=max_time.date(),
                min_value=min_time.date(),
                max_value=max_time.date(),
                help="Last recording start date to include, based on timestamps in file names.",
            )
            end_time = st.time_input(
                "End time",
                value=max_time.time(),
                help="Last recording start time to include. Files starting after this time are excluded.",
            )

        selected_start = datetime.combine(start_date, start_time)
        selected_end = datetime.combine(end_date, end_time)
        if selected_start > selected_end:
            st.error("Start time must be before end time.")
            return
        selected_files = included_files[
            (included_files["timestamp"] >= pd.Timestamp(selected_start))
            & (included_files["timestamp"] <= pd.Timestamp(selected_end))
        ].copy()
        st.caption(f"Selected {len(selected_files)} file(s).")

        if selected_files.empty:
            st.warning("No files fall inside the selected time range.")
            return

        selected_paths = tuple(selected_files["path"].tolist())
        combined = load_combined(selected_paths)
        active_catalog = combined.sensor_catalog[combined.sensor_catalog["active"]]
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

        page = st.radio(
            "View",
            [
                "Files",
                "Raw Signals",
                "Event Detection",
                "Correlation Groups",
                "Anomaly Review",
                "Trends",
                "Sensor Health",
            ],
            help="Only the selected view is computed. This keeps display changes from recomputing every model and plot.",
        )

        needs_gaps = page == "Files" or (
            show_data_gaps and page in {"Raw Signals", "Event Detection", "Trends"}
        )
        gaps = (
            cached_plot_gaps(selected_paths, float(gap_threshold_seconds))
            if needs_gaps
            else _empty_gaps()
        )

    metadata_cols = st.columns(6)
    metadata_cols[0].metric("Normal files", len(included_files))
    metadata_cols[1].metric("Selected files", len(selected_files))
    metadata_cols[2].metric("Ignored files", len(ignored_files))
    metadata_cols[3].metric("Active channels", len(active_catalog))
    metadata_cols[4].metric("Rows", f"{len(combined.samples):,}")
    metadata_cols[5].metric(
        "Selected span",
        _format_duration(
            combined.samples["timestamp"].max() - combined.samples["timestamp"].min()
        ),
    )

    st.caption(
        "Detected file-name range: "
        f"{min_time:%Y-%m-%d %H:%M:%S} to {max_time:%Y-%m-%d %H:%M:%S}. "
        "Analysis excludes version copies and decimated files."
    )

    if page == "Files":
        st.subheader("Selected Normal Files")
        st.caption(
            "These are the normal TDMS recordings included in the current analysis range. "
            "Version copies and decimated files are listed separately and excluded."
        )
        display_cols = ["file", "timestamp", "size_mb", "included"]
        st.dataframe(
            selected_files[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Ignored Files")
        ignored_cols = ["file", "timestamp", "size_mb", "ignored_reason"]
        st.dataframe(
            ignored_files[ignored_cols],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Sensor Catalog Across Selection")
        st.dataframe(combined.sensor_catalog, use_container_width=True, hide_index=True)

        st.subheader("Detected Gaps")
        st.caption(
            "Positive timestamp gaps above the display threshold are treated as plot discontinuities. "
            "They are shaded in charts when data-gap display is enabled."
        )
        st.dataframe(gaps, use_container_width=True, hide_index=True)

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
                raw_overlay_events = cached_event_families(
                    selected_paths,
                    channels,
                    corr_metric,
                    corr_window,
                    min_abs_corr,
                    int(group_min_channels),
                    5,
                    4.0,
                    3.0,
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
            plot_data = _downsample(
                combined.samples[["timestamp", "source_file", *channels]], max_points
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
        st.dataframe(cached_summary(selected_paths), use_container_width=True, hide_index=True)

    elif page == "Event Detection":
        st.subheader("Traffic, Impact, and Drawbridge-Operation Event Detection")
        st.caption(
            "Events start as rolling-RMS bursts, then are classified as traffic/vibration, boat collision or impact candidates, or drawbridge-operation-like events. "
            "Impact and behavior reports require support from correlated channel groups."
        )
        traffic_candidates = active_catalog[
            active_catalog["sensor_type"].isin(["Accelerometer", "Quarterarm", "Half Bridge I"])
        ]["channel"].tolist()
        event_channels = st.multiselect(
            "Event channels",
            traffic_candidates,
            default=[
                channel for channel in traffic_candidates if channel.startswith("A-")
            ]
            or traffic_candidates[:4],
            help="Start with accelerometers for traffic/vibration. Add strain or bridge channels to see whether structural response lines up with vibration bursts.",
        )
        event_cols = st.columns(2)
        window_seconds = event_cols[0].slider(
            "RMS window seconds",
            1,
            20,
            5,
            help="Length of the rolling RMS window. Shorter windows catch sharp bursts; longer windows smooth activity and favor sustained events.",
        )
        threshold_sigma = event_cols[1].slider(
            "Threshold sigma",
            2.0,
            8.0,
            4.0,
            0.5,
            help="Sensitivity threshold above background RMS. Lower values find more events and more false positives; higher values keep only stronger bursts.",
        )
        impact_ratio = st.slider(
            "Impact severity ratio",
            1.5,
            8.0,
            3.0,
            0.5,
            help="A boat collision / impact candidate needs a peak RMS at least this many times the event threshold, plus three-channel correlated support.",
        )
        st.caption(
            "Detection rule: subtract a 2-minute rolling median baseline, compute rolling RMS, then flag RMS above "
            "`background + threshold sigma * robust spread`. A boat collision candidate is a short, high-severity, multi-channel response. "
            "A drawbridge-operation-like event is sustained coordinated response across a correlated group."
        )

        raw_events = cached_events(
            selected_paths, event_channels, window_seconds, threshold_sigma
        )
        event_families = cached_event_families(
            selected_paths,
            event_channels,
            corr_metric,
            corr_window,
            min_abs_corr,
            int(group_min_channels),
            window_seconds,
            threshold_sigma,
            impact_ratio,
        )
        st.subheader("Classified Event Families")
        st.dataframe(event_families, use_container_width=True, hide_index=True)
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
                for channel in str(event["channels"]).split(", ")
                if channel in combined.samples.columns
            ][:8]
            event_data = combined.samples[
                (combined.samples["timestamp"] >= span_start)
                & (combined.samples["timestamp"] <= span_end)
            ][["timestamp", *event_plot_channels]]
            long_event_data = event_data.melt(
                id_vars="timestamp", var_name="channel", value_name="value"
            )
            fig = px.line(long_event_data, x="timestamp", y="value", color="channel")
            fig.add_vrect(x0=event["start"], x1=event["end"], fillcolor="red", opacity=0.18)
            if show_data_gaps:
                add_gap_bands(fig, gaps)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw channel-level event detections"):
            st.dataframe(raw_events, use_container_width=True, hide_index=True)

    elif page == "Correlation Groups":
        st.subheader("Correlated Sensor Channel Groups")
        st.caption(
            "Groups are discovered from channels whose selected metric moves together over the selected time range. "
            "These groups are used to validate reported shifts, requiring at least three agreeing channels by default."
        )
        corr_result = cached_correlation_groups(
            selected_paths,
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
        st.dataframe(corr_result.groups, use_container_width=True, hide_index=True)
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
            st.dataframe(
                corr_result.pairs.head(100),
                use_container_width=True,
                hide_index=True,
            )

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
        shifts = cached_behavior_shifts(
            selected_paths,
            corr_metric,
            corr_window,
            min_abs_corr,
            int(group_min_channels),
            shift_window,
            z_threshold,
        )
        impact_candidates = cached_event_families(
            selected_paths,
            traffic_candidates,
            corr_metric,
            corr_window,
            min_abs_corr,
            int(group_min_channels),
            5,
            4.0,
            3.0,
        )
        impact_candidates = impact_candidates[
            impact_candidates["event_family"].eq("Boat collision / impact candidate")
        ]
        review_cols = st.columns(3)
        review_cols[0].metric("Reportable shifts", len(shifts))
        review_cols[1].metric("Impact candidates", len(impact_candidates))
        review_cols[2].metric("Required channel support", int(group_min_channels))

        st.subheader("Reportable Operation / Behavior Shifts")
        st.dataframe(shifts, use_container_width=True, hide_index=True)
        st.subheader("Urgent Boat Collision / Impact Candidates")
        st.dataframe(impact_candidates, use_container_width=True, hide_index=True)
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
        features = cached_features(selected_paths, trend_window)
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
        overlay_trend_events = st.checkbox(
            "Overlay detected events on trend",
            value=False,
            help="Add event-family bands to the trend line plot.",
        )
        trend_overlay_events = pd.DataFrame()
        trend_overlay_families = []
        if overlay_trend_events:
            trend_overlay_events = cached_event_families(
                selected_paths,
                chosen_trends,
                corr_metric,
                corr_window,
                min_abs_corr,
                int(group_min_channels),
                5,
                4.0,
                3.0,
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

    elif page == "Sensor Health":
        st.subheader("Sensor Health Across Selected Files")
        st.caption(
            "Health flags highlight channels that may need skepticism before interpretation, such as inactive sensors, flatlines, or extreme bridge values."
        )
        health = cached_health(selected_paths)
        flag_filter = st.multiselect(
            "Flags",
            sorted(health["flags"].unique()),
            default=sorted(health["flags"].unique()),
            help="Filter the health table to focus on suspicious channels or confirm which channels are okay.",
        )
        st.dataframe(
            health[health["flags"].isin(flag_filter)],
            use_container_width=True,
            hide_index=True,
        )


def detect_plot_gaps(samples: pd.DataFrame, threshold_seconds: float) -> pd.DataFrame:
    if samples.empty or "timestamp" not in samples:
        return _empty_gaps()
    frame = samples[["timestamp", "source_file"]].dropna(subset=["timestamp"]).copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    deltas = frame["timestamp"].diff().dt.total_seconds()
    positive_deltas = deltas[deltas > 0]
    expected_interval = positive_deltas.quantile(0.10) if not positive_deltas.empty else 0.0
    expected_threshold = expected_interval * 10 if expected_interval else threshold_seconds
    gap_threshold = min(threshold_seconds, expected_threshold) if expected_threshold else threshold_seconds

    rows = []
    for idx in range(1, len(frame)):
        delta = (frame.loc[idx, "timestamp"] - frame.loc[idx - 1, "timestamp"]).total_seconds()
        if delta <= 0 or delta <= gap_threshold:
            continue
        previous_file = frame.loc[idx - 1, "source_file"]
        next_file = frame.loc[idx, "source_file"]
        rows.append(
            {
                "gap_start": frame.loc[idx - 1, "timestamp"],
                "gap_end": frame.loc[idx, "timestamp"],
                "duration_s": float(delta),
                "previous_file": previous_file,
                "next_file": next_file,
            }
        )
    return pd.DataFrame(rows) if rows else _empty_gaps()


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
