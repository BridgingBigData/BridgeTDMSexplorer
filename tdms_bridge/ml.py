from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tdms_bridge.parser import detect_events, feature_windows


@dataclass(frozen=True)
class CorrelationResult:
    matrix: pd.DataFrame
    groups: pd.DataFrame
    pairs: pd.DataFrame


def correlation_groups(
    parsed,
    metric: str = "rms",
    window: str = "5min",
    min_abs_corr: float = 0.75,
    min_group_size: int = 3,
) -> CorrelationResult:
    features = feature_windows(parsed, window)
    return correlation_groups_from_features(
        features, parsed.sensor_catalog, metric, min_abs_corr, min_group_size
    )


def correlation_groups_from_features(
    features: pd.DataFrame,
    catalog: pd.DataFrame,
    metric: str = "rms",
    min_abs_corr: float = 0.75,
    min_group_size: int = 3,
) -> CorrelationResult:
    if features.empty or metric not in features:
        return CorrelationResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    wide = features.pivot_table(index="timestamp", columns="channel", values=metric)
    wide = wide.dropna(axis=1, how="all")
    wide = wide.loc[:, wide.std(ddof=0) > 0]
    if wide.shape[1] < 2:
        return CorrelationResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    matrix = wide.corr(method="spearman").fillna(0.0)
    pairs = _correlation_pairs(matrix)
    groups = _connected_correlation_groups(
        matrix, catalog, min_abs_corr, min_group_size
    )
    return CorrelationResult(matrix, groups, pairs)


def classify_event_families(
    parsed,
    groups: pd.DataFrame,
    channels: list[str],
    window_seconds: int = 5,
    threshold_sigma: float = 4.0,
    impact_ratio: float = 3.0,
    group_min_channels: int = 3,
    merge_gap_seconds: int = 10,
) -> pd.DataFrame:
    events = detect_events(parsed, channels, window_seconds, threshold_sigma)
    return classify_event_families_from_events(
        events, groups, impact_ratio, group_min_channels, merge_gap_seconds
    )


def classify_event_families_from_events(
    events: pd.DataFrame,
    groups: pd.DataFrame,
    impact_ratio: float = 3.0,
    group_min_channels: int = 3,
    merge_gap_seconds: int = 10,
) -> pd.DataFrame:
    if events.empty:
        return _empty_event_families()

    channel_to_group = _channel_to_group(groups)
    rows = []
    for cluster_id, cluster in enumerate(_merge_events(events, merge_gap_seconds), start=1):
        support_channels = sorted(cluster["channel"].unique())
        support_count = len(support_channels)
        group_ids = sorted(
            {channel_to_group.get(channel) for channel in support_channels}
            - {None}
        )
        grouped_support = max(
            (
                len(
                    [
                        channel
                        for channel in support_channels
                        if channel_to_group.get(channel) == group_id
                    ]
                )
                for group_id in group_ids
            ),
            default=0,
        )
        peak_ratio = float(
            (cluster["peak_rms"] / cluster["threshold"].replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .max()
        )
        start = cluster["start"].min()
        end = cluster["end"].max()
        duration_s = (end - start).total_seconds()
        family, priority, rationale = _classify_event(
            support_count=support_count,
            grouped_support=grouped_support,
            peak_ratio=peak_ratio,
            duration_s=duration_s,
            impact_ratio=impact_ratio,
            group_min_channels=group_min_channels,
        )
        rows.append(
            {
                "event_id": cluster_id,
                "event_family": family,
                "priority": priority,
                "start": start,
                "end": end,
                "duration_s": duration_s,
                "supporting_channels": support_count,
                "same_group_channels": grouped_support,
                "groups": ", ".join(str(group_id) for group_id in group_ids),
                "peak_ratio": peak_ratio,
                "channels": ", ".join(support_channels),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def detect_operation_and_behavior_shifts(
    parsed,
    groups: pd.DataFrame,
    window: str = "5min",
    z_threshold: float = 3.5,
    group_min_channels: int = 3,
) -> pd.DataFrame:
    features = feature_windows(parsed, window)
    return detect_operation_and_behavior_shifts_from_features(
        features, groups, z_threshold, group_min_channels
    )


def detect_operation_and_behavior_shifts_from_features(
    features: pd.DataFrame,
    groups: pd.DataFrame,
    z_threshold: float = 3.5,
    group_min_channels: int = 3,
) -> pd.DataFrame:
    if features.empty:
        return _empty_shift_table()

    channel_to_group = _channel_to_group(groups)
    scored = []
    for metric in ["mean", "rms", "peak_to_peak"]:
        if metric not in features:
            continue
        for channel, channel_frame in features.groupby("channel"):
            values = channel_frame[metric]
            z = _robust_z(values)
            for timestamp, score, value in zip(channel_frame["timestamp"], z, values):
                if np.isfinite(score) and abs(score) >= z_threshold:
                    scored.append(
                        {
                            "timestamp": timestamp,
                            "channel": channel,
                            "group_id": channel_to_group.get(channel),
                            "metric": metric,
                            "z_score": float(score),
                            "value": float(value),
                        }
                    )

    if not scored:
        return _empty_shift_table()

    scored_frame = pd.DataFrame(scored)
    rows = []
    for (timestamp, group_id, metric), frame in scored_frame.groupby(
        ["timestamp", "group_id", "metric"], dropna=False
    ):
        support_channels = sorted(frame["channel"].unique())
        if pd.isna(group_id) or len(support_channels) < group_min_channels:
            continue
        direction = "up" if frame["z_score"].median() > 0 else "down"
        family = (
            "Drawbridge operation-like event"
            if metric == "mean"
            else "Group-confirmed behavior shift"
        )
        rows.append(
            {
                "timestamp": timestamp,
                "event_family": family,
                "group_id": int(group_id),
                "metric": metric,
                "direction": direction,
                "supporting_channels": len(support_channels),
                "max_abs_z": float(frame["z_score"].abs().max()),
                "channels": ", ".join(support_channels),
                "reportable": True,
                "rationale": (
                    f"{len(support_channels)} correlated channels exceeded robust "
                    f"z-score {z_threshold:g} on {metric}."
                ),
            }
        )
    if not rows:
        return _empty_shift_table()
    return pd.DataFrame(rows).sort_values(["timestamp", "event_family"])


def _correlation_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    channels = list(matrix.columns)
    for index, left in enumerate(channels):
        for right in channels[index + 1 :]:
            corr = float(matrix.loc[left, right])
            rows.append(
                {
                    "channel_a": left,
                    "channel_b": right,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def _connected_correlation_groups(
    matrix: pd.DataFrame,
    catalog: pd.DataFrame,
    min_abs_corr: float,
    min_group_size: int,
) -> pd.DataFrame:
    channels = list(matrix.columns)
    neighbors = {channel: set() for channel in channels}
    for left in channels:
        for right in channels:
            if left == right:
                continue
            if abs(float(matrix.loc[left, right])) >= min_abs_corr:
                neighbors[left].add(right)

    rows = []
    seen = set()
    group_id = 1
    sensor_lookup = catalog.set_index("channel")["sensor_type"].to_dict()
    for channel in channels:
        if channel in seen:
            continue
        stack = [channel]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbors[current] - component)
        seen.update(component)
        if len(component) < min_group_size:
            continue
        component_list = sorted(component)
        submatrix = matrix.loc[component_list, component_list].abs()
        avg_abs_corr = (
            submatrix.where(~np.eye(len(component_list), dtype=bool)).stack().mean()
        )
        rows.append(
            {
                "group_id": group_id,
                "channel_count": len(component_list),
                "avg_abs_correlation": float(avg_abs_corr),
                "sensor_types": ", ".join(
                    sorted(
                        {
                            sensor_lookup.get(channel, "")
                            for channel in component_list
                            if sensor_lookup.get(channel, "")
                        }
                    )
                ),
                "channels": ", ".join(component_list),
            }
        )
        group_id += 1
    if not rows:
        return pd.DataFrame(
            columns=[
                "group_id",
                "channel_count",
                "avg_abs_correlation",
                "sensor_types",
                "channels",
            ]
        )
    return pd.DataFrame(rows)


def _merge_events(events: pd.DataFrame, merge_gap_seconds: int) -> list[pd.DataFrame]:
    merged = []
    current = []
    current_end = None
    for _, row in events.sort_values("start").iterrows():
        if current_end is None or row["start"] <= current_end + pd.Timedelta(seconds=merge_gap_seconds):
            current.append(row)
            current_end = max(current_end, row["end"]) if current_end is not None else row["end"]
        else:
            merged.append(pd.DataFrame(current))
            current = [row]
            current_end = row["end"]
    if current:
        merged.append(pd.DataFrame(current))
    return merged


def _classify_event(
    support_count: int,
    grouped_support: int,
    peak_ratio: float,
    duration_s: float,
    impact_ratio: float,
    group_min_channels: int,
) -> tuple[str, str, str]:
    if (
        support_count >= group_min_channels
        and grouped_support >= group_min_channels
        and peak_ratio >= impact_ratio
        and duration_s <= 60
    ):
        return (
            "Boat collision / impact candidate",
            "urgent review",
            "High simultaneous response across at least three correlated channels.",
        )
    if duration_s >= 90 and grouped_support >= group_min_channels:
        return (
            "Drawbridge operation-like event",
            "classify before alerting",
            "Sustained coordinated response across a correlated group.",
        )
    if support_count >= group_min_channels:
        return (
            "Group-supported traffic/vibration event",
            "review",
            "Short burst supported by multiple channels.",
        )
    return (
        "Single/few-channel traffic-like event",
        "low",
        "Short burst without three-channel group confirmation.",
    )


def _channel_to_group(groups: pd.DataFrame) -> dict[str, int]:
    if groups.empty:
        return {}
    mapping = {}
    for row in groups.to_dict("records"):
        for channel in str(row["channels"]).split(", "):
            if channel:
                mapping[channel] = int(row["group_id"])
    return mapping


def _robust_z(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    median = np.nanmedian(array)
    mad = np.nanmedian(np.abs(array - median))
    scale = 1.4826 * mad if mad else np.nanstd(array)
    if not scale or not np.isfinite(scale):
        return np.zeros_like(array, dtype=float)
    return (array - median) / scale


def _empty_event_families() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "event_family",
            "priority",
            "start",
            "end",
            "duration_s",
            "supporting_channels",
            "same_group_channels",
            "groups",
            "peak_ratio",
            "channels",
            "rationale",
        ]
    )


def _empty_shift_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "event_family",
            "group_id",
            "metric",
            "direction",
            "supporting_channels",
            "max_abs_z",
            "channels",
            "reportable",
            "rationale",
        ]
    )
