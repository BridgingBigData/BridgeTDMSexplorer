from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tdms_bridge.locations import enrich_sensor_catalog
from tdms_bridge.parser import detect_events, feature_windows


ROSETTE_ORIENTATIONS = {"H", "V", "D"}


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
    group_lookup = _group_lookup(groups)
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
        strongest_group_id = _strongest_group_id(
            support_channels, channel_to_group, group_ids
        )
        strongest_group = group_lookup.get(strongest_group_id, {})
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
            group_kind=str(strongest_group.get("group_kind", "")),
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
                "strongest_group_kind": strongest_group.get("group_kind", ""),
                "strongest_group_label": strongest_group.get("group_label", ""),
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
    group_lookup = _group_lookup(groups)
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
        group = group_lookup.get(int(group_id), {})
        group_kind = str(group.get("group_kind", ""))
        group_label = str(group.get("group_label", ""))
        direction = "up" if frame["z_score"].median() > 0 else "down"
        if group_kind == "rosette":
            family = (
                "Rosette-confirmed operation-like shift"
                if metric == "mean"
                else "Rosette-confirmed behavior shift"
            )
        else:
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
                "group_kind": group_kind,
                "group_label": group_label,
                "metric": metric,
                "direction": direction,
                "supporting_channels": len(support_channels),
                "max_abs_z": float(frame["z_score"].abs().max()),
                "channels": ", ".join(support_channels),
                "reportable": True,
                "rationale": (
                    _shift_rationale(
                        len(support_channels), group_kind, group_label, z_threshold, metric
                    )
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
    catalog = enrich_sensor_catalog(catalog)
    domain_groups = _rosette_domain_groups(catalog, matrix)
    rosette_channels = set()
    for row in domain_groups.to_dict("records"):
        rosette_channels.update(str(row["channels"]).split(", "))

    channels = [channel for channel in matrix.columns if channel not in rosette_channels]
    neighbors = {channel: set() for channel in channels}
    for left in channels:
        for right in channels:
            if left == right:
                continue
            if _is_incompatible_correlation_pair(catalog, left, right):
                continue
            if abs(float(matrix.loc[left, right])) >= min_abs_corr:
                neighbors[left].add(right)

    rows = domain_groups.to_dict("records")
    seen = set()
    group_id = len(rows) + 1
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
                "group_kind": "correlation",
                "group_label": f"Correlation group {group_id}",
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
                "group_kind",
                "group_label",
            ]
        )
    return pd.DataFrame(rows)


def _rosette_domain_groups(catalog: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty or "sensor_family" not in catalog:
        return pd.DataFrame()
    available = set(matrix.columns)
    rosettes = catalog[
        catalog["channel"].isin(available)
        & catalog["sensor_family"].eq("Rosette strain gage")
        & catalog["orientation_code"].isin(ROSETTE_ORIENTATIONS)
    ].copy()
    if rosettes.empty:
        return pd.DataFrame()

    rows = []
    group_id = 1
    for (location, side_code, designation), frame in rosettes.groupby(
        ["longitudinal_location", "side_code", "sensor_designation"], dropna=False
    ):
        by_orientation = {
            str(row["orientation_code"]): row["channel"]
            for _, row in frame.iterrows()
        }
        if not ROSETTE_ORIENTATIONS.issubset(by_orientation):
            continue
        channels = [by_orientation[orientation] for orientation in ["H", "V", "D"]]
        first = frame.iloc[0]
        location_label = str(int(location)) if pd.notna(location) else str(location)
        side_label = str(first.get("side_of_bridge") or side_code)
        submatrix = matrix.loc[channels, channels].abs()
        avg_abs_corr = submatrix.where(~np.eye(len(channels), dtype=bool)).stack().mean()
        rows.append(
            {
                "group_id": group_id,
                "channel_count": len(channels),
                "avg_abs_correlation": avg_abs_corr,
                "sensor_types": ", ".join(sorted(frame["sensor_type"].dropna().unique())),
                "channels": ", ".join(channels),
                "group_kind": "rosette",
                "group_label": f"Location {location_label} {side_label} Rosette {designation}",
            }
        )
        group_id += 1
    return pd.DataFrame(rows)


def _is_incompatible_correlation_pair(
    catalog: pd.DataFrame, left: str, right: str
) -> bool:
    family_lookup = catalog.set_index("channel")["sensor_family"].to_dict()
    left_family = str(family_lookup.get(left, ""))
    right_family = str(family_lookup.get(right, ""))
    left_guide_post = "guide post" in left_family.lower() or left.startswith(("VGP", "VGT"))
    right_guide_post = "guide post" in right_family.lower() or right.startswith(("VGP", "VGT"))
    left_rosette = left_family == "Rosette strain gage"
    right_rosette = right_family == "Rosette strain gage"
    return (left_guide_post and right_rosette) or (right_guide_post and left_rosette)


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
    group_kind: str = "",
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
            (
                "High simultaneous response across all three rosette orientations."
                if group_kind == "rosette"
                else "High simultaneous response across at least three correlated channels."
            ),
        )
    if duration_s >= 90 and grouped_support >= group_min_channels:
        return (
            "Drawbridge operation-like event",
            "classify before alerting",
            (
                "Sustained coordinated response across all three rosette orientations."
                if group_kind == "rosette"
                else "Sustained coordinated response across a correlated group."
            ),
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


def _group_lookup(groups: pd.DataFrame) -> dict[int, dict]:
    if groups.empty:
        return {}
    return {int(row["group_id"]): row for row in groups.to_dict("records")}


def _strongest_group_id(
    support_channels: list[str],
    channel_to_group: dict[str, int],
    group_ids: list[int],
) -> int | None:
    if not group_ids:
        return None
    return max(
        group_ids,
        key=lambda group_id: len(
            [
                channel
                for channel in support_channels
                if channel_to_group.get(channel) == group_id
            ]
        ),
    )


def _shift_rationale(
    support_count: int,
    group_kind: str,
    group_label: str,
    z_threshold: float,
    metric: str,
) -> str:
    if group_kind == "rosette":
        return (
            f"All {support_count} rosette orientations in {group_label} exceeded "
            f"robust z-score {z_threshold:g} on {metric}."
        )
    return (
        f"{support_count} correlated channels exceeded robust "
        f"z-score {z_threshold:g} on {metric}."
    )


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
            "strongest_group_kind",
            "strongest_group_label",
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
            "group_kind",
            "group_label",
            "metric",
            "direction",
            "supporting_channels",
            "max_abs_z",
            "channels",
            "reportable",
            "rationale",
        ]
    )
