"""Posted NH Memorial Bridge lift schedule: opens every 30 min, 7am-7pm, May 15-Oct 31; on signal otherwise."""

from __future__ import annotations

import pandas as pd

BRIDGE_TIME_ZONE = "America/New_York"

SEASON_START = (5, 15)
SEASON_END = (10, 31)
DAILY_OPEN_HOUR = 7
DAILY_CLOSE_HOUR = 19
SCHEDULED_INTERVAL_MINUTES = 30
DEFAULT_TOLERANCE_MINUTES = 5.0


def to_bridge_local(timestamps: pd.Series) -> pd.Series:
    # Ingest pipeline timestamps are naive and represent UTC.
    parsed = pd.to_datetime(timestamps, errors="coerce")
    if parsed.empty:
        return parsed
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize("UTC")
    else:
        parsed = parsed.dt.tz_convert("UTC")
    return parsed.dt.tz_convert(BRIDGE_TIME_ZONE).dt.tz_localize(None)


def classify_operation_timing(
    bridge_local_timestamps: pd.Series,
    tolerance_minutes: float = DEFAULT_TOLERANCE_MINUTES,
) -> pd.Series:
    # Per timestamp: "scheduled" (in-season, in-hours, on the :00/:30 grid),
    # "unexpected" (in-season, in-hours, but off the grid), or "on_demand" (otherwise).
    if bridge_local_timestamps.empty:
        return pd.Series(dtype=object)

    month_day = list(zip(bridge_local_timestamps.dt.month, bridge_local_timestamps.dt.day))
    in_season = pd.Series(
        [SEASON_START <= md <= SEASON_END for md in month_day],
        index=bridge_local_timestamps.index,
    )

    minutes = (
        bridge_local_timestamps.dt.hour * 60
        + bridge_local_timestamps.dt.minute
        + bridge_local_timestamps.dt.second / 60
    )
    in_daytime_window = minutes.between(
        DAILY_OPEN_HOUR * 60 - tolerance_minutes,
        DAILY_CLOSE_HOUR * 60 + tolerance_minutes,
    )

    remainder = minutes % SCHEDULED_INTERVAL_MINUTES
    distance_to_slot = pd.concat(
        [remainder, SCHEDULED_INTERVAL_MINUTES - remainder], axis=1
    ).min(axis=1)
    near_slot = distance_to_slot <= tolerance_minutes

    scheduled_window = in_season & in_daytime_window
    result = pd.Series("on_demand", index=bridge_local_timestamps.index)
    result[scheduled_window & near_slot] = "scheduled"
    result[scheduled_window & ~near_slot] = "unexpected"
    return result
