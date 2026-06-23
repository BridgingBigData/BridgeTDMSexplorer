from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TYPE_SIZES = {
    1: 1,
    2: 2,
    3: 4,
    4: 8,
    5: 1,
    6: 2,
    7: 4,
    8: 8,
    9: 4,
    10: 8,
    0x21: 1,
    0x44: 16,
}

TYPE_NAMES = {
    1: "int8",
    2: "int16",
    3: "int32",
    4: "int64",
    5: "uint8",
    6: "uint16",
    7: "uint32",
    8: "uint64",
    9: "float32",
    10: "float64",
    0x20: "string",
    0x21: "boolean",
    0x44: "timestamp",
}


@dataclass(frozen=True)
class ParsedTDMS:
    path: Path
    sha256: str
    file_properties: dict[str, Any]
    group_properties: dict[str, Any]
    sensor_catalog: pd.DataFrame
    samples: pd.DataFrame


def recording_start_from_name(path: Path) -> datetime | None:
    return _recording_start_from_name(path)


def discover_tdms_files(folder: Path) -> list[Path]:
    folder = Path(folder)
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == ".tdms"
    )


def discover_analysis_files(folder: Path) -> pd.DataFrame:
    rows = []
    for path in discover_tdms_files(folder):
        ignored_reasons = []
        if "_version_" in path.name:
            ignored_reasons.append("version copy")
        if "Decimate" in path.name:
            ignored_reasons.append("decimate file")
        timestamp = recording_start_from_name(path)
        rows.append(
            {
                "file": path.name,
                "path": str(path),
                "timestamp": timestamp,
                "size_mb": path.stat().st_size / (1024 * 1024),
                "included": not ignored_reasons and timestamp is not None,
                "ignored_reason": ", ".join(ignored_reasons),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["included", "timestamp", "file"], ascending=[False, True, True])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_tdms_files(folder: Path) -> tuple[list[Path], pd.DataFrame]:
    rows = []
    seen: dict[str, Path] = {}
    unique: list[Path] = []
    for path in discover_tdms_files(folder):
        sha = file_sha256(path)
        duplicate_of = seen.get(sha)
        if duplicate_of is None:
            seen[sha] = path
            unique.append(path)
        rows.append(
            {
                "file": path.name,
                "sha256": sha,
                "size_mb": path.stat().st_size / (1024 * 1024),
                "duplicate_of": duplicate_of.name if duplicate_of else "",
            }
        )
    return unique, pd.DataFrame(rows)


def parse_tdms(path: Path) -> ParsedTDMS:
    data = path.read_bytes()
    objects: dict[str, dict[str, Any]] = {}
    data_segments: list[tuple[int, list[tuple[str, int, int]], int, int]] = []

    offset = 0
    while offset < len(data):
        if data[offset : offset + 4] != b"TDSm":
            raise ValueError(f"{path.name}: invalid TDMS tag at byte {offset}")

        next_offset = struct.unpack_from("<Q", data, offset + 12)[0]
        raw_offset = struct.unpack_from("<Q", data, offset + 20)[0]
        metadata_start = offset + 28
        raw_start = metadata_start + raw_offset
        next_start = (
            metadata_start + next_offset
            if next_offset != 0xFFFFFFFFFFFFFFFF
            else len(data)
        )

        metadata_offset = metadata_start
        object_count, metadata_offset = _read_u32(data, metadata_offset)
        segment_channels: list[tuple[str, int, int]] = []

        for _ in range(object_count):
            object_path, metadata_offset = _read_string(data, metadata_offset)
            raw_index, metadata_offset = _read_u32(data, metadata_offset)
            object_info = objects.setdefault(
                object_path, {"properties": {}, "raw": None}
            )

            raw_info = None
            if raw_index == 0xFFFFFFFF:
                raw_info = None
            elif raw_index == 0:
                raw_info = object_info["raw"]
            else:
                dtype, metadata_offset = _read_u32(data, metadata_offset)
                dimension, metadata_offset = _read_u32(data, metadata_offset)
                value_count, metadata_offset = _read_u64(data, metadata_offset)
                raw_info = {
                    "dtype": dtype,
                    "dimension": dimension,
                    "value_count": value_count,
                }
                if dtype == 0x20:
                    total_bytes, metadata_offset = _read_u64(data, metadata_offset)
                    raw_info["total_string_bytes"] = total_bytes
                object_info["raw"] = raw_info

            property_count, metadata_offset = _read_u32(data, metadata_offset)
            for _ in range(property_count):
                key, metadata_offset = _read_string(data, metadata_offset)
                value_type, metadata_offset = _read_u32(data, metadata_offset)
                value, metadata_offset = _read_property_value(
                    data, metadata_offset, value_type
                )
                object_info["properties"][key] = value

            if raw_info and raw_info.get("dtype") in TYPE_SIZES:
                segment_channels.append(
                    (
                        object_path,
                        int(raw_info["dtype"]),
                        int(raw_info["value_count"]),
                    )
                )

        chunk_size = sum(
            value_count * TYPE_SIZES[dtype]
            for _, dtype, value_count in segment_channels
        )
        raw_bytes = next_start - raw_start
        if segment_channels and chunk_size:
            repeat_count = raw_bytes // chunk_size
            if raw_bytes % chunk_size:
                raise ValueError(
                    f"{path.name}: raw segment at byte {offset} has uneven chunks"
                )
            data_segments.append((raw_start, segment_channels, repeat_count, chunk_size))

        offset = next_start

    file_properties = objects.get("/", {}).get("properties", {})
    group_path = _primary_group_path(objects)
    group_properties = objects.get(group_path, {}).get("properties", {})
    channel_data = _read_channel_arrays(data, data_segments)
    catalog = _build_catalog(objects, channel_data)
    samples = _build_samples(path, channel_data)
    return ParsedTDMS(
        path=path,
        sha256=file_sha256(path),
        file_properties=file_properties,
        group_properties=group_properties,
        sensor_catalog=catalog,
        samples=samples,
    )


def channel_summary(parsed: ParsedTDMS) -> pd.DataFrame:
    rows = []
    for channel in _sample_channels(parsed.samples):
        values = parsed.samples[channel]
        rows.append(
            {
                "channel": channel,
                "samples": int(values.count()),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "rms": float(np.sqrt(np.mean(np.square(values.to_numpy())))),
                "peak_to_peak": float(values.max() - values.min()),
            }
        )
    summary = pd.DataFrame(rows)
    catalog = parsed.sensor_catalog.drop(columns=["samples"], errors="ignore")
    return summary.merge(catalog, on="channel", how="left")


def feature_windows(parsed: ParsedTDMS, window: str = "1min") -> pd.DataFrame:
    samples = parsed.samples.set_index("timestamp")
    channels = [col for col in _sample_channels(parsed.samples) if col != "Time"]
    if not channels:
        return pd.DataFrame()

    values = samples[channels]
    grouped = values.resample(window)
    minimum = grouped.min()
    maximum = grouped.max()
    metrics = {
        "mean": grouped.mean(),
        "std": grouped.std(ddof=0),
        "min": minimum,
        "max": maximum,
        "rms": values.pow(2).resample(window).mean().pow(0.5),
        "peak_to_peak": maximum - minimum,
    }

    long_metrics = []
    for metric_name, metric_frame in metrics.items():
        metric_frame.columns.name = "channel"
        long_metrics.append(
            metric_frame.stack(future_stack=True).rename(metric_name).reset_index()
        )

    features = long_metrics[0]
    for metric_frame in long_metrics[1:]:
        features = features.merge(metric_frame, on=["timestamp", "channel"], how="outer")
    return features.dropna(
        subset=["mean", "std", "min", "max", "rms", "peak_to_peak"], how="all"
    ).reset_index(drop=True)


def detect_events(
    parsed: ParsedTDMS,
    channels: list[str],
    window_seconds: int = 5,
    threshold_sigma: float = 4.0,
) -> pd.DataFrame:
    samples = parsed.samples.set_index("timestamp")
    rows = []
    for channel in channels:
        if channel not in samples:
            continue
        series = samples[channel].dropna()
        if series.empty:
            continue
        centered = series - series.rolling("2min", min_periods=10).median()
        rms = (
            centered.pow(2)
            .rolling(f"{window_seconds}s", min_periods=max(2, window_seconds * 2))
            .mean()
            .pow(0.5)
        )
        baseline = rms.median()
        spread = _mad(rms.dropna())
        threshold = baseline + threshold_sigma * spread
        active = rms > threshold
        if not active.any():
            continue
        event_ids = (active.ne(active.shift(fill_value=False))).cumsum()
        for _, event_points in rms[active].groupby(event_ids[active]):
            start = event_points.index.min()
            end = event_points.index.max()
            if (end - start).total_seconds() < 1:
                continue
            rows.append(
                {
                    "channel": channel,
                    "start": start,
                    "end": end,
                    "duration_s": (end - start).total_seconds(),
                    "peak_rms": float(event_points.max()),
                    "threshold": float(threshold),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["channel", "start", "end", "duration_s", "peak_rms", "threshold"]
        )
    return pd.DataFrame(rows).sort_values(["start", "channel"]).reset_index(drop=True)


def sensor_health(parsed: ParsedTDMS) -> pd.DataFrame:
    summary = channel_summary(parsed)
    rows = []
    for row in summary.to_dict("records"):
        flags = []
        if row["samples"] == 0:
            flags.append("missing")
        if math.isclose(row["std"], 0.0, abs_tol=1e-12):
            flags.append("flatline")
        sensor_type = row.get("sensor_type", "")
        if sensor_type in {"Quarterarm", "Half Bridge I"} and abs(row["mean"]) > 100_000:
            flags.append("extreme bridge value")
        if sensor_type == "Accelerometer" and row["peak_to_peak"] > 1:
            flags.append("large accelerometer swing")
        rows.append(
            {
                "channel": row["channel"],
                "sensor_type": sensor_type,
                "samples": row["samples"],
                "std": row["std"],
                "min": row["min"],
                "max": row["max"],
                "flags": ", ".join(flags) if flags else "ok",
            }
        )

    inactive = parsed.sensor_catalog[~parsed.sensor_catalog["active"]]
    for row in inactive.to_dict("records"):
        rows.append(
            {
                "channel": row["channel"],
                "sensor_type": row.get("sensor_type", ""),
                "samples": 0,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "flags": "configured but inactive",
            }
        )
    return pd.DataFrame(rows)


def cache_parsed_file(path: Path, cache_dir: Path) -> dict[str, Path]:
    parsed = parse_tdms(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{path.stem}_{parsed.sha256[:12]}"
    samples_path = cache_dir / f"{stem}_samples.parquet"
    catalog_path = cache_dir / f"{stem}_catalog.parquet"
    summary_path = cache_dir / f"{stem}_summary.parquet"
    features_path = cache_dir / f"{stem}_features_1min.parquet"
    health_path = cache_dir / f"{stem}_health.parquet"

    parsed.samples.to_parquet(samples_path, index=False)
    parsed.sensor_catalog.to_parquet(catalog_path, index=False)
    channel_summary(parsed).to_parquet(summary_path, index=False)
    feature_windows(parsed).to_parquet(features_path, index=False)
    sensor_health(parsed).to_parquet(health_path, index=False)
    return {
        "samples": samples_path,
        "catalog": catalog_path,
        "summary": summary_path,
        "features": features_path,
        "health": health_path,
    }


def _read_channel_arrays(
    data: bytes, data_segments: list[tuple[int, list[tuple[str, int, int]], int, int]]
) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {}
    for raw_start, channels, repeat_count, _ in data_segments:
        offset = raw_start
        for _ in range(repeat_count):
            for object_path, dtype, value_count in channels:
                if dtype != 10:
                    offset += TYPE_SIZES[dtype] * value_count
                    continue
                values = np.frombuffer(
                    data, dtype="<f8", count=value_count, offset=offset
                ).copy()
                arrays.setdefault(_channel_name(object_path), []).append(values)
                offset += TYPE_SIZES[dtype] * value_count
    return {key: np.concatenate(chunks) for key, chunks in arrays.items()}


def _build_samples(path: Path, channel_data: dict[str, np.ndarray]) -> pd.DataFrame:
    if not channel_data:
        return pd.DataFrame()
    max_len = max(len(values) for values in channel_data.values())
    frame = pd.DataFrame(
        {
            channel: _pad(values, max_len)
            for channel, values in sorted(channel_data.items())
            if channel != "Time"
        }
    )
    if "Time" in channel_data:
        time_values = _pad(channel_data["Time"], max_len)
        elapsed = time_values - time_values[0]
        frame["Time"] = time_values
    else:
        elapsed = np.arange(max_len, dtype=float)
    start = _recording_start_from_name(path) or datetime.fromtimestamp(path.stat().st_mtime)
    frame.insert(0, "timestamp", [start + timedelta(seconds=float(x)) for x in elapsed])
    return frame


def _build_catalog(
    objects: dict[str, dict[str, Any]], channel_data: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    group_path = _primary_group_path(objects)
    prefix = f"{group_path}/" if group_path else ""
    for object_path, info in objects.items():
        if not prefix or not object_path.startswith(prefix):
            continue
        channel = _channel_name(object_path)
        properties = info["properties"]
        raw = info.get("raw") or {}
        rows.append(
            {
                "channel": channel,
                "active": channel in channel_data,
                "sensor_type": properties.get("Type", "Time" if channel == "Time" else ""),
                "sensor_name": properties.get("SensorName", ""),
                "hardware_location": properties.get("Location", ""),
                "gain": properties.get("Gain", ""),
                "offset": properties.get("Offset", ""),
                "cal_factor": properties.get("CalFactor", ""),
                "excitation_mode": properties.get("ExcitationMode", ""),
                "array_column": properties.get("NI_ArrayColumn", np.nan),
                "dtype": TYPE_NAMES.get(raw.get("dtype"), ""),
                "samples": len(channel_data[channel]) if channel in channel_data else 0,
            }
        )
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return catalog
    return catalog.sort_values(["active", "array_column", "channel"], ascending=[False, True, True])


def _recording_start_from_name(path: Path) -> datetime | None:
    match = re.search(
        r"_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})_", path.name
    )
    if not match:
        return None
    month, day, year, hour, minute, second = map(int, match.groups())
    return datetime(year, month, day, hour, minute, second)


def _primary_group_path(objects: dict[str, dict[str, Any]]) -> str:
    groups = [
        path
        for path in objects
        if path.startswith("/'") and path.count("/") == 1
    ]
    if "/'AllNormal'" in groups:
        return "/'AllNormal'"
    if not groups:
        return ""
    groups.sort()
    return groups[0]


def _channel_name(object_path: str) -> str:
    return object_path.rsplit("/", 1)[-1].strip("'")


def _pad(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values
    output = np.full(length, np.nan)
    output[: len(values)] = values
    return output


def _mad(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    median = values.median()
    mad = (values - median).abs().median()
    return float(1.4826 * mad if mad else values.std(ddof=0) or 0.0)


def _sample_channels(samples: pd.DataFrame) -> list[str]:
    ignored = {"timestamp", "source_file"}
    return [
        column
        for column in samples.columns
        if column not in ignored and pd.api.types.is_numeric_dtype(samples[column])
    ]


def _read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _read_u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _read_u32(data, offset)
    return data[offset : offset + length].decode("utf-8", "replace"), offset + length


def _read_property_value(data: bytes, offset: int, value_type: int) -> tuple[Any, int]:
    if value_type == 0x20:
        return _read_string(data, offset)
    if value_type == 10:
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    if value_type == 9:
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    if value_type == 3:
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    if value_type == 7:
        return struct.unpack_from("<I", data, offset)[0], offset + 4
    if value_type == 4:
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    if value_type == 8:
        return struct.unpack_from("<Q", data, offset)[0], offset + 8
    if value_type == 2:
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    if value_type == 6:
        return struct.unpack_from("<H", data, offset)[0], offset + 2
    if value_type == 1:
        return struct.unpack_from("<b", data, offset)[0], offset + 1
    if value_type == 5:
        return struct.unpack_from("<B", data, offset)[0], offset + 1
    if value_type == 0x21:
        return struct.unpack_from("<?", data, offset)[0], offset + 1
    if value_type == 0x44:
        fractions, seconds = struct.unpack_from("<Qq", data, offset)
        # TDMS timestamps count seconds since 1904-01-01 plus 2^-64 fractions.
        base = datetime(1904, 1, 1)
        timestamp = base + timedelta(seconds=seconds + fractions / 2**64)
        return timestamp.isoformat(), offset + 16
    raise ValueError(f"Unsupported TDMS property type: {value_type}")
