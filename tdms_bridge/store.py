from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pandas as pd

from tdms_bridge.parser import (
    channel_summary,
    detect_events,
    discover_analysis_files,
    feature_windows,
    parse_tdms,
    sensor_health,
)


TREND_WINDOWS = ("30s", "1min", "5min", "15min", "1h")
DEFAULT_EVENT_WINDOW_SECONDS = 5
DEFAULT_EVENT_THRESHOLD_SIGMA = 4.0


@dataclass
class IngestionReport:
    scanned: int = 0
    included: int = 0
    ignored: int = 0
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    rows: int = 0
    messages: list[str] = field(default_factory=list)


ProgressCallback = Callable[[str, int, int, str], None]


def ingest_folder(
    folder: Path, cache_dir: Path, progress_callback: ProgressCallback | None = None
) -> IngestionReport:
    duckdb = _duckdb()
    folder = Path(folder).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _init_schema(cache_dir)

    report = IngestionReport()
    _progress(progress_callback, "Scanning files", 0, 1, "Scanning TDMS folder...")
    file_table = discover_analysis_files(folder)
    report.scanned = len(file_table)
    if file_table.empty:
        _progress(progress_callback, "Ready", 1, 1, "No TDMS files found.")
        return report

    included = file_table[file_table["included"]].copy()
    ignored = file_table[~file_table["included"]].copy()
    report.included = len(included)
    report.ignored = len(ignored)
    _record_ignored_files(cache_dir, ignored)

    total = len(included)
    for index, row in enumerate(included.to_dict("records"), start=1):
        path = Path(row["path"])
        _progress(
            progress_callback,
            "Ingesting new files",
            index - 1,
            max(total, 1),
            f"Checking {path.name}...",
        )
        try:
            status = _file_status(cache_dir, path)
            if status == "ready":
                report.skipped += 1
                continue

            _cleanup_artifacts(cache_dir, str(path))
            parsed = parse_tdms(path)
            sample_frame = parsed.samples.copy()
            sample_frame["source_file"] = parsed.path.name
            sample_frame["source_path"] = str(parsed.path)
            sample_frame["date"] = sample_frame["timestamp"].dt.strftime("%Y-%m-%d")

            sample_path = _artifact_path(
                cache_dir,
                "samples",
                parsed.path,
                parsed.sha256,
                sample_frame["date"].iloc[0],
            )
            _write_parquet(sample_frame, sample_path)
            artifacts = [("samples", sample_path)]

            for window in TREND_WINDOWS:
                feature_frame = feature_windows(parsed, window)
                if feature_frame.empty:
                    continue
                feature_frame["source_file"] = parsed.path.name
                feature_frame["source_path"] = str(parsed.path)
                feature_frame["date"] = feature_frame["timestamp"].dt.strftime("%Y-%m-%d")
                feature_path = _artifact_path(
                    cache_dir,
                    f"features_{window}",
                    parsed.path,
                    parsed.sha256,
                    feature_frame["date"].iloc[0],
                    window=window,
                )
                _write_parquet(feature_frame, feature_path)
                artifacts.append((f"features_{window}", feature_path))

            catalog = parsed.sensor_catalog.copy()
            catalog["source_file"] = parsed.path.name
            catalog["source_path"] = str(parsed.path)
            catalog_path = _metadata_artifact_path(cache_dir, "catalog", parsed.path, parsed.sha256)
            _write_parquet(catalog, catalog_path)
            artifacts.append(("catalog", catalog_path))

            summary = channel_summary(parsed)
            summary["source_file"] = parsed.path.name
            summary["source_path"] = str(parsed.path)
            summary_path = _metadata_artifact_path(cache_dir, "summary", parsed.path, parsed.sha256)
            _write_parquet(summary, summary_path)
            artifacts.append(("summary", summary_path))

            health = sensor_health(parsed)
            health["source_file"] = parsed.path.name
            health["source_path"] = str(parsed.path)
            health_path = _metadata_artifact_path(cache_dir, "health", parsed.path, parsed.sha256)
            _write_parquet(health, health_path)
            artifacts.append(("health", health_path))

            traffic_channels = catalog[
                catalog["sensor_type"].isin(["Accelerometer", "Quarterarm", "Half Bridge I"])
                & catalog["active"]
            ]["channel"].tolist()
            events = detect_events(
                parsed,
                traffic_channels,
                DEFAULT_EVENT_WINDOW_SECONDS,
                DEFAULT_EVENT_THRESHOLD_SIGMA,
            )
            if not events.empty:
                events["source_file"] = parsed.path.name
                events["source_path"] = str(parsed.path)
                events["date"] = pd.to_datetime(events["start"]).dt.strftime("%Y-%m-%d")
            events_path = _metadata_artifact_path(cache_dir, "events_default", parsed.path, parsed.sha256)
            _write_parquet(events, events_path)
            artifacts.append(("events_default", events_path))

            _upsert_ready_file(cache_dir, path, row, parsed, artifacts)
            report.ingested += 1
            report.rows += len(parsed.samples)
        except Exception as exc:
            report.failed += 1
            report.messages.append(f"{path.name}: {exc}")
            _record_failed_file(cache_dir, path, row, exc)

    _progress(progress_callback, "Updating index", total, max(total, 1), "Updating DuckDB index...")
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        con.execute("CHECKPOINT")
    _progress(
        progress_callback,
        "Ready",
        max(total, 1),
        max(total, 1),
        f"Ready: {report.ingested} ingested, {report.skipped} skipped, {report.failed} failed.",
    )
    return report


def get_available_range(cache_dir: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    duckdb = _duckdb()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)
        row = con.execute(
            """
            select min(sample_start) as start_time, max(sample_end) as end_time
            from file_index
            where included and status = 'ready'
            """
        ).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None, None
    return pd.Timestamp(row[0]), pd.Timestamp(row[1])


def query_file_index(cache_dir: Path, start, end) -> pd.DataFrame:
    duckdb = _duckdb()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)
        return con.execute(
            """
            select
                file,
                path,
                timestamp,
                size_mb,
                included,
                ignored_reason,
                status,
                error,
                sample_start,
                sample_end,
                sample_rows,
                sha256
            from file_index
            where (
                sample_start is not null
                and sample_end >= ?
                and sample_start <= ?
            )
            or (
                sample_start is null
                and timestamp between ? and ?
            )
            order by timestamp, file
            """,
            [
                pd.Timestamp(start),
                pd.Timestamp(end),
                pd.Timestamp(start),
                pd.Timestamp(end),
            ],
        ).fetchdf()


def query_ignored_files(cache_dir: Path) -> pd.DataFrame:
    duckdb = _duckdb()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)
        return con.execute(
            """
            select file, path, timestamp, size_mb, included, ignored_reason, status, error
            from file_index
            where not included
            order by file
            """
        ).fetchdf()


def query_catalog(cache_dir: Path, start, end) -> pd.DataFrame:
    return _read_metadata_for_range(cache_dir, start, end, "catalog")


def query_summary(cache_dir: Path, start, end) -> pd.DataFrame:
    return _read_metadata_for_range(cache_dir, start, end, "summary")


def query_health(cache_dir: Path, start, end) -> pd.DataFrame:
    return _read_metadata_for_range(cache_dir, start, end, "health")


def query_samples(
    cache_dir: Path,
    start,
    end,
    channels: list[str],
    max_points: int | None = None,
) -> pd.DataFrame:
    duckdb = _duckdb()
    sample_glob = _glob(cache_dir, "samples")
    if not list((Path(cache_dir) / "samples").glob("date=*/*.parquet")):
        return pd.DataFrame(columns=["timestamp", "source_file", *channels])
    columns = ["timestamp", "source_file", *channels]
    select_columns = ", ".join(_quote_identifier(column) for column in columns)
    where = "timestamp between ? and ?"
    params = [pd.Timestamp(start), pd.Timestamp(end)]
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        if max_points:
            count = con.execute(
                f"select count(*) from read_parquet('{sample_glob}', union_by_name=true) where {where}",
                params,
            ).fetchone()[0]
            step = max(1, int(count) // int(max_points))
            return con.execute(
                f"""
                with rows as (
                    select {select_columns},
                           row_number() over (order by timestamp) as rn
                    from read_parquet('{sample_glob}', union_by_name=true)
                    where {where}
                )
                select {select_columns}
                from rows
                where rn % ? = 1
                order by timestamp
                """,
                [*params, step],
            ).fetchdf()
        return con.execute(
            f"""
            select {select_columns}
            from read_parquet('{sample_glob}', union_by_name=true)
            where {where}
            order by timestamp
            """,
            params,
        ).fetchdf()


def query_features(
    cache_dir: Path,
    start,
    end,
    window: str,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    duckdb = _duckdb()
    feature_root = Path(cache_dir) / "features" / f"window={window}"
    if not feature_root.exists():
        return pd.DataFrame()
    feature_glob = str(feature_root / "date=*" / "*.parquet")
    params: list = [pd.Timestamp(start), pd.Timestamp(end)]
    channel_clause = ""
    if channels:
        placeholders = ", ".join(["?"] * len(channels))
        channel_clause = f" and channel in ({placeholders})"
        params.extend(channels)
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        return con.execute(
            f"""
            select *
            from read_parquet('{feature_glob}', union_by_name=true)
            where timestamp between ? and ? {channel_clause}
            order by timestamp, channel
            """,
            params,
        ).fetchdf()


def query_events(
    cache_dir: Path,
    start,
    end,
    channels: list[str],
    window_seconds: int,
    threshold_sigma: float,
) -> pd.DataFrame:
    if (
        int(window_seconds) == DEFAULT_EVENT_WINDOW_SECONDS
        and float(threshold_sigma) == DEFAULT_EVENT_THRESHOLD_SIGMA
    ):
        events = _read_metadata_for_range(cache_dir, start, end, "events_default")
        if events.empty:
            return events
        events = events[
            (pd.to_datetime(events["start"]) >= pd.Timestamp(start))
            & (pd.to_datetime(events["start"]) <= pd.Timestamp(end))
        ]
        return events[events["channel"].isin(channels)].reset_index(drop=True)

    samples = query_samples(cache_dir, start, end, channels)
    if samples.empty:
        return pd.DataFrame(
            columns=["channel", "start", "end", "duration_s", "peak_rms", "threshold"]
        )
    parsed = SimpleNamespace(samples=samples, sensor_catalog=pd.DataFrame())
    return detect_events(parsed, channels, int(window_seconds), float(threshold_sigma))


def selected_range_signature(selected_files: pd.DataFrame, start, end) -> tuple:
    if selected_files.empty:
        return (pd.Timestamp(start).isoformat(), pd.Timestamp(end).isoformat())
    return tuple(
        [pd.Timestamp(start).isoformat(), pd.Timestamp(end).isoformat()]
        + [
            f"{row.path}:{row.sha256}:{row.sample_rows}"
            for row in selected_files.itertuples()
        ]
    )


def _duckdb():
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DuckDB is required for the scalable TDMS cache. Install it with "
            "`python3 -m pip install duckdb` and restart the app."
        ) from exc
    return duckdb


def _init_schema(cache_dir: Path) -> None:
    duckdb = _duckdb()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)


def _ensure_schema(con) -> None:
    con.execute(
        """
        create table if not exists file_index (
            path varchar primary key,
            file varchar,
            timestamp timestamp,
            size_bytes bigint,
            size_mb double,
            mtime_ns bigint,
            sha256 varchar,
            included boolean,
            ignored_reason varchar,
            status varchar,
            error varchar,
            sample_start timestamp,
            sample_end timestamp,
            sample_rows bigint,
            ingested_at timestamp
        )
        """
    )
    con.execute(
        """
        create table if not exists artifacts (
            path varchar,
            kind varchar,
            artifact_path varchar,
            primary key(path, kind)
        )
        """
    )


def _db_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "bridge_index.duckdb"


def _record_ignored_files(cache_dir: Path, ignored: pd.DataFrame) -> None:
    if ignored.empty:
        return
    duckdb = _duckdb()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)
        for row in ignored.to_dict("records"):
            path = Path(row["path"])
            stat = path.stat()
            con.execute(
                """
                insert or replace into file_index
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(path),
                    path.name,
                    row.get("timestamp"),
                    stat.st_size,
                    stat.st_size / (1024 * 1024),
                    stat.st_mtime_ns,
                    "",
                    False,
                    row.get("ignored_reason", ""),
                    "ignored",
                    "",
                    None,
                    None,
                    0,
                    datetime.now(),
                ],
            )


def _file_status(cache_dir: Path, path: Path) -> str:
    duckdb = _duckdb()
    stat = path.stat()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)
        row = con.execute(
            """
            select status, size_bytes, mtime_ns
            from file_index
            where path = ?
            """,
            [str(path)],
        ).fetchone()
        if not row:
            return "new"
        status, size_bytes, mtime_ns = row
        if status == "ready" and int(size_bytes) == stat.st_size and int(mtime_ns) == stat.st_mtime_ns:
            artifact_count = con.execute(
                "select count(*) from artifacts where path = ?", [str(path)]
            ).fetchone()[0]
            return "ready" if artifact_count else "changed"
        return "changed"


def _cleanup_artifacts(cache_dir: Path, path: str) -> None:
    duckdb = _duckdb()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)
        artifacts = con.execute(
            "select artifact_path from artifacts where path = ?", [path]
        ).fetchall()
        for (artifact_path,) in artifacts:
            try:
                Path(artifact_path).unlink(missing_ok=True)
            except OSError:
                pass
        con.execute("delete from artifacts where path = ?", [path])


def _upsert_ready_file(
    cache_dir: Path,
    path: Path,
    file_row: dict,
    parsed,
    artifacts: list[tuple[str, Path]],
) -> None:
    duckdb = _duckdb()
    stat = path.stat()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)
        con.execute(
            """
            insert or replace into file_index
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(path),
                path.name,
                file_row.get("timestamp"),
                stat.st_size,
                stat.st_size / (1024 * 1024),
                stat.st_mtime_ns,
                parsed.sha256,
                True,
                "",
                "ready",
                "",
                parsed.samples["timestamp"].min().to_pydatetime(),
                parsed.samples["timestamp"].max().to_pydatetime(),
                len(parsed.samples),
                datetime.now(),
            ],
        )
        for kind, artifact_path in artifacts:
            con.execute(
                "insert or replace into artifacts values (?, ?, ?)",
                [str(path), kind, str(artifact_path)],
            )


def _record_failed_file(cache_dir: Path, path: Path, file_row: dict, exc: Exception) -> None:
    duckdb = _duckdb()
    stat = path.stat()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)
        con.execute(
            """
            insert or replace into file_index
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(path),
                path.name,
                file_row.get("timestamp"),
                stat.st_size,
                stat.st_size / (1024 * 1024),
                stat.st_mtime_ns,
                "",
                True,
                "",
                "failed",
                str(exc),
                None,
                None,
                0,
                datetime.now(),
            ],
        )


def _read_metadata_for_range(cache_dir: Path, start, end, kind: str) -> pd.DataFrame:
    duckdb = _duckdb()
    with duckdb.connect(str(_db_path(cache_dir))) as con:
        _ensure_schema(con)
        artifacts = con.execute(
            """
            select a.artifact_path
            from artifacts a
            join file_index f on f.path = a.path
            where a.kind = ?
              and f.included
              and f.status = 'ready'
              and f.sample_start is not null
              and f.sample_end >= ?
              and f.sample_start <= ?
            order by f.timestamp
            """,
            [kind, pd.Timestamp(start), pd.Timestamp(end)],
        ).fetchall()
    paths = [Path(row[0]) for row in artifacts if Path(row[0]).exists()]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _artifact_path(
    cache_dir: Path,
    kind: str,
    path: Path,
    sha256: str,
    date: str,
    window: str | None = None,
) -> Path:
    safe = _safe_stem(path)
    if kind == "samples":
        return cache_dir / "samples" / f"date={date}" / f"{safe}_{sha256[:12]}.parquet"
    if kind.startswith("features_"):
        feature_window = window or kind.removeprefix("features_")
        return (
            cache_dir
            / "features"
            / f"window={feature_window}"
            / f"date={date}"
            / f"{safe}_{sha256[:12]}.parquet"
        )
    return _metadata_artifact_path(cache_dir, kind, path, sha256)


def _metadata_artifact_path(cache_dir: Path, kind: str, path: Path, sha256: str) -> Path:
    return cache_dir / "metadata" / kind / f"{_safe_stem(path)}_{sha256[:12]}.parquet"


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def _glob(cache_dir: Path, kind: str) -> str:
    if kind == "samples":
        return str(Path(cache_dir) / "samples" / "date=*" / "*.parquet")
    raise ValueError(kind)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _progress(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback:
        callback(stage, current, total, message)
