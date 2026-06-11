from __future__ import annotations

import argparse
from pathlib import Path

from tdms_bridge.parser import (
    cache_parsed_file,
    channel_summary,
    parse_tdms,
    sensor_health,
    unique_tdms_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize bridge TDMS files.")
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing .tdms files")
    parser.add_argument("--cache", action="store_true", help="Write parquet cache files")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    unique_files, file_table = unique_tdms_files(folder)
    print("\nFiles")
    print(file_table.to_string(index=False))

    for path in unique_files:
        parsed = parse_tdms(path)
        print(f"\n{path.name}")
        print(f"sha256: {parsed.sha256}")
        print(f"title: {parsed.file_properties.get('Title', '')}")
        print(f"sample rate: {parsed.group_properties.get('SampleRate[s/s]', '')}")
        print(f"samples shape: {parsed.samples.shape[0]} rows x {parsed.samples.shape[1]} columns")
        print(
            f"time span: {parsed.samples['timestamp'].min()} to {parsed.samples['timestamp'].max()}"
        )

        active = parsed.sensor_catalog[parsed.sensor_catalog["active"]]
        inactive = parsed.sensor_catalog[~parsed.sensor_catalog["active"]]
        print(f"active channels: {len(active)} including Time")
        print(f"configured inactive channels: {len(inactive)}")

        print("\nChannel summary")
        columns = [
            "channel",
            "sensor_type",
            "samples",
            "min",
            "max",
            "mean",
            "std",
            "hardware_location",
        ]
        print(channel_summary(parsed)[columns].to_string(index=False))

        flagged = sensor_health(parsed)
        flagged = flagged[flagged["flags"] != "ok"]
        print("\nHealth flags")
        print(flagged.to_string(index=False) if not flagged.empty else "none")

        if args.cache:
            outputs = cache_parsed_file(path, folder / "cache")
            print("\nCached outputs")
            for kind, output_path in outputs.items():
                print(f"{kind}: {output_path}")


if __name__ == "__main__":
    main()
