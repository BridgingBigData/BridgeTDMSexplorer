from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go


SIDE_LABELS = {"E": "East", "W": "West"}
BRIDGE_LOCATION_LABELS = {
    "T": "Span top chord",
    "B": "Span bottom chord",
    "D": "Span diagonal",
    "TW": "Tower",
}
SENSOR_LOCATION_LABELS = {
    "TF": "Top flange",
    "SF": "North face of south flange",
    "WB": "Web",
}
ORIENTATION_LABELS = {"H": "Horizontal", "D": "Diagonal", "V": "Vertical"}
FACE_LABELS = {"T": "Top face of member", "B": "Bottom face of member"}


def enrich_sensor_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return catalog.copy()
    rows = []
    for row in catalog.to_dict("records"):
        decoded = decode_sensor_location(str(row.get("channel", "")))
        rows.append({**row, **decoded})
    return pd.DataFrame(rows)


def decode_sensor_location(channel: str) -> dict:
    accelerator = re.match(r"^A-(\d+)-([EW])-(TW|[TBD])-([A-Z]+)$", channel)
    if accelerator:
        location, side, bridge_location, sensor_location = accelerator.groups()
        return _base_location(
            channel=channel,
            sensor_family="Accelerometer",
            longitudinal_location=int(location),
            side_code=side,
            bridge_location_code=bridge_location,
            sensor_location_code=sensor_location,
        )

    rosette = re.match(r"^SG-(\d+)-([EW])-R-([A-E])-([HDV])$", channel)
    if rosette:
        location, side, designation, orientation = rosette.groups()
        decoded = _base_location(
            channel=channel,
            sensor_family="Rosette strain gage",
            longitudinal_location=int(location),
            side_code=side,
            bridge_location_code="TW" if int(location) >= 8 else "T",
            sensor_location_code="WB",
        )
        decoded["sensor_designation"] = designation
        decoded["orientation"] = ORIENTATION_LABELS.get(orientation, orientation)
        decoded["orientation_code"] = orientation
        return decoded

    guide_post = re.match(r"^VGP_(\d+)_([TB])$", channel)
    if guide_post:
        member, face = guide_post.groups()
        return {
            "sensor_family": "Vertical guide post half bridge",
            "longitudinal_location": 9,
            "side_of_bridge": "Tower guide post",
            "side_code": "VGP",
            "location_on_bridge": "Vertical guide post",
            "bridge_location_code": "VGP",
            "sensor_location": FACE_LABELS.get(face, face),
            "sensor_location_code": face,
            "sensor_designation": f"Member {member}",
            "orientation": "",
            "orientation_code": "",
            "plan_x": 9.35,
            "plan_y": 0.18 + (int(member) - 4) * 0.055,
            "placement_confidence": "decoded from vertical-guide-post label",
            "placement_source": "UNH_MemorialBridge_InstPlan_V3_FieldInstall.pdf, LTM-11",
        }

    return {
        "sensor_family": "",
        "longitudinal_location": pd.NA,
        "side_of_bridge": "",
        "side_code": "",
        "location_on_bridge": "",
        "bridge_location_code": "",
        "sensor_location": "",
        "sensor_location_code": "",
        "sensor_designation": "",
        "orientation": "",
        "orientation_code": "",
        "plan_x": pd.NA,
        "plan_y": pd.NA,
        "placement_confidence": "unknown",
        "placement_source": "",
    }


def sensor_location_figure(
    catalog: pd.DataFrame,
    highlight_channels: list[str] | None = None,
    title: str = "Sensor Placement",
) -> go.Figure:
    highlight_channels = highlight_channels or []
    sensors = catalog.copy()
    if "plan_x" not in sensors:
        sensors = enrich_sensor_catalog(sensors)
    sensors = sensors.drop_duplicates("channel").dropna(subset=["plan_x", "plan_y"])
    sensors["display_x"] = sensors.apply(_display_x, axis=1)
    sensors["display_y"] = sensors.apply(_display_y, axis=1)
    sensors["highlight"] = sensors["channel"].isin(highlight_channels)
    sensors["marker_size"] = sensors["highlight"].map({True: 18, False: 10})
    sensors["marker_line_width"] = sensors["highlight"].map({True: 3, False: 1})

    fig = go.Figure()
    _add_bridge_schematic(fig)
    for family, frame in sensors.groupby("sensor_family", dropna=False):
        fig.add_trace(
            go.Scatter(
                x=frame["display_x"],
                y=frame["display_y"],
                mode="markers+text",
                name=family or "Sensor",
                text=frame["channel"].where(frame["highlight"], ""),
                textposition="top center",
                marker={
                    "size": frame["marker_size"],
                    "symbol": _marker_symbol(family),
                    "color": frame["highlight"].map({True: "#e11d48", False: _family_color(family)}),
                    "line": {
                        "color": frame["highlight"].map({True: "#111827", False: "#334155"}),
                        "width": frame["marker_line_width"],
                    },
                },
                customdata=frame[
                    [
                        "channel",
                        "side_of_bridge",
                        "location_on_bridge",
                        "sensor_location",
                        "sensor_designation",
                        "orientation",
                        "placement_source",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Side: %{customdata[1]}<br>"
                    "Bridge location: %{customdata[2]}<br>"
                    "Sensor location: %{customdata[3]}<br>"
                    "Designation: %{customdata[4]}<br>"
                    "Orientation: %{customdata[5]}<br>"
                    "%{customdata[6]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        height=460,
        margin={"l": 24, "r": 24, "t": 54, "b": 24},
        xaxis={
            "title": "Longitudinal location from south fixed span toward south tower",
            "range": [0.5, 10.5],
            "dtick": 1,
        },
        yaxis={
            "title": "Bridge side / tower face",
            "range": [-0.85, 0.85],
            "tickmode": "array",
            "tickvals": [-0.45, 0.0, 0.45],
            "ticktext": ["East", "Tower / guide post", "West"],
        },
        legend_title_text="",
    )
    return fig


def sensor_location_table(catalog: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    if catalog.empty or not channels:
        return pd.DataFrame()
    sensors = catalog.copy()
    if "plan_x" not in sensors:
        sensors = enrich_sensor_catalog(sensors)
    sensors = sensors.drop_duplicates("channel")
    columns = [
        "channel",
        "sensor_family",
        "longitudinal_location",
        "side_of_bridge",
        "location_on_bridge",
        "sensor_location",
        "sensor_designation",
        "orientation",
        "placement_source",
    ]
    return sensors[sensors["channel"].isin(channels)][columns].sort_values("channel")


def event_channels(event: pd.Series | dict) -> list[str]:
    channels = str(event.get("channels", "")).split(", ")
    return [channel for channel in channels if channel]


def _base_location(
    channel: str,
    sensor_family: str,
    longitudinal_location: int,
    side_code: str,
    bridge_location_code: str,
    sensor_location_code: str,
) -> dict:
    side_y = -0.45 if side_code == "E" else 0.45
    bridge_label = BRIDGE_LOCATION_LABELS.get(bridge_location_code, bridge_location_code)
    sensor_label = SENSOR_LOCATION_LABELS.get(sensor_location_code, sensor_location_code)
    tower_offset = 0.1 if bridge_location_code == "TW" else 0.0
    return {
        "sensor_family": sensor_family,
        "longitudinal_location": longitudinal_location,
        "side_of_bridge": SIDE_LABELS.get(side_code, side_code),
        "side_code": side_code,
        "location_on_bridge": bridge_label,
        "bridge_location_code": bridge_location_code,
        "sensor_location": sensor_label,
        "sensor_location_code": sensor_location_code,
        "sensor_designation": "",
        "orientation": "",
        "orientation_code": "",
        "plan_x": float(longitudinal_location),
        "plan_y": side_y + tower_offset,
        "placement_confidence": "decoded from BDI sensor label",
        "placement_source": _placement_source(longitudinal_location),
    }


def _placement_source(location: int) -> str:
    if location == 8:
        return "UNH_MemorialBridge_InstPlan_V3_FieldInstall.pdf, LTM-09"
    if location in {9, 10}:
        return "UNH_MemorialBridge_InstPlan_V3_FieldInstall.pdf, LTM-10"
    return "UNH_MemorialBridge_InstPlan_V3_FieldInstall.pdf, LTM-04 to LTM-08"


def _add_bridge_schematic(fig: go.Figure) -> None:
    fig.add_shape(
        type="rect",
        x0=1,
        x1=8,
        y0=-0.62,
        y1=0.62,
        line={"color": "#94a3b8", "width": 1},
        fillcolor="#f8fafc",
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=8.75,
        x1=10.25,
        y0=-0.7,
        y1=0.7,
        line={"color": "#64748b", "width": 2},
        fillcolor="#eef2ff",
        layer="below",
    )
    for x in range(1, 11):
        fig.add_vline(x=x, line_color="#cbd5e1", line_width=1, opacity=0.55)
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1, opacity=0.5)
    fig.add_annotation(x=4.5, y=0.72, text="South fixed span", showarrow=False)
    fig.add_annotation(x=9.5, y=0.72, text="South tower", showarrow=False)
    fig.add_annotation(x=8.05, y=-0.75, text="Pier 2 / tower interface", showarrow=False)


def _marker_symbol(family: str) -> str:
    if family == "Accelerometer":
        return "circle"
    if "Rosette" in str(family):
        return "diamond"
    if "guide post" in str(family):
        return "square"
    return "circle-open"


def _family_color(family: str) -> str:
    if family == "Accelerometer":
        return "#2563eb"
    if "Rosette" in str(family):
        return "#f59e0b"
    if "guide post" in str(family):
        return "#16a34a"
    return "#64748b"


def _display_x(row: pd.Series) -> float:
    x = float(row["plan_x"])
    designation = str(row.get("sensor_designation", ""))
    if designation:
        x += (ord(designation[0]) - ord("A")) * 0.035 if designation[0].isalpha() else 0.0
    return x


def _display_y(row: pd.Series) -> float:
    y = float(row["plan_y"])
    orientation_offsets = {"H": 0.0, "D": 0.035, "V": -0.035}
    y += orientation_offsets.get(str(row.get("orientation_code", "")), 0.0)
    if str(row.get("sensor_location_code", "")) == "B":
        y -= 0.025
    return y
