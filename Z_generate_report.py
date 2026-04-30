from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from project_utils import (
    DOCS_DIR,
    EVENT_COLORS,
    EVENT_TYPE_ORDER,
    MACRO_COLORS,
    PROCESSED_DIR,
    admin_to_geo_name,
    build_actor_network,
    build_summary_tables,
    clean_actor_label,
    download_sudan_admin1_geojson,
    force_layout,
    geojson_centroids,
    load_events,
    load_panel,
    save_table,
    train_evaluate_risk_model,
)


def create_hero_image(events: pd.DataFrame, geojson: dict, output_path: Path) -> None:
    """Create a local bitmap hero image from Sudan boundaries and ACLED event points."""
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1100
    lon_min, lon_max = 20.5, 39.5
    lat_min, lat_max = 6.7, 23.4

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = (lon - lon_min) / (lon_max - lon_min) * width
        y = height - (lat - lat_min) / (lat_max - lat_min) * height
        return x, y

    image = Image.new("RGB", (width, height), "#e8edf2")
    draw = ImageDraw.Draw(image, "RGBA")

    for y in range(height):
        shade = int(236 - 42 * (y / height))
        draw.line([(0, y), (width, y)], fill=(shade, shade + 3, shade + 8, 255))

    for feature in geojson["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            if not polygon:
                continue
            exterior = [project(lon, lat) for lon, lat in polygon[0]]
            if len(exterior) >= 3:
                draw.polygon(exterior, fill=(246, 242, 232, 245), outline=(75, 85, 99, 185))

    point_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    point_draw = ImageDraw.Draw(point_layer, "RGBA")
    color_lookup = {
        "Battles": (53, 92, 125, 92),
        "Explosions/Remote violence": (192, 108, 132, 88),
        "Violence against civilians": (217, 95, 2, 108),
        "Strategic developments": (42, 157, 143, 76),
        "Protests": (106, 76, 147, 72),
        "Riots": (141, 110, 99, 72),
    }
    for row in events.dropna(subset=["longitude", "latitude"]).itertuples():
        x, y = project(float(row.longitude), float(row.latitude))
        if not (-20 <= x <= width + 20 and -20 <= y <= height + 20):
            continue
        radius = 2.1 + min(7, max(0, float(row.fatalities)) ** 0.35)
        color = color_lookup.get(row.event_type, (15, 23, 42, 70))
        point_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    point_layer = point_layer.filter(ImageFilter.GaussianBlur(radius=0.55))
    image = Image.alpha_composite(image.convert("RGBA"), point_layer)

    vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette, "RGBA")
    for i in range(260):
        alpha = int(120 * (i / 260) ** 2)
        vignette_draw.rectangle((i, i, width - i, height - i), outline=(17, 24, 39, alpha))
    image = Image.alpha_composite(image, vignette)
    image.convert("RGB").save(output_path, quality=92)


def write_processed_outputs(summaries: dict[str, pd.DataFrame], actor_edges: pd.DataFrame, actor_nodes: pd.DataFrame, ml: dict) -> None:
    save_table(summaries["monthly_event_type"], PROCESSED_DIR / "monthly_event_type_summary.csv")
    save_table(summaries["monthly_total"], PROCESSED_DIR / "monthly_total_summary.csv")
    save_table(summaries["admin_summary"], PROCESSED_DIR / "admin1_conflict_summary.csv")
    save_table(summaries["macro_summary"], PROCESSED_DIR / "macro_region_month_summary.csv")
    save_table(actor_edges, PROCESSED_DIR / "actor_network_edges.csv")
    save_table(actor_nodes, PROCESSED_DIR / "actor_network_nodes.csv")
    save_table(ml["latest_predictions"], PROCESSED_DIR / "ml_latest_risk_admin1.csv")
    save_table(ml["feature_importance"], PROCESSED_DIR / "ml_feature_importance.csv")
    save_table(pd.DataFrame([ml["metrics"]]), PROCESSED_DIR / "ml_evaluation_metrics.csv")
    save_table(ml["test_frame"], PROCESSED_DIR / "ml_test_predictions.csv")


def monthly_trend_figure(monthly_event_type: pd.DataFrame, monthly_total: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    pivot = monthly_event_type.pivot_table(
        index="month", columns="event_type", values="events", aggfunc="sum", fill_value=0
    ).sort_index()

    for event_type in EVENT_TYPE_ORDER:
        if event_type not in pivot.columns:
            continue
        fig.add_bar(
            x=pivot.index,
            y=pivot[event_type],
            name=event_type,
            marker_color=EVENT_COLORS.get(event_type, "#999999"),
            hovertemplate="%{x|%b %Y}<br>%{y:,} events<extra>" + event_type + "</extra>",
        )

    fig.add_scatter(
        x=monthly_total["month"],
        y=monthly_total["fatalities"],
        name="Reported fatalities",
        mode="lines+markers",
        line=dict(color="#111827", width=3),
        marker=dict(size=6),
        secondary_y=True,
        hovertemplate="%{x|%b %Y}<br>%{y:,} reported fatalities<extra></extra>",
    )
    fig.update_layout(
        barmode="stack",
        height=520,
        margin=dict(l=20, r=20, t=50, b=40),
        title="Monthly conflict events and reported fatalities",
        legend=dict(orientation="h", y=-0.20),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_yaxes(title_text="Events", secondary_y=False, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="Reported fatalities", secondary_y=True, gridcolor="#ffffff")
    fig.update_xaxes(title_text="")
    return fig


def event_point_map(events: pd.DataFrame, geojson: dict | None = None) -> go.Figure:
    if geojson is None:
        geojson = download_sudan_admin1_geojson()
    x_range, y_range = sudan_map_ranges(geojson, pad=0.28)
    map_events = events.dropna(subset=["latitude", "longitude"]).copy()
    map_events["fatalities_for_size"] = np.where(map_events["fatalities"].gt(0), map_events["fatalities"], 1)
    map_events["hover_label"] = (
        map_events["event_date"].dt.strftime("%Y-%m-%d")
        + "<br>"
        + map_events["admin1"].astype(str)
        + " / "
        + map_events["location"].astype(str)
        + "<br>"
        + map_events["event_type"].astype(str)
        + "<br>"
        + "Actor 1: "
        + map_events["actor1"].fillna("Unknown").astype(str)
        + "<br>"
        + "Actor 2: "
        + map_events["actor2"].fillna("None recorded").astype(str)
        + "<br>"
        + "Fatalities: "
        + map_events["fatalities"].astype(int).astype(str)
    )

    fig = go.Figure()
    for trace in sudan_boundary_traces(geojson, fill=True):
        fig.add_trace(trace)

    for event_type in EVENT_TYPE_ORDER:
        group = map_events.loc[map_events["event_type"].eq(event_type)]
        if group.empty:
            continue
        sizes = 3.5 + np.sqrt(group["fatalities_for_size"].clip(lower=1)) * 2.2
        fig.add_trace(
            go.Scattergl(
                x=group["longitude"],
                y=group["latitude"],
                mode="markers",
                name=event_type,
                marker=dict(
                    size=sizes.clip(3.5, 18),
                    color=EVENT_COLORS.get(event_type, "#64748b"),
                    opacity=0.66,
                    line=dict(width=0),
                ),
                text=group["hover_label"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Event-level geography of Sudan's civil war",
        height=880,
        margin=dict(l=10, r=10, t=56, b=18),
        legend=dict(orientation="h", y=0.02, x=0.5, xanchor="center", yanchor="bottom"),
        xaxis=dict(visible=False, range=x_range, constrain="domain"),
        yaxis=dict(visible=False, range=y_range, scaleanchor="x", scaleratio=1, domain=[0.08, 1.0]),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


POLYGON_COLORSCALE = [
    [0.0, "#edf2f7"],
    [0.25, "#fed7aa"],
    [0.55, "#fb923c"],
    [0.8, "#dc2626"],
    [1.0, "#7f1d1d"],
]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, value)):02x}" for value in rgb)


def _color_from_scale(value: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(value):
        return "#f8fafc"
    if vmax <= vmin:
        t = 0.5
    else:
        t = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
    for i in range(len(POLYGON_COLORSCALE) - 1):
        left_pos, left_color = POLYGON_COLORSCALE[i]
        right_pos, right_color = POLYGON_COLORSCALE[i + 1]
        if left_pos <= t <= right_pos:
            local_t = (t - left_pos) / (right_pos - left_pos) if right_pos > left_pos else 0
            left_rgb = _hex_to_rgb(left_color)
            right_rgb = _hex_to_rgb(right_color)
            rgb = tuple(int(left_rgb[j] + (right_rgb[j] - left_rgb[j]) * local_t) for j in range(3))
            return _rgb_to_hex(rgb)
    return POLYGON_COLORSCALE[-1][1]


def _feature_exterior_rings(feature: dict) -> list[list[list[float]]]:
    geometry = feature["geometry"]
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [polygon[0] for polygon in geometry["coordinates"] if polygon]
    return []


def sudan_boundary_traces(geojson: dict, fill: bool = False) -> list[go.Scatter]:
    traces = []
    for feature in geojson["features"]:
        for ring in _feature_exterior_rings(feature):
            x = [point[0] for point in ring]
            y = [point[1] for point in ring]
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    fill="toself" if fill else None,
                    fillcolor="rgba(248, 245, 239, 0.85)" if fill else None,
                    line=dict(color="#334155", width=0.75),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    return traces


def sudan_map_ranges(geojson: dict, pad: float = 0.35) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for feature in geojson["features"]:
        for ring in _feature_exterior_rings(feature):
            xs.extend(point[0] for point in ring)
            ys.extend(point[1] for point in ring)
    return [min(xs) - pad, max(xs) + pad], [min(ys) - pad, max(ys) + pad]


def _polygon_map_figure(
    table: pd.DataFrame,
    geojson: dict,
    metric_specs: list[tuple[str, str, str]],
    title_prefix: str,
    customdata_columns: list[str],
    hovertemplate: str,
    value_ranges: dict[str, tuple[float, float]] | None = None,
    height: int = 630,
) -> go.Figure:
    fig = go.Figure()
    value_ranges = value_ranges or {}
    table = table.copy()
    if "centroid_lon" not in table.columns or "centroid_lat" not in table.columns:
        table = table.merge(geojson_centroids(geojson), on=["admin1", "geo_admin1"], how="left")
    lookup = table.set_index("geo_admin1").to_dict(orient="index")
    all_traces_by_metric: list[list[int]] = []

    for metric_index, (metric, label, colorbar_title) in enumerate(metric_specs):
        metric_trace_indices = []
        values = pd.to_numeric(table[metric], errors="coerce").fillna(0)
        vmin, vmax = value_ranges.get(metric, (float(values.min()), float(values.max())))
        if vmax <= vmin:
            vmax = vmin + 1

        for feature in geojson["features"]:
            shape_name = feature["properties"]["shapeName"]
            row = lookup.get(shape_name)
            if row is None:
                continue
            value = float(row.get(metric, 0) or 0)
            fillcolor = _color_from_scale(value, vmin, vmax)
            for ring in _feature_exterior_rings(feature):
                x = [point[0] for point in ring]
                y = [point[1] for point in ring]
                customdata = [[row.get(col, "") for col in customdata_columns] for _ in x]
                fig.add_scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    fill="toself",
                    fillcolor=fillcolor,
                    line=dict(color="#334155", width=0.8),
                    name=str(row.get("admin1", shape_name)),
                    text=[row.get("admin1", shape_name)] * len(x),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    showlegend=False,
                    visible=metric_index == 0,
                )
                metric_trace_indices.append(len(fig.data) - 1)

        fig.add_scatter(
            x=table["centroid_lon"],
            y=table["centroid_lat"],
            mode="markers",
            marker=dict(
                size=0.1,
                opacity=0,
                color=values,
                cmin=vmin,
                cmax=vmax,
                colorscale=POLYGON_COLORSCALE,
                colorbar=dict(title=colorbar_title, thickness=18),
            ),
            hoverinfo="skip",
            showlegend=False,
            visible=metric_index == 0,
        )
        metric_trace_indices.append(len(fig.data) - 1)
        all_traces_by_metric.append(metric_trace_indices)

    buttons = []
    for metric_index, (_, label, _) in enumerate(metric_specs):
        visible = [False] * len(fig.data)
        for trace_index in all_traces_by_metric[metric_index]:
            visible[trace_index] = True
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[{"visible": visible}, {"title": f"{title_prefix}: {label}"}],
            )
        )

    if len(metric_specs) > 1:
        updatemenus = [
            dict(
                buttons=buttons,
                direction="down",
                x=0.02,
                y=0.99,
                xanchor="left",
                yanchor="top",
                showactive=True,
            )
        ]
    else:
        updatemenus = []

    fig.update_layout(
        title=f"{title_prefix}: {metric_specs[0][1]}" if len(metric_specs) > 1 else title_prefix,
        height=height,
        margin=dict(l=10, r=10, t=98 if len(metric_specs) > 1 else 54, b=10),
        updatemenus=updatemenus,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(
            visible=False,
            range=[21.0, 39.0],
            constrain="domain",
        ),
        yaxis=dict(
            visible=False,
            range=[7.5, 23.5],
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    return fig


def admin_choropleth(admin_summary: pd.DataFrame, geojson: dict) -> go.Figure:
    metric_specs = [
        ("events", "All events", "# events"),
        ("fatalities", "Reported fatalities", "# fatalities"),
        ("civilian_targeting_events", "Civilian-targeting events", "# civilian-targeting events"),
        ("civilian_targeting_share_pct", "Civilian-targeting share", "Share of all events (%)"),
    ]
    map_table = admin_summary.copy()
    map_table["civilian_targeting_share_pct"] = (map_table["civilian_targeting_share"] * 100).round(1)
    return _polygon_map_figure(
        table=map_table,
        geojson=geojson,
        metric_specs=metric_specs,
        title_prefix="State-level geography",
        customdata_columns=[
            "events",
            "fatalities",
            "civilian_targeting_events",
            "civilian_targeting_share_pct",
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Events: %{customdata[0]:,}<br>"
            "Fatalities: %{customdata[1]:,}<br>"
            "Civilian targeting: %{customdata[2]:,}<br>"
            "Civilian-targeting share: %{customdata[3]}%<extra></extra>"
        ),
        value_ranges={"civilian_targeting_share_pct": (0, max(1, map_table["civilian_targeting_share_pct"].max()))},
        height=580,
    )


def animated_admin_map(panel: pd.DataFrame, centroids: pd.DataFrame, geojson: dict | None = None) -> go.Figure:
    if geojson is None:
        geojson = download_sudan_admin1_geojson()
    x_range, y_range = sudan_map_ranges(geojson, pad=0.30)
    frame = panel.merge(centroids[["admin1", "centroid_lat", "centroid_lon"]], on="admin1", how="left")
    frame["events_for_size"] = frame["events"].clip(lower=1)
    frame["risk_label"] = np.where(frame["civilian_targeting_events"].gt(0), "Recorded civilian targeting", "No recorded civilian targeting")

    months = sorted(frame["year_month"].dropna().unique().tolist())
    max_civ = max(1, float(frame["civilian_targeting_events"].max()))
    max_events = max(1, float(frame["events_for_size"].max()))

    def marker_trace(month: str) -> go.Scatter:
        subset = frame.loc[frame["year_month"].eq(month)].copy()
        sizes = 8 + np.sqrt(subset["events_for_size"] / max_events) * 42
        return go.Scatter(
            x=subset["centroid_lon"],
            y=subset["centroid_lat"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=subset["civilian_targeting_events"],
                cmin=0,
                cmax=max_civ,
                colorscale="OrRd",
                opacity=0.82,
                line=dict(width=1.1, color="#ffffff"),
                colorbar=dict(title="civilian_targeting_events", x=0.90, y=0.66, len=0.62, thickness=16),
            ),
            text=subset["admin1"],
            customdata=np.stack(
                [
                    subset["events"],
                    subset["fatalities"],
                    subset["civilian_targeting_events"],
                    subset["battles"],
                    subset["explosions_remote"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Events: %{customdata[0]:,}<br>"
                "Fatalities: %{customdata[1]:,}<br>"
                "Civilian targeting: %{customdata[2]:,}<br>"
                "Battles: %{customdata[3]:,}<br>"
                "Remote violence: %{customdata[4]:,}<extra></extra>"
            ),
            name=month,
            showlegend=False,
        )

    initial_month = months[0] if months else ""
    fig = go.Figure()
    for trace in sudan_boundary_traces(geojson, fill=True):
        fig.add_trace(trace)
    marker_index = len(fig.data)
    fig.add_trace(marker_trace(initial_month))

    frames = [
        go.Frame(data=[marker_trace(month)], traces=[marker_index], name=month)
        for month in months
    ]
    fig.frames = frames

    slider_steps = [
        dict(
            method="animate",
            label=month,
            args=[
                [month],
                dict(mode="immediate", frame=dict(duration=350, redraw=True), transition=dict(duration=120)),
            ],
        )
        for month in months
    ]

    fig.update_layout(
        title="Monthly conflict spread by state",
        height=600,
        margin=dict(l=0, r=16, t=54, b=0),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(visible=False, range=x_range, constrain="domain", domain=[0.04, 0.86]),
        yaxis=dict(visible=False, range=y_range, scaleanchor="x", scaleratio=1, domain=[0.30, 0.97]),
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.22,
                y=0.18,
                len=0.50,
                currentvalue=dict(prefix="year_month=", font=dict(size=13)),
                pad=dict(t=10, b=0),
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.05,
                y=0.205,
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=450, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=120),
                            ),
                        ],
                    ),
                    dict(
                        label="■",
                        method="animate",
                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))],
                    ),
                ],
                pad=dict(r=4, t=0),
                showactive=False,
            )
        ],
    )
    return fig


def actor_network_figure(edges: pd.DataFrame, nodes: pd.DataFrame) -> go.Figure:
    layout_nodes = force_layout(edges, nodes)
    positions = layout_nodes.set_index("actor")[["x", "y"]].to_dict("index")

    edge_x, edge_y = [], []
    for _, edge in edges.iterrows():
        if edge["actor1"] not in positions or edge["actor2"] not in positions:
            continue
        x0, y0 = positions[edge["actor1"]]["x"], positions[edge["actor1"]]["y"]
        x1, y1 = positions[edge["actor2"]]["x"], positions[edge["actor2"]]["y"]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.1, color="rgba(71,85,105,0.32)"),
        hoverinfo="skip",
        showlegend=False,
    )

    actor_type_colors = {
        "SAF": "#355C7D",
        "RSF": "#D95F02",
        "Civilians": "#111827",
        "Civil society": "#6A4C93",
        "Armed group": "#2A9D8F",
        "State-aligned": "#7C3AED",
        "Other": "#8D6E63",
    }

    top_label_cutoff = layout_nodes["weighted_degree"].rank(method="first", ascending=False).le(14)
    layout_nodes["display_text"] = np.where(top_label_cutoff, layout_nodes["actor_short"], "")

    for actor_type, group in layout_nodes.groupby("actor_type"):
        fig.add_scatter(
            x=group["x"],
            y=group["y"],
            mode="markers+text",
            text=group["display_text"],
            textposition="top center",
            name=actor_type,
            marker=dict(
                size=group["node_size"],
                color=actor_type_colors.get(actor_type, "#64748b"),
                line=dict(width=1.5, color="#ffffff"),
                opacity=0.92,
            ),
            customdata=np.stack([group["actor"], group["weighted_degree"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>Weighted degree: %{customdata[1]:,}<extra></extra>",
        )

    fig.update_layout(
        title="Actor co-involvement network, top event-pair ties",
        height=620,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", y=-0.04),
    )
    return fig


def ml_feature_figure(feature_importance: pd.DataFrame) -> go.Figure:
    top = feature_importance.head(14).copy()
    top["direction"] = np.where(top["coefficient"].ge(0), "Increases predicted risk", "Decreases predicted risk")
    top = top.sort_values("coefficient")
    fig = px.bar(
        top,
        x="coefficient",
        y="feature",
        color="direction",
        orientation="h",
        color_discrete_map={
            "Increases predicted risk": "#D95F02",
            "Decreases predicted risk": "#355C7D",
        },
        title="Risk model feature effects",
    )
    fig.add_vline(x=0, line_width=1, line_color="#111827")
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=60, b=30),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", y=-0.16),
        xaxis_title="Standardized logistic coefficient",
        yaxis_title="",
    )
    fig.update_xaxes(gridcolor="#e5e7eb")
    return fig


def ml_risk_map(latest_predictions: pd.DataFrame, geojson: dict) -> go.Figure:
    latest = latest_predictions.copy()
    latest["geo_admin1"] = latest["admin1"].map(admin_to_geo_name)
    latest["risk_percent"] = latest["predicted_high_risk_probability"] * 100
    latest["risk_percent_label"] = latest["risk_percent"].round(1)
    return _polygon_map_figure(
        table=latest,
        geojson=geojson,
        metric_specs=[("risk_percent", "Predicted risk", "Predicted risk (%)")],
        title_prefix="Model-estimated next-month high civilian-targeting risk",
        customdata_columns=[
            "risk_percent_label",
            "events",
            "civilian_targeting_events",
            "fatalities",
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Predicted high-risk probability: %{customdata[0]}%<br>"
            "Current-month events: %{customdata[1]:,}<br>"
            "Current civilian-targeting events: %{customdata[2]:,}<br>"
            "Current fatalities: %{customdata[3]:,}<extra></extra>"
        ),
        value_ranges={"risk_percent": (0, 100)},
    )


def build_dashboard_section(events: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> tuple[str, str]:
    dash_events = events.copy()
    dash_events["civilian_targeting_count"] = dash_events["is_civilian_targeting"].astype(int)

    monthly = (
        dash_events.groupby(["macro_region", "admin1", "year_month", "event_type"], as_index=False)
        .agg(
            events=("event_id_cnty", "count"),
            fatalities=("fatalities", "sum"),
            civilian_targeting_events=("civilian_targeting_count", "sum"),
        )
        .sort_values(["year_month", "macro_region", "admin1", "event_type"])
    )

    actor_long = (
        dash_events[["macro_region", "admin1", "actor1", "actor2"]]
        .melt(id_vars=["macro_region", "admin1"], value_vars=["actor1", "actor2"], value_name="actor")
        .dropna(subset=["actor"])
    )
    actor_long["actor"] = actor_long["actor"].astype(str)
    actor_long = actor_long.loc[~actor_long["actor"].str.contains("Civilians", case=False, na=False)]
    actor_long["actor_label"] = actor_long["actor"].map(lambda value: clean_actor_label(value, max_len=26))
    actors = (
        actor_long.groupby(["macro_region", "admin1", "actor_label"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )

    payload = {
        "monthly": monthly.to_dict(orient="records"),
        "actors": actors.to_dict(orient="records"),
        "regions": ["All"] + sorted(dash_events["macro_region"].dropna().unique().tolist()),
        "admins": ["All"] + sorted(dash_events["admin1"].dropna().unique().tolist()),
        "eventTypes": ["All"] + EVENT_TYPE_ORDER,
    }
    payload_json = json.dumps(payload).replace("</", "<\\/")

    html_markup = """
    <section class="dashboard-zone" aria-label="Linked conflict story explorer">
      <div class="dashboard-head">
        <div>
          <p class="eyebrow-dark">Interactive dashboard layer</p>
          <h2><span class="figure-label">Figure 2.</span> Conflict Story Explorer</h2>
          <p>Use the controls once and the time series, state ranking, event-type mix, actor ranking, and narrative insight update together. Click a state bar to drill into that state.</p>
        </div>
      </div>
      <div class="dash-controls">
        <label>Region
          <select id="dash-region"></select>
        </label>
        <label>State (admin1)
          <select id="dash-admin"></select>
        </label>
        <label>Event Type
          <select id="dash-event-type"></select>
        </label>
        <label>Measure
          <select id="dash-measure">
            <option value="events">Events</option>
            <option value="fatalities">Reported fatalities</option>
            <option value="civilian_targeting_events">Civilian-targeting events</option>
            <option value="civilian_targeting_share">Civilian-targeting share</option>
          </select>
        </label>
      </div>
      <div class="dash-kpis" id="dash-kpis">
        <div><span id="dash-kpi-events">0</span><p>events</p></div>
        <div><span id="dash-kpi-fatalities">0</span><p>reported fatalities</p></div>
        <div><span id="dash-kpi-civ">0</span><p>civilian-targeting events</p></div>
        <div><span id="dash-kpi-share">0%</span><p>civilian-targeting share</p></div>
      </div>
      <div class="dash-grid">
        <div class="dash-panel wide"><div id="dash-time"></div></div>
        <div class="dash-panel"><div id="dash-states"></div></div>
        <div class="dash-panel"><div id="dash-types"></div></div>
        <div class="dash-panel wide"><div id="dash-actors"></div></div>
      </div>
      <p class="dash-insight" id="dash-insight"></p>
    </section>
    """

    script = f"""
    <script>
    (function() {{
      const payload = {payload_json};
      const regionEl = document.getElementById('dash-region');
      const adminEl = document.getElementById('dash-admin');
      const eventTypeEl = document.getElementById('dash-event-type');
      const measureEl = document.getElementById('dash-measure');
      const fmt = new Intl.NumberFormat('en-US');

      function fillSelect(el, values) {{
        const current = el.value;
        el.innerHTML = '';
        values.forEach(value => {{
          const option = document.createElement('option');
          option.value = value;
          option.textContent = value;
          el.appendChild(option);
        }});
        if (values.includes(current)) el.value = current;
      }}

      fillSelect(regionEl, payload.regions);
      fillSelect(adminEl, payload.admins);
      fillSelect(eventTypeEl, payload.eventTypes);

      function adminsForRegion(region) {{
        if (region === 'All') return payload.admins;
        const admins = new Set(['All']);
        payload.monthly.forEach(row => {{ if (row.macro_region === region) admins.add(row.admin1); }});
        return Array.from(admins).sort((a, b) => a === 'All' ? -1 : b === 'All' ? 1 : a.localeCompare(b));
      }}

      function filteredMonthly() {{
        const region = regionEl.value;
        const admin = adminEl.value;
        const eventType = eventTypeEl.value;
        return payload.monthly.filter(row =>
          (region === 'All' || row.macro_region === region) &&
          (admin === 'All' || row.admin1 === admin) &&
          (eventType === 'All' || row.event_type === eventType)
        );
      }}

      function filteredActors() {{
        const region = regionEl.value;
        const admin = adminEl.value;
        return payload.actors.filter(row =>
          (region === 'All' || row.macro_region === region) &&
          (admin === 'All' || row.admin1 === admin)
        );
      }}

      function sumBy(rows, key, valueFields) {{
        const map = new Map();
        rows.forEach(row => {{
          const id = row[key];
          if (!map.has(id)) map.set(id, Object.fromEntries(valueFields.map(v => [v, 0])));
          const target = map.get(id);
          valueFields.forEach(v => target[v] += Number(row[v] || 0));
        }});
        return Array.from(map, ([name, values]) => ({{ name, ...values }}));
      }}

      function metricValue(row, measure) {{
        if (measure === 'civilian_targeting_share') {{
          return row.events ? (row.civilian_targeting_events / row.events) * 100 : 0;
        }}
        return row[measure] || 0;
      }}

      function measureTitle(measure) {{
        return {{
          events: 'Events',
          fatalities: 'Reported fatalities',
          civilian_targeting_events: 'Civilian-targeting events',
          civilian_targeting_share: 'Civilian-targeting share (%)'
        }}[measure];
      }}

      function render() {{
        const rows = filteredMonthly();
        const measure = measureEl.value;
        const totals = rows.reduce((acc, row) => {{
          acc.events += Number(row.events || 0);
          acc.fatalities += Number(row.fatalities || 0);
          acc.civilian += Number(row.civilian_targeting_events || 0);
          return acc;
        }}, {{ events: 0, fatalities: 0, civilian: 0 }});
        const share = totals.events ? totals.civilian / totals.events * 100 : 0;

        document.getElementById('dash-kpi-events').textContent = fmt.format(totals.events);
        document.getElementById('dash-kpi-fatalities').textContent = fmt.format(totals.fatalities);
        document.getElementById('dash-kpi-civ').textContent = fmt.format(totals.civilian);
        document.getElementById('dash-kpi-share').textContent = share.toFixed(1) + '%';

        const monthly = sumBy(rows, 'year_month', ['events', 'fatalities', 'civilian_targeting_events'])
          .sort((a, b) => a.name.localeCompare(b.name));
        const monthlyY = monthly.map(row => metricValue(row, measure));
        Plotly.react('dash-time', [{{
          type: measure === 'civilian_targeting_share' ? 'scatter' : 'bar',
          mode: 'lines+markers',
          x: monthly.map(row => row.name),
          y: monthlyY,
          marker: {{ color: measure === 'fatalities' ? '#111827' : '#d95f02' }},
          line: {{ color: '#355c7d', width: 3 }},
          hovertemplate: '%{{x}}<br>%{{y:,.2f}}<extra></extra>'
        }}], {{
          title: measureTitle(measure) + ' over time',
          height: 330,
          margin: {{ l: 52, r: 20, t: 54, b: 72 }},
          plot_bgcolor: '#ffffff',
          paper_bgcolor: '#ffffff',
          xaxis: {{ tickangle: -45, rangeslider: {{ visible: true, thickness: 0.08 }} }},
          yaxis: {{ title: measureTitle(measure), gridcolor: '#e5e7eb' }}
        }}, {{ responsive: true, displayModeBar: false }});

        const states = sumBy(rows, 'admin1', ['events', 'fatalities', 'civilian_targeting_events'])
          .map(row => ({{ ...row, value: metricValue(row, measure) }}))
          .sort((a, b) => b.value - a.value)
          .slice(0, 12)
          .reverse();
        Plotly.react('dash-states', [{{
          type: 'bar',
          orientation: 'h',
          y: states.map(row => row.name),
          x: states.map(row => row.value),
          marker: {{ color: '#2a9d8f' }},
          customdata: states.map(row => row.name),
          hovertemplate: '%{{y}}<br>%{{x:,.2f}}<extra>Click to drill down</extra>'
        }}], {{
          title: 'Top states by selected measure',
          height: 330,
          margin: {{ l: 122, r: 18, t: 54, b: 42 }},
          plot_bgcolor: '#ffffff',
          paper_bgcolor: '#ffffff',
          xaxis: {{ gridcolor: '#e5e7eb' }}
        }}, {{ responsive: true, displayModeBar: false }});

        const typeRows = sumBy(rows, 'event_type', ['events', 'fatalities', 'civilian_targeting_events'])
          .filter(row => row.events > 0)
          .sort((a, b) => b.events - a.events);
        Plotly.react('dash-types', [{{
          type: 'pie',
          labels: typeRows.map(row => row.name),
          values: typeRows.map(row => row.events),
          hole: 0.48,
          marker: {{ colors: ['#355c7d','#c06c84','#d95f02','#2a9d8f','#6a4c93','#8d6e63'] }},
          hovertemplate: '%{{label}}<br>%{{value:,}} events<extra></extra>'
        }}], {{
          title: 'Event-type composition',
          height: 330,
          margin: {{ l: 18, r: 18, t: 54, b: 18 }},
          paper_bgcolor: '#ffffff',
          legend: {{ orientation: 'h', y: -0.08 }}
        }}, {{ responsive: true, displayModeBar: false }});

        const actorMap = new Map();
        filteredActors().forEach(row => actorMap.set(row.actor_label, (actorMap.get(row.actor_label) || 0) + Number(row.events || 0)));
        const actors = Array.from(actorMap, ([actor, events]) => ({{ actor, events }}))
          .sort((a, b) => b.events - a.events)
          .slice(0, 10)
          .reverse();
        Plotly.react('dash-actors', [{{
          type: 'bar',
          orientation: 'h',
          y: actors.map(row => row.actor),
          x: actors.map(row => row.events),
          marker: {{ color: '#6a4c93' }},
          hovertext: actors.map(row => row.actor),
          hovertemplate: '%{{hovertext}}<br>%{{x:,}} mentions<extra></extra>'
        }}], {{
          title: 'Most frequent non-civilian actors',
          height: 330,
          margin: {{ l: 230, r: 18, t: 54, b: 42 }},
          plot_bgcolor: '#ffffff',
          paper_bgcolor: '#ffffff',
          xaxis: {{ gridcolor: '#e5e7eb' }}
        }}, {{ responsive: true, displayModeBar: false }});

        const peak = monthly.length ? monthly.reduce((a, b) => metricValue(b, measure) > metricValue(a, measure) ? b : a) : null;
        const topState = states.length ? states[states.length - 1] : null;
        const scope = [regionEl.value !== 'All' ? regionEl.value : 'Sudan', adminEl.value !== 'All' ? adminEl.value : null].filter(Boolean).join(' / ');
        document.getElementById('dash-insight').textContent = peak && topState
          ? `${{scope}}: ${{measureTitle(measure).toLowerCase()}} peaks in ${{peak.name}}, while ${{topState.name}} is the leading state under the current filters. Civilian-targeting events make up ${{share.toFixed(1)}}% of recorded events in this selection.`
          : 'No records match the current filter.';

        const statePlot = document.getElementById('dash-states');
        statePlot.removeAllListeners && statePlot.removeAllListeners('plotly_click');
        statePlot.on && statePlot.on('plotly_click', event => {{
          const clicked = event.points && event.points[0] && event.points[0].customdata;
          if (clicked) {{
            if (!Array.from(adminEl.options).some(option => option.value === clicked)) {{
              fillSelect(adminEl, ['All', clicked]);
            }}
            adminEl.value = clicked;
            render();
          }}
        }});
      }}

      regionEl.addEventListener('change', () => {{
        fillSelect(adminEl, adminsForRegion(regionEl.value));
        adminEl.value = 'All';
        render();
      }});
      adminEl.addEventListener('change', render);
      eventTypeEl.addEventListener('change', render);
      measureEl.addEventListener('change', render);
      render();
    }})();
    </script>
    """
    return html_markup, script


def confusion_matrix_figure(metrics: dict) -> go.Figure:
    z = [
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ]
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{labels[i][j]}<br>{z[i][j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted low", "Predicted high"],
            y=["Actual low", "Actual high"],
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
            colorbar=dict(thickness=18, len=0.74, x=1.02),
        )
    )
    fig.update_layout(
        title="Held-out test confusion matrix",
        height=520,
        margin=dict(l=70, r=58, t=62, b=70),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(side="bottom", constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def figure_html(figures: list[go.Figure]) -> list[str]:
    snippets = []
    for i, fig in enumerate(figures):
        snippets.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs=True if i == 0 else False,
                config={"displayModeBar": False, "responsive": True},
            )
        )
    return snippets


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def build_report_html(events: pd.DataFrame, panel: pd.DataFrame, summaries: dict[str, pd.DataFrame], ml: dict, fig_snippets: list[str]) -> str:
    monthly_total = summaries["monthly_total"]
    admin_summary = summaries["admin_summary"]
    actor_top = pd.read_csv(PROCESSED_DIR / "actor_network_nodes.csv").head(8)
    latest = ml["latest_predictions"].copy()
    metrics = ml["metrics"]

    date_min = events["event_date"].min().strftime("%d %B %Y")
    date_max = events["event_date"].max().strftime("%d %B %Y")
    total_events = len(events)
    total_fatalities = int(events["fatalities"].sum())
    civilian_targeting = int(events["is_civilian_targeting"].sum())
    violence_against_civilians = int(events["is_violence_against_civilians"].sum())
    top_admin_events = admin_summary.iloc[0]["admin1"]
    top_admin_events_count = int(admin_summary.iloc[0]["events"])
    top_admin_vac = admin_summary.sort_values("civilian_targeting_events", ascending=False).iloc[0]
    top_admin_fatal = admin_summary.sort_values("fatalities", ascending=False).iloc[0]
    peak_month = monthly_total.sort_values("events", ascending=False).iloc[0]
    peak_fatal_month = monthly_total.sort_values("fatalities", ascending=False).iloc[0]
    latest_month_label = panel["month"].max().strftime("%B %Y")
    test_period = f"{metrics['test_start']} to {metrics['test_end']}"
    high_risk_states = latest.head(6)[["admin1", "predicted_high_risk_probability", "events", "civilian_targeting_events"]]
    dashboard_markup, dashboard_script = build_dashboard_section(events, summaries)
    event_type_counts = events["event_type"].value_counts()

    def event_type_text(event_type: str) -> str:
        count = int(event_type_counts.get(event_type, 0))
        share = count / total_events * 100 if total_events else 0
        return f"{fmt_int(count)} ({share:.1f}%)"

    high_risk_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row.admin1)}</td>"
        f"<td>{row.predicted_high_risk_probability * 100:.1f}%</td>"
        f"<td>{int(row.events):,}</td>"
        f"<td>{int(row.civilian_targeting_events):,}</td>"
        "</tr>"
        for row in high_risk_states.itertuples()
    )

    actor_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row.actor_short)}</td>"
        f"<td>{html.escape(row.actor_type)}</td>"
        f"<td>{int(row.weighted_degree):,}</td>"
        "</tr>"
        for row in actor_top.itertuples()
    )

    kpi_cards = f"""
    <section class="kpis" aria-label="Key project metrics">
      <div><span>{fmt_int(total_events)}</span><p>ACLED events</p></div>
      <div><span>{fmt_int(total_fatalities)}</span><p>reported fatalities</p></div>
      <div><span>{fmt_int(civilian_targeting)}</span><p>civilian-targeting events</p></div>
      <div><span>{events['admin1'].nunique()}</span><p>state-level areas</p></div>
    </section>
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sudan Civil War ACLED Analysis</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #526071;
      --line: #d9e2ec;
      --paper: #ffffff;
      --wash: #f6f8fb;
      --accent: #d95f02;
      --blue: #355c7d;
      --green: #2a9d8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--paper);
      line-height: 1.58;
    }}
    header {{
      min-height: 88vh;
      display: grid;
      align-items: end;
      background:
        linear-gradient(90deg, rgba(17,24,39,0.74), rgba(17,24,39,0.28) 58%, rgba(17,24,39,0.68)),
        linear-gradient(180deg, rgba(17,24,39,0.08), rgba(17,24,39,0.78)),
        url("assets/hero_sudan_conflict.png");
      background-size: cover;
      background-position: center;
      color: white;
      padding: 8vw 7vw 5vw;
    }}
    header .eyebrow {{
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: .82rem;
      font-weight: 700;
      color: #d9f3ef;
      margin: 0 0 1rem;
    }}
    .eyebrow-dark {{
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: .78rem;
      font-weight: 800;
      color: #0f766e;
      margin: 0 0 .65rem;
    }}
    header h1 {{
      max-width: 1100px;
      font-size: clamp(2.2rem, 6vw, 5.8rem);
      line-height: .98;
      margin: 0;
      font-weight: 820;
    }}
    header p {{
      max-width: 780px;
      font-size: clamp(1rem, 1.7vw, 1.35rem);
      margin: 1.2rem 0 0;
      color: #edf7f4;
    }}
    main {{ background: white; }}
    section {{
      width: min(1180px, calc(100vw - 36px));
      margin: 0 auto;
      padding: 3.5rem 0;
    }}
    .narrative {{
      width: min(860px, calc(100vw - 36px));
    }}
    h2 {{
      font-size: clamp(1.7rem, 3vw, 2.55rem);
      line-height: 1.08;
      margin: 0 0 1rem;
      color: var(--ink);
    }}
    h3 {{
      font-size: 1.15rem;
      margin: 1.6rem 0 .5rem;
    }}
    p {{ color: var(--muted); font-size: 1.04rem; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1rem;
      padding-top: 2rem;
      padding-bottom: 2rem;
    }}
    .kpis div {{
      border-top: 3px solid var(--accent);
      background: var(--wash);
      padding: 1rem;
      min-height: 110px;
    }}
    .kpis span {{
      display: block;
      font-size: clamp(1.8rem, 4vw, 3rem);
      font-weight: 800;
      color: var(--ink);
    }}
    .kpis p {{ margin: .25rem 0 0; }}
    .figure-band {{
      width: 100%;
      background: #f7fafc;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      margin: 1rem 0;
      padding: 2.5rem 0;
    }}
    .figure-wrap {{
      width: min(1220px, calc(100vw - 28px));
      margin: 0 auto;
      background: white;
      border: 1px solid var(--line);
      padding: 1rem;
    }}
    .figure-wrap.map-focus {{
      width: min(1320px, calc(100vw - 18px));
    }}
    .figure-pair {{
      width: min(1220px, calc(100vw - 28px));
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1rem;
      align-items: stretch;
    }}
    .figure-pair .figure-wrap {{
      width: 100%;
      min-width: 0;
      margin: 0;
    }}
    .figure-stack {{
      width: min(1220px, calc(100vw - 28px));
      margin: 0 auto;
      display: grid;
      gap: 0;
    }}
    .figure-stack .figure-wrap {{
      width: 100%;
      margin: 0;
    }}
    .caption {{
      color: var(--muted);
      font-size: .95rem;
      margin: .7rem .2rem 0;
    }}
    .caption strong,
    .figure-label {{
      color: var(--ink);
      font-weight: 820;
    }}
    .figure-label {{
      margin-right: .25rem;
    }}
    .dashboard-zone {{
      width: min(1220px, calc(100vw - 28px));
      background: #ffffff;
      padding: 3rem 0 3.5rem;
    }}
    .dashboard-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 1.5rem;
      margin-bottom: 1rem;
    }}
    .dashboard-head p {{
      max-width: 820px;
    }}
    .dash-controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: .8rem;
      padding: 1rem;
      border: 1px solid var(--line);
      background: #f8fafc;
      margin-bottom: 1rem;
    }}
    .dash-controls label {{
      display: grid;
      gap: .35rem;
      color: #334155;
      font-size: .84rem;
      font-weight: 750;
    }}
    .dash-controls select {{
      width: 100%;
      min-height: 42px;
      border: 1px solid #cbd5e1;
      background: white;
      color: var(--ink);
      padding: .45rem .55rem;
      font-size: .95rem;
    }}
    .dash-kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: .8rem;
      margin-bottom: 1rem;
    }}
    .dash-kpis div {{
      border-left: 4px solid var(--green);
      background: #f8fafc;
      padding: .9rem 1rem;
      min-height: 96px;
    }}
    .dash-kpis span {{
      display: block;
      font-size: clamp(1.45rem, 3vw, 2.45rem);
      line-height: 1.05;
      font-weight: 850;
      color: var(--ink);
    }}
    .dash-kpis p {{
      margin: .3rem 0 0;
      font-size: .92rem;
    }}
    .dash-grid {{
      display: grid;
      grid-template-columns: 1.15fr .85fr;
      gap: 1rem;
    }}
    .dash-panel {{
      min-height: 350px;
      border: 1px solid var(--line);
      background: #ffffff;
      padding: .35rem;
    }}
    .dash-panel.wide {{
      grid-column: span 2;
    }}
    .dash-insight {{
      border-left: 4px solid var(--accent);
      background: #fff7ed;
      padding: .9rem 1rem;
      margin: 1rem 0 0;
      color: #7c2d12;
      font-weight: 650;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1rem 0;
      font-size: .96rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: .7rem .6rem;
      text-align: left;
    }}
    th {{ color: var(--ink); background: #f8fafc; }}
    .method-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1.2rem;
    }}
    .method-grid > div {{
      border-left: 4px solid var(--blue);
      background: var(--wash);
      padding: 1rem;
    }}
    .term-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 1rem;
      margin: 1.2rem 0 0;
    }}
    .term-grid > div {{
      background: #f8fafc;
      border-top: 3px solid var(--green);
      padding: 1rem;
    }}
    .term-grid strong {{
      display: block;
      color: var(--ink);
      margin-bottom: .35rem;
    }}
    .term-grid p {{
      margin: 0;
      font-size: .96rem;
    }}
    .refs li {{ margin-bottom: .55rem; color: var(--muted); }}
    .apa-references {{
      list-style: none;
      padding-left: 0;
    }}
    .apa-references li {{
      margin: 0 0 .85rem 1.8rem;
      text-indent: -1.8rem;
      color: var(--muted);
    }}
    a {{ color: #0f766e; }}
    footer {{
      border-top: 1px solid var(--line);
      padding: 2rem 7vw;
      color: var(--muted);
      background: #f8fafc;
    }}
    @media (max-width: 820px) {{
      .kpis, .method-grid, .term-grid, .dash-controls, .dash-kpis, .dash-grid, .figure-pair {{ grid-template-columns: 1fr; }}
      .dash-panel.wide {{ grid-column: span 1; }}
      header {{ min-height: 78vh; }}
      .figure-wrap {{ padding: .4rem; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <p class="eyebrow">POLI3148 Assignment 1 · ACLED event data</p>
      <h1>From Capital War to Fragmented Civilian Insecurity</h1>
      <p>Mapping and identifying (by predictive modelling) violence against civilians in Sudan's civil war using ACLED records from {date_min} through {date_max}.</p>
    </div>
  </header>
  <main>
    {kpi_cards}
    <section class="narrative">
      <h2>Research Question</h2>
      <p>This report asks: <strong>How has Sudan's civil war shifted geographically and politically since April 2023, and can ACLED event patterns help identify where violence against civilians is most likely to intensify?</strong> The analysis treats ACLED as an event-level record of reported political disorder rather than a complete census of harm. That distinction matters: the dataset is excellent for comparing reported patterns across time, place, event type, and actors, but fatalities and civilian harm are still shaped by source availability, access, and coding rules.</p>
    </section>

    <section class="narrative">
      <h2>Background and Key Terms</h2>
      <p>Sudan's current war began on 15 April 2023 as a confrontation between the Sudanese Armed Forces (SAF) and the Rapid Support Forces (RSF). The conflict first produced intense fighting in Khartoum, where control over the capital carried both military and symbolic value, but it later expanded through Darfur, Al Jazirah, Kordofan, Sennar, and other state-level arenas. ACLED's Sudan analysis describes this evolution as a war shaped by shifting front lines, fragmented alliances, and external support, rather than a static two-party battlefield (Ali et al., 2025; Birru, 2024).</p>
      <p>ACLED records reported political violence, demonstrations, and strategic developments by date, location, actors, event type, and reported fatalities (ACLED, 2024; Raleigh et al., 2010). The report uses these records to compare conflict patterns over time and across Sudan's states, while treating the dataset as a record of observed and coded events rather than a full measure of all harm.</p>
      <div class="term-grid">
        <div><strong>Admin1 / state</strong><p>ACLED's <code>admin1</code> field is the largest subnational administrative unit. In this dashboard, it is shown as “state” or “state-level area” for readability.</p></div>
        <div><strong>State-month</strong><p>A state-month is one Sudanese state in one calendar month. This is the unit used for trend summaries and the predictive risk model.</p></div>
        <div><strong>Civilian targeting</strong><p>ACLED marks events where civilians are directly targeted. This report uses that coding to track civilian-facing insecurity, not total civilian harm.</p></div>
      </div>
    </section>

    <div class="figure-band"><div class="figure-wrap">{fig_snippets[0]}<p class="caption"><strong>Figure 1.</strong> Monthly ACLED events and reported fatalities in Sudan, April 2023-April 2025. Stacked bars show monthly event counts by ACLED event type; the black line shows reported fatalities.</p></div></div>
    {dashboard_markup}
    {dashboard_script}

    <section class="narrative">
      <h2>Finding 1: The War Begins as a Capital-Centered Contest</h2>
      <p>The first phase is unmistakably centered on Khartoum. Across the export, {html.escape(top_admin_events)} records {fmt_int(top_admin_events_count)} events, more than any other state. The monthly series (<em>Figure 1</em>) shows an immediate surge after 15 April 2023, with battles and explosions/remote violence dominating the opening pattern. The peak month by event count is {pd.to_datetime(peak_month['month']).strftime('%B %Y')}, when ACLED records {fmt_int(peak_month['events'])} events. Fatalities follow a less stable trajectory, peaking in {pd.to_datetime(peak_fatal_month['month']).strftime('%B %Y')} with {fmt_int(peak_fatal_month['fatalities'])} reported deaths.</p>
      <p>The linked Conflict Story Explorer (<em>Figure 2</em>) lets readers test this pattern interactively by changing region, state, event type, and measure. This distinction between events and fatalities is substantively important. Event counts capture the tempo of recorded disorder; fatalities capture reported lethality, which is more volatile and often concentrated in a small number of severe events. In Sudan, the capital fight generates enormous event volume, but later phases of violence in Darfur and the central Nile corridor carry a civilian-security logic that cannot be reduced to the Khartoum front line.</p>
      <p>The event-type mix also matters. Battles account for {event_type_text("Battles")} of events, explosions and remote violence for {event_type_text("Explosions/Remote violence")}, violence against civilians for {event_type_text("Violence against civilians")}, and strategic developments for {event_type_text("Strategic developments")}. Strategic developments are not equivalent to direct violent incidents; ACLED uses this category for politically important non-violent activity such as looting, recruitment, arrests, or territorial transfers, so it is best read as context for later violence rather than as a simple violence count.</p>
    </section>

    <div class="figure-band"><div class="figure-wrap map-focus event-map-wrap">{fig_snippets[1]}<p class="caption"><strong>Figure 3.</strong> Event-level geography of Sudan's civil war. Each point is an ACLED event. Use the legend to isolate event types and hover for event details.</p></div></div>

    <section class="narrative">
      <h2>Finding 2: Civilian Insecurity Fragments Across Regions</h2>
      <p>The event-level point map (<em>Figure 3</em>) and state choropleth (<em>Figure 4</em>) show a movement from the capital toward a wider arch of insecurity. Khartoum remains the largest event cluster, but the highest reported fatality burden is in {html.escape(str(top_admin_fatal['admin1']))}, with {fmt_int(top_admin_fatal['fatalities'])} reported fatalities. Civilian-targeting events are most numerous in {html.escape(str(top_admin_vac['admin1']))}, which records {fmt_int(top_admin_vac['civilian_targeting_events'])} such events. That pattern supports the project's central claim: the war is not only a SAF-RSF battlefield; it is also a dispersed civilian-protection crisis.</p>
      <p>The monthly state animation (<em>Figure 5</em>) makes the timing of this spread easier to inspect. Two spatial shifts stand out. First, Darfur remains a high-lethality theater, especially when North and West Darfur are viewed alongside South and Central Darfur. Second, Al Jazirah and surrounding central/Nile states become major sites of civilian targeting after the conflict expands beyond the capital. This matters politically because violence against civilians often signals territorial control, predation, reprisal, and local governance breakdown rather than conventional battlefield exchange.</p>
    </section>

    <div class="figure-band">
      <div class="figure-stack">
        <div class="figure-wrap">{fig_snippets[2]}<p class="caption"><strong>Figure 4.</strong> State-level distribution of ACLED events, fatalities, civilian targeting, and civilian-targeting share. Dropdown controls switch the state-level map between measures.</p></div>
        <div class="figure-wrap">{fig_snippets[3]}<p class="caption"><strong>Figure 5.</strong> Monthly spread of conflict and civilian targeting by state. The animation aggregates events to state-month bubbles. Bubble size reflects total events; color reflects civilian-targeting events.</p></div>
      </div>
    </div>

    <section class="narrative">
      <h2>Finding 3: Actor Structure Is Bipolar, But Localized</h2>
      <p>The actor network (<em>Figure 6</em>) makes the SAF-RSF axis visible without making it the whole story. Nodes represent actors and edges represent co-involvement in the same ACLED event. SAF and RSF are central because they appear across the largest number of recorded interactions, but civilian nodes, unidentified armed groups, Darfur communal militias, joint forces, police, and local armed movements fill the surrounding structure. This is what fragmented insecurity looks like in event data: a national confrontation generates local actor constellations that vary by state and month.</p>
      <table>
        <thead><tr><th>Actor</th><th>Type</th><th>Weighted degree</th></tr></thead>
        <tbody>{actor_rows}</tbody>
      </table>
      <p>The network should not be read as an alliance map. ACLED actor1-actor2 ties indicate co-presence in recorded events, not durable cooperation. Still, the network is useful because it shows where a simple two-actor war narrative is too thin. Civilian targeting is partly associated with the front line, but also with militia activity, looting/property destruction, abductions, and local armed governance.</p>
    </section>

    <div class="figure-band"><div class="figure-wrap">{fig_snippets[4]}<p class="caption"><strong>Figure 6.</strong> Actor co-involvement network for major conflict actors. The graph displays the strongest actor-pair ties by event count. Larger nodes have higher weighted degree.</p></div></div>

    <section class="narrative">
      <h2>Finding 4: Recent Conflict Intensity Predicts Civilian-Targeting Risk</h2>
      <p>The machine-learning task predicts whether a state-month will experience a high civilian-targeting count in the following month. Here, a state-month means one Sudanese <code>admin1</code> unit in one calendar month, and “high” means above the 75th percentile among labelled state-months. The feature-effects plot (<em>Figure 7</em>) and held-out confusion matrix (<em>Figure 8</em>) summarize how the transparent logistic classifier behaves. The model is trained on lagged conflict features: previous-month events, fatalities, battles, remote violence, SAF and RSF involvement, distinct actor count, three-month rolling conflict totals, and broad macro-region indicators. It is intentionally simple so that its results can be interpreted rather than treated as a black box.</p>
      <div class="method-grid">
        <div><strong>Target:</strong><p>1 if next month's civilian-targeting count is above the 75th percentile of labelled state-months; 0 otherwise.</p></div>
        <div><strong>Validation:</strong><p>Held-out state-months from {html.escape(test_period)}; train rows = {metrics['train_rows']}, test rows = {metrics['test_rows']}.</p></div>
        <div><strong>Performance:</strong><p>Accuracy {metrics['accuracy']:.2f}, precision {metrics['precision']:.2f}, recall {metrics['recall']:.2f}, F1 {metrics['f1']:.2f}, AUC {metrics['auc']:.2f}.</p></div>
        <div><strong>Interpretation:</strong><p>Probabilities are early-warning indicators, not deterministic forecasts; they summarize patterns in reported ACLED events.</p></div>
      </div>
      <p>The model's strongest positive predictors are lagged civilian targeting, recent event concentration, actor diversity, and location spread. In plain terms, areas with a recent mix of fighting, civilian targeting, and multiple actors are more likely to remain high-risk in the following month. This is useful for early warning, but partly reflects conflict autocorrelation: violence is often most likely where violence has recently occurred. The latest model-estimated risk map (<em>Figure 9</em>) is based on {latest_month_label} features and estimates risk for the next month after the export window.</p>
      <table>
        <thead><tr><th>State (admin1)</th><th>Predicted high-risk probability</th><th>Current events</th><th>Current civilian targeting</th></tr></thead>
        <tbody>{high_risk_rows}</tbody>
      </table>
    </section>

    <div class="figure-band">
      <div class="figure-pair">
        <div class="figure-wrap">{fig_snippets[5]}<p class="caption"><strong>Figure 7.</strong> Feature effects in the civilian-targeting risk model. Positive coefficients increase predicted high-risk probability; negative coefficients reduce it.</p></div>
        <div class="figure-wrap">{fig_snippets[6]}<p class="caption"><strong>Figure 8.</strong> Held-out test confusion matrix for the simple logistic model.</p></div>
      </div>
    </div>
    <div class="figure-band"><div class="figure-wrap">{fig_snippets[7]}<p class="caption"><strong>Figure 9.</strong> Model-estimated next-month high civilian-targeting risk by state. The map is generated from the latest month in the available export and should be refreshed with newer ACLED data before policy use.</p></div></div>

    <section class="narrative">
      <h2>Conclusion</h2>
      <p>The evidence supports a three-part answer to the research question. Geographically, the war begins as a capital-centered contest, then widens into Darfur, Kordofan, Al Jazirah, Sennar, White Nile, and other state-level arenas. Politically, the SAF-RSF confrontation remains central but does not exhaust the conflict structure; local militias, civilians, police, armed movements, and unidentified groups shape the lived geography of insecurity. Predictively, recent civilian targeting and broader conflict intensity provide useful warning signals for where civilian-targeting risk may intensify next.</p>
      <p> A main limitation is measurement: ACLED records reported events and estimated fatalities, so access constraints and source coverage can affect comparisons across remote and urban areas. The analysis also does not adjust for population size, displacement, humanitarian access, or local media density, all of which could change how state-level risk should be interpreted. Secondly, the model predicts state-month risk, not individual incidents or causal mechanisms. Its value is as an interpretable early-warning layer to guide closer qualitative and geographic investigation.</p>
      <h3>References</h3>
      <ul class="apa-references">
        <li>Ali, A. M., Birru, J. G., & Eltayeb, N. (2025, April 15). <em>Two years of war in Sudan: How the SAF is gaining the upper hand</em>. ACLED. <a href="https://acleddata.com/report/two-years-war-sudan-how-saf-gaining-upper-hand">https://acleddata.com/report/two-years-war-sudan-how-saf-gaining-upper-hand</a></li>
        <li>Armed Conflict Location & Event Data Project. (2023, November 1). <em>What types of events does ACLED code?</em> ACLED. <a href="https://acleddata.com/faq/what-types-events-does-acled-code">https://acleddata.com/faq/what-types-events-does-acled-code</a></li>
        <li>Armed Conflict Location & Event Data Project. (2024, October 18). <em>ACLED codebook</em>. ACLED. <a href="https://acleddata.com/methodology/acled-codebook">https://acleddata.com/methodology/acled-codebook</a></li>
        <li>Armed Conflict Location & Event Data Project. (2025). <em>ACLED data export: Sudan, 15 April 2023-22 April 2025</em> [Data set]. <a href="https://acleddata.com">https://acleddata.com</a></li>
        <li>Birru, J. G. (2024, December 12). <em>Foreign meddling and fragmentation fuel the war in Sudan</em>. ACLED. <a href="https://acleddata.com/report/foreign-meddling-and-fragmentation-fuel-war-sudan">https://acleddata.com/report/foreign-meddling-and-fragmentation-fuel-war-sudan</a></li>
        <li>Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). Introducing ACLED: An Armed Conflict Location and Event Dataset. <em>Journal of Peace Research, 47</em>(5), 651-660. <a href="https://doi.org/10.1177/0022343310378914">https://doi.org/10.1177/0022343310378914</a></li>
        <li>Runfola, D., Anderson, A., Baier, H., Crittenden, M., Dowker, E., Fuhrig, S., Goodman, S., Grimsley, G., Layko, R., Melville, G., Mulder, M., Oberman, R., Panganiban, J., Peck, A., Seitz, L., Shea, S., Slevin, H., Youngerman, R., & Hobbs, L. (2020). geoBoundaries: A global database of political administrative boundaries. <em>PLOS ONE, 15</em>(4), Article e0231866. <a href="https://doi.org/10.1371/journal.pone.0231866">https://doi.org/10.1371/journal.pone.0231866</a></li>
      </ul>
    </section>
  </main>
  <footer>
    Generated by <code>code/Z_generate_report.py</code>. Report file: <code>docs/index.html</code>.
  </footer>
  <script>
    (function() {{
      let resizeTimer;
      function eventMapHeight(width) {{
        if (width <= 520) return 560;
        if (width <= 820) return 640;
        if (width <= 1100) return 740;
        return 880;
      }}
      function fitEventMap() {{
        const graph = document.querySelector('.event-map-wrap .plotly-graph-div');
        if (!graph || !window.Plotly) return;
        const compact = window.innerWidth <= 620;
        const height = eventMapHeight(window.innerWidth);
        graph.style.height = height + 'px';
        Plotly.relayout(graph, {{
          height: height,
          'legend.orientation': 'h',
          'legend.x': 0.5,
          'legend.xanchor': 'center',
          'legend.y': 0.01,
          'legend.yanchor': 'bottom',
          'yaxis.domain': compact ? [0.12, 1] : [0.08, 1]
        }}).then(() => Plotly.Plots.resize(graph));
      }}
      window.addEventListener('load', () => setTimeout(fitEventMap, 80));
      window.addEventListener('resize', () => {{
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(fitEventMap, 120);
      }});
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    panel = load_panel()
    geojson = download_sudan_admin1_geojson()
    create_hero_image(events, geojson, DOCS_DIR / "assets" / "hero_sudan_conflict.png")
    summaries = build_summary_tables(events, panel, geojson)
    centroids = summaries["admin_summary"][["admin1", "centroid_lat", "centroid_lon"]].drop_duplicates()

    actor_edges, actor_nodes = build_actor_network(events)
    actor_nodes = force_layout(actor_edges, actor_nodes)
    ml = train_evaluate_risk_model(panel)
    write_processed_outputs(summaries, actor_edges, actor_nodes, ml)

    figures = [
        monthly_trend_figure(summaries["monthly_event_type"], summaries["monthly_total"]),
        event_point_map(events, geojson),
        admin_choropleth(summaries["admin_summary"], geojson),
        animated_admin_map(panel, centroids, geojson),
        actor_network_figure(actor_edges, actor_nodes),
        ml_feature_figure(ml["feature_importance"]),
        confusion_matrix_figure(ml["metrics"]),
        ml_risk_map(ml["latest_predictions"], geojson),
    ]
    snippets = figure_html(figures)
    report_html = build_report_html(events, panel, summaries, ml, snippets)
    output_path = DOCS_DIR / "index.html"
    output_path.write_text(report_html, encoding="utf-8")

    summary = {
        "events": int(len(events)),
        "date_min": str(events["event_date"].min().date()),
        "date_max": str(events["event_date"].max().date()),
        "html_report": str(output_path),
        "model_metrics": ml["metrics"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
