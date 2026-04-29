from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
DOCS_DIR = PROJECT_ROOT / "docs"

EVENTS_CLEAN_PATH = PROCESSED_DIR / "acled_sudan_events_clean.csv"
PANEL_PATH = PROCESSED_DIR / "acled_sudan_admin1_month_panel.csv"
GEOJSON_PATH = EXTERNAL_DIR / "sudan_admin1_geoboundaries.geojson"
GEOJSON_SOURCE_PATH = EXTERNAL_DIR / "sudan_admin1_geoboundaries_source.json"

DATE_START = pd.Timestamp("2023-04-15")

EVENT_TYPE_ORDER = [
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
    "Strategic developments",
    "Protests",
    "Riots",
]

EVENT_COLORS = {
    "Battles": "#355C7D",
    "Explosions/Remote violence": "#C06C84",
    "Violence against civilians": "#D95F02",
    "Strategic developments": "#2A9D8F",
    "Protests": "#6A4C93",
    "Riots": "#8D6E63",
}

MACRO_COLORS = {
    "Khartoum": "#355C7D",
    "Darfur": "#D95F02",
    "Kordofan": "#2A9D8F",
    "Central/Nile": "#6A4C93",
    "Eastern Sudan": "#E9C46A",
    "Northern/Nile": "#8D6E63",
    "Abyei": "#7F8C8D",
}

ADMIN1_TO_GEO = {
    "Abyei": "Abyei PCA",
    "Al Jazirah": "Gezira",
}

GEO_TO_ADMIN1 = {value: key for key, value in ADMIN1_TO_GEO.items()}


def ensure_directories() -> None:
    for folder in [DATA_DIR, RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, DOCS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_events(path: Path = EVENTS_CLEAN_PATH) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["event_date", "month"], low_memory=False)
    for col in [
        "is_battle",
        "is_explosion_remote",
        "is_strategic_development",
        "is_violence_against_civilians",
        "is_protest_or_riot",
        "is_civilian_targeting",
        "has_fatalities",
        "rsf_involved",
        "saf_involved",
        "civilian_actor_involved",
    ]:
        if col in events.columns:
            events[col] = events[col].astype(bool)
    events["year_month"] = events["month"].dt.strftime("%Y-%m")
    return events


def load_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["month"], low_memory=False)
    panel["year_month"] = panel["month"].dt.strftime("%Y-%m")
    return panel


def clean_actor_label(actor: str | float | None, max_len: int = 42) -> str:
    if actor is None or pd.isna(actor):
        return "Unknown"
    text = str(actor)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("Military Forces of Sudan (2019-)", "SAF")
    text = text.replace("Rapid Support Forces", "RSF")
    text = text.replace("Sudan People's Liberation Movement", "SPLM")
    text = text.replace("(Sudan)", "")
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "..."
    return text


def classify_actor(actor: str | float | None) -> str:
    if actor is None or pd.isna(actor):
        return "Other"
    text = str(actor).lower()
    if "rapid support forces" in text:
        return "RSF"
    if "military forces of sudan" in text:
        return "SAF"
    if "civilian" in text:
        return "Civilians"
    if "protester" in text or "labor group" in text:
        return "Civil society"
    if "militia" in text or "splm" in text or "joint force" in text or "liberation movement" in text:
        return "Armed group"
    if "police" in text or "government" in text:
        return "State-aligned"
    return "Other"


def admin_to_geo_name(admin1: str) -> str:
    return ADMIN1_TO_GEO.get(admin1, admin1)


def geo_to_admin_name(shape_name: str) -> str:
    return GEO_TO_ADMIN1.get(shape_name, shape_name)


def download_sudan_admin1_geojson(force: bool = False) -> dict:
    """Download a compact Sudan ADM1 GeoJSON from geoBoundaries if absent."""
    ensure_directories()
    if GEOJSON_PATH.exists() and not force:
        return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))

    api_url = "https://www.geoboundaries.org/api/current/gbOpen/SDN/ADM1/"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8"))

    geojson_url = metadata["simplifiedGeometryGeoJSON"]
    with urllib.request.urlopen(geojson_url, timeout=120) as response:
        geojson = json.loads(response.read().decode("utf-8"))

    GEOJSON_PATH.write_text(json.dumps(geojson), encoding="utf-8")
    GEOJSON_SOURCE_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return geojson


def _iter_coordinates(geometry: dict) -> Iterable[tuple[float, float]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        for ring in coords:
            for lon, lat in ring:
                yield float(lon), float(lat)
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                for lon, lat in ring:
                    yield float(lon), float(lat)


def geojson_centroids(geojson: dict) -> pd.DataFrame:
    rows = []
    for feature in geojson["features"]:
        shape_name = feature["properties"]["shapeName"]
        points = list(_iter_coordinates(feature["geometry"]))
        if not points:
            continue
        lon = float(np.mean([p[0] for p in points]))
        lat = float(np.mean([p[1] for p in points]))
        rows.append(
            {
                "geo_admin1": shape_name,
                "admin1": geo_to_admin_name(shape_name),
                "centroid_lon": lon,
                "centroid_lat": lat,
            }
        )
    return pd.DataFrame(rows)


def build_summary_tables(events: pd.DataFrame, panel: pd.DataFrame, geojson: dict) -> dict[str, pd.DataFrame]:
    monthly_event_type = (
        events.groupby(["month", "year_month", "event_type"], as_index=False)
        .agg(events=("event_id_cnty", "count"), fatalities=("fatalities", "sum"))
        .sort_values(["month", "event_type"])
    )

    monthly_total = (
        events.groupby(["month", "year_month"], as_index=False)
        .agg(
            events=("event_id_cnty", "count"),
            fatalities=("fatalities", "sum"),
            civilian_targeting_events=("is_civilian_targeting", "sum"),
            violence_against_civilians_events=("is_violence_against_civilians", "sum"),
            battles=("is_battle", "sum"),
            explosions_remote=("is_explosion_remote", "sum"),
            strategic_developments=("is_strategic_development", "sum"),
        )
        .sort_values("month")
    )

    admin_summary = (
        events.groupby(["admin1", "macro_region"], as_index=False)
        .agg(
            events=("event_id_cnty", "count"),
            fatalities=("fatalities", "sum"),
            civilian_targeting_events=("is_civilian_targeting", "sum"),
            violence_against_civilians_events=("is_violence_against_civilians", "sum"),
            battles=("is_battle", "sum"),
            explosions_remote=("is_explosion_remote", "sum"),
            strategic_developments=("is_strategic_development", "sum"),
            locations=("location", "nunique"),
            admin2_units=("admin2", "nunique"),
            first_event=("event_date", "min"),
            last_event=("event_date", "max"),
        )
        .sort_values("events", ascending=False)
    )
    admin_summary["geo_admin1"] = admin_summary["admin1"].map(admin_to_geo_name)
    admin_summary["civilian_targeting_share"] = np.where(
        admin_summary["events"].gt(0),
        admin_summary["civilian_targeting_events"] / admin_summary["events"],
        0,
    )
    admin_summary["fatalities_per_event"] = np.where(
        admin_summary["events"].gt(0),
        admin_summary["fatalities"] / admin_summary["events"],
        0,
    )

    centroids = geojson_centroids(geojson)
    admin_summary = admin_summary.merge(centroids, on=["admin1", "geo_admin1"], how="left")

    macro_summary = (
        events.groupby(["macro_region", "year_month"], as_index=False)
        .agg(
            events=("event_id_cnty", "count"),
            fatalities=("fatalities", "sum"),
            civilian_targeting_events=("is_civilian_targeting", "sum"),
        )
        .sort_values(["year_month", "macro_region"])
    )

    latest_month = panel["month"].max()
    latest_features = panel.loc[panel["month"].eq(latest_month)].copy()

    return {
        "monthly_event_type": monthly_event_type,
        "monthly_total": monthly_total,
        "admin_summary": admin_summary,
        "macro_summary": macro_summary,
        "latest_features": latest_features,
    }


def build_actor_network(events: pd.DataFrame, top_edges: int = 45) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = events.dropna(subset=["actor1", "actor2"]).copy()
    pairs = pairs.loc[pairs["actor1"].ne(pairs["actor2"])]
    edges = (
        pairs.groupby(["actor1", "actor2"], as_index=False)
        .agg(
            events=("event_id_cnty", "count"),
            fatalities=("fatalities", "sum"),
            civilian_targeting_events=("is_civilian_targeting", "sum"),
            first_event=("event_date", "min"),
            last_event=("event_date", "max"),
        )
        .sort_values(["events", "fatalities"], ascending=False)
        .head(top_edges)
        .reset_index(drop=True)
    )

    degree_rows = []
    for _, row in edges.iterrows():
        degree_rows.append({"actor": row["actor1"], "weighted_degree": row["events"]})
        degree_rows.append({"actor": row["actor2"], "weighted_degree": row["events"]})

    nodes = (
        pd.DataFrame(degree_rows)
        .groupby("actor", as_index=False)
        .agg(weighted_degree=("weighted_degree", "sum"))
        .sort_values("weighted_degree", ascending=False)
        .reset_index(drop=True)
    )
    nodes["actor_short"] = nodes["actor"].map(clean_actor_label)
    nodes["actor_type"] = nodes["actor"].map(classify_actor)
    nodes["node_size"] = 10 + 35 * np.sqrt(nodes["weighted_degree"] / nodes["weighted_degree"].max())
    return edges, nodes


def force_layout(edges: pd.DataFrame, nodes: pd.DataFrame, iterations: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    node_list = nodes["actor"].tolist()
    node_index = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)
    if n == 0:
        return nodes.assign(x=[], y=[])

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    positions += rng.normal(0, 0.08, size=positions.shape)

    weights = []
    edge_indices = []
    for _, row in edges.iterrows():
        if row["actor1"] in node_index and row["actor2"] in node_index:
            edge_indices.append((node_index[row["actor1"]], node_index[row["actor2"]]))
            weights.append(float(row["events"]))
    weights = np.array(weights) if weights else np.array([1.0])
    max_weight = weights.max() if len(weights) else 1.0

    area = 4.0
    k = math.sqrt(area / max(n, 1))
    temperature = 0.15

    for _ in range(iterations):
        disp = np.zeros_like(positions)
        for i in range(n):
            delta = positions[i] - positions
            distance = np.linalg.norm(delta, axis=1) + 1e-6
            force = (k * k / distance)[:, None] * (delta / distance[:, None])
            disp[i] += force.sum(axis=0)

        for edge_number, (i, j) in enumerate(edge_indices):
            delta = positions[i] - positions[j]
            distance = np.linalg.norm(delta) + 1e-6
            edge_weight = 0.35 + 0.65 * (weights[edge_number] / max_weight)
            force = (distance * distance / k) * edge_weight * delta / distance
            disp[i] -= force
            disp[j] += force

        lengths = np.linalg.norm(disp, axis=1) + 1e-6
        positions += (disp / lengths[:, None]) * np.minimum(lengths, temperature)[:, None]
        positions = np.clip(positions, -1.7, 1.7)
        temperature *= 0.985

    laid_out = nodes.copy()
    laid_out["x"] = positions[:, 0]
    laid_out["y"] = positions[:, 1]
    return laid_out


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class LogisticModel:
    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    threshold: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_names].fillna(0).to_numpy(dtype=float)
        x_scaled = (x - self.means) / self.stds
        x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        return sigmoid(x_design @ self.weights)

    def coefficients(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"feature": self.feature_names, "coefficient": self.weights[1:]})
            .assign(abs_coefficient=lambda d: d["coefficient"].abs())
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )


def _make_model_frame(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    frame = panel.copy()
    frame["target"] = frame["target_high_civilian_targeting_next_month"]

    numeric_features = [
        "months_since_war_start",
        "events_lag1",
        "fatalities_lag1",
        "civilian_targeting_events_lag1",
        "violence_against_civilians_events_lag1",
        "battles_lag1",
        "explosions_remote_lag1",
        "strategic_developments_lag1",
        "protests_riots_lag1",
        "saf_events_lag1",
        "rsf_events_lag1",
        "civilian_actor_events_lag1",
        "fatal_event_count_lag1",
        "distinct_actor_count_lag1",
        "unique_locations_lag1",
        "unique_admin2_lag1",
        "events_prev3mo",
        "fatalities_prev3mo",
        "civilian_targeting_events_prev3mo",
        "battles_prev3mo",
        "explosions_remote_prev3mo",
    ]

    macro_dummies = pd.get_dummies(frame["macro_region"], prefix="macro", dtype=float)
    frame = pd.concat([frame, macro_dummies], axis=1)
    feature_names = numeric_features + macro_dummies.columns.tolist()
    for col in feature_names:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return frame, feature_names, "target"


def fit_logistic_model(
    train_frame: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    learning_rate: float = 0.04,
    epochs: int = 6500,
    l2: float = 0.05,
) -> LogisticModel:
    x = train_frame[feature_names].fillna(0).to_numpy(dtype=float)
    y = train_frame[target_col].to_numpy(dtype=float)

    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    x_scaled = (x - means) / stds
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])

    positive = y.sum()
    negative = len(y) - positive
    pos_weight = negative / positive if positive else 1.0
    sample_weight = np.where(y == 1, pos_weight, 1.0)
    sample_weight = sample_weight / sample_weight.mean()

    weights = np.zeros(x_design.shape[1])
    for _ in range(epochs):
        probabilities = sigmoid(x_design @ weights)
        error = (probabilities - y) * sample_weight
        gradient = (x_design.T @ error) / len(y)
        gradient[1:] += l2 * weights[1:] / len(y)
        weights -= learning_rate * gradient

    provisional = LogisticModel(feature_names, means, stds, weights, threshold=0.5)
    train_prob = provisional.predict_proba(train_frame)
    threshold = choose_threshold(y, train_prob)
    return LogisticModel(feature_names, means, stds, weights, threshold=threshold)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.20, 0.80, 61):
        metrics = evaluate_predictions(y_true, probabilities, threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


def roc_auc_score_manual(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    positives = probabilities[y_true == 1]
    negatives = probabilities[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = 0.5 * (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + ties)


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    predictions = (probabilities >= threshold).astype(int)
    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())
    accuracy = (tp + tn) / len(y_true) if len(y_true) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": roc_auc_score_manual(y_true, probabilities),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def train_evaluate_risk_model(panel: pd.DataFrame) -> dict[str, object]:
    frame, feature_names, target_col = _make_model_frame(panel)
    labelled = frame.dropna(subset=[target_col]).copy()
    labelled[target_col] = labelled[target_col].astype(int)

    max_label_month = labelled["month"].max()
    test_start = max_label_month - pd.DateOffset(months=4)
    train_frame = labelled.loc[labelled["month"].lt(test_start)].copy()
    test_frame = labelled.loc[labelled["month"].ge(test_start)].copy()

    model = fit_logistic_model(train_frame, feature_names, target_col)
    test_prob = model.predict_proba(test_frame)
    metrics = evaluate_predictions(test_frame[target_col].to_numpy(), test_prob, model.threshold)
    metrics.update(
        {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "test_start": str(test_start.date()),
            "test_end": str(max_label_month.date()),
            "positive_rate_train": float(train_frame[target_col].mean()),
            "positive_rate_test": float(test_frame[target_col].mean()),
        }
    )

    final_model = fit_logistic_model(labelled, feature_names, target_col)
    current_month = frame["month"].max()
    latest = frame.loc[frame["month"].eq(current_month)].copy()
    latest["predicted_high_risk_probability"] = final_model.predict_proba(latest)
    latest["predicted_high_risk"] = (
        latest["predicted_high_risk_probability"] >= final_model.threshold
    ).astype(int)
    latest = latest.sort_values("predicted_high_risk_probability", ascending=False)

    return {
        "model": model,
        "final_model": final_model,
        "metrics": metrics,
        "test_frame": test_frame.assign(
            predicted_probability=test_prob,
            predicted_high_risk=(test_prob >= model.threshold).astype(int),
        ),
        "latest_predictions": latest,
        "feature_importance": final_model.coefficients(),
    }


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_directories()
    df.to_csv(path, index=False)
