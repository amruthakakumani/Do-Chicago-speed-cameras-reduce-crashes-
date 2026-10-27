"""Road-segment geography for a prediction-oriented Chicago crash task.

The unit is a TIGER/Line 2024 EDGES road segment clipped to the City of Chicago
community-area union.  Crashes are assigned to the nearest segment within a
documented distance, with assignment distance and ambiguity recorded.  The full
network counts are retained; a fixed, reproducible stratified sample is used for
modeling because a full 73k x 108 segment-month panel does not fit comfortably
on an 8 GB machine.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer
from shapely import contains_xy, wkt
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree

LOGGER = logging.getLogger(__name__)

ROAD_CLASSES = {"S1100": "primary", "S1200": "secondary", "S1400": "local"}
MAJOR_CLASSES = {"S1100", "S1200"}


@dataclass(frozen=True)
class RoadConfig:
    tiger_dir: Path = Path("data/raw/tiger")
    boundary_path: Path = Path("data/raw/boundaries/community_areas.geojson")
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results/bottom_up/roads")
    epsg: int = 26916
    min_length_m: float = 20.0
    assignment_radius_m: float = 30.0
    ambiguity_margin_m: float = 5.0
    low_confidence_m: float = 20.0
    local_sample_size: int = 12_000
    seed: int = 20261024
    target_start: str = "2018-01-01"
    target_end: str = "2026-08-01"
    test_start: str = "2025-09-01"


def load_city_boundary(path: Path) -> Any:
    payload = json.loads(path.read_text())
    return unary_union([shape(feature["geometry"]) for feature in payload["features"]])


def _project_geometry(geometry, transformer: Transformer):
    return shp_transform(transformer.transform, geometry)


def extract_road_segments(config: RoadConfig | None = None) -> pd.DataFrame:
    config = config or RoadConfig()
    """Clip 2024 TIGER road edges to Chicago and persist stable TLID segments."""
    city = load_city_boundary(config.boundary_path)
    transformer = Transformer.from_crs(4326, config.epsg, always_xy=True)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for county in ("17031", "17043"):
        base = config.tiger_dir / f"tl_2024_{county}_edges" / f"tl_2024_{county}_edges"
        reader = shapefile.Reader(str(base))
        fields = [field[0] for field in reader.fields[1:]]
        index = {name: position for position, name in enumerate(fields)}
        for shape_record in reader.iterShapeRecords():
            record = shape_record.record
            if record[index["ROADFLG"]] != "Y":
                continue
            mtfcc = record[index["MTFCC"]]
            if mtfcc not in ROAD_CLASSES:
                continue
            try:
                geometry = shape(shape_record.shape.__geo_interface__)
            except Exception as exc:  # noqa: BLE001 - malformed TIGER geometry
                LOGGER.warning("skipping malformed TIGER geometry: %s", exc)
                continue
            if not geometry.intersects(city):
                continue
            clipped = geometry.intersection(city)
            if clipped.is_empty:
                continue
            metric = _project_geometry(clipped, transformer)
            if metric.length < config.min_length_m:
                continue
            centroid = metric.centroid
            rows.append(
                {
                    "segment_id": str(record[index["TLID"]]),
                    "county_fips": county,
                    "mtfcc": mtfcc,
                    "road_class": ROAD_CLASSES[mtfcc],
                    "major_road": int(mtfcc in MAJOR_CLASSES),
                    "name": (record[index["FULLNAME"]] or "").strip(),
                    "from_node": int(record[index["TNIDF"]]),
                    "to_node": int(record[index["TNIDT"]]),
                    "length_m": float(metric.length),
                    "centroid_x": float(centroid.x),
                    "centroid_y": float(centroid.y),
                    "geometry_wkt": metric.wkt,
                }
            )
            if len(rows) % 10_000 == 0:
                LOGGER.info("kept %s road segments in %.0fs", len(rows), time.time() - started)
    frame = pd.DataFrame(rows)
    if frame["segment_id"].duplicated().any():
        raise ValueError("TIGER segment IDs are not unique")
    node_degree: dict[int, int] = {}
    for frame_row in frame.itertuples():
        node_degree[int(frame_row.from_node)] = node_degree.get(int(frame_row.from_node), 0) + 1
        node_degree[int(frame_row.to_node)] = node_degree.get(int(frame_row.to_node), 0) + 1
    frame["from_degree"] = frame["from_node"].map(node_degree).astype(int)
    frame["to_degree"] = frame["to_node"].map(node_degree).astype(int)
    frame["node_degree_max"] = frame[["from_degree", "to_degree"]].max(axis=1)
    frame["is_intersection_edge"] = (frame["node_degree_max"] >= 3).astype(int)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.processed_dir / "road_segments.parquet", index=False)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "US Census TIGER/Line 2024 EDGES, Cook and DuPage counties",
        "boundary": "Union of City of Chicago community areas (igwz-8jzy)",
        "n_segments": len(frame),
        "n_major": int(frame["major_road"].sum()),
        "n_intersection_edges": int(frame["is_intersection_edge"].sum()),
        "length_m": {
            "total_km": float(frame["length_m"].sum() / 1000),
            "median_m": float(frame["length_m"].median()),
        },
        "county_counts": frame["county_fips"].value_counts().to_dict(),
        "class_counts": frame["road_class"].value_counts().to_dict(),
        "note": (
            "2024 geometry is static; historical TIGER 2017 comparison is reported separately. "
            "Segment IDs are TIGER TLIDs."
        ),
    }
    (config.results_dir / "segment_inventory.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return frame


def _metric_crash_points(crashes: pd.DataFrame, config: RoadConfig | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    config = config or RoadConfig()
    city = load_city_boundary(config.boundary_path)
    in_city = contains_xy(city, crashes["lon"].to_numpy(float), crashes["lat"].to_numpy(float))
    work = crashes.loc[
        crashes["month"].between("2017-09-01", "2026-08-01")
        & crashes["x"].notna()
        & crashes["y"].notna()
        & in_city
    ].copy()
    points = work[["x", "y"]].to_numpy(float)
    return points, work


def assign_crashes_to_segments(
    segments: pd.DataFrame,
    crashes: pd.DataFrame,
    config: RoadConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or RoadConfig()
    """Assign each crash once to the nearest road segment within a radius.

    Candidates are all segments intersecting a 30 m buffer.  The nearest segment
    is selected; ties are broken by the lexicographically smallest TLID.  A crash
    is flagged ambiguous when a second candidate is within 5 m of the winner.
    """
    from shapely import points as shapely_points

    points, work = _metric_crash_points(crashes, config)
    if len(work) == 0:
        raise ValueError("no crashes in the road-analysis window")
    geometries = [wkt.loads(value) for value in segments["geometry_wkt"]]
    tree = STRtree(geometries)
    ids = segments["segment_id"].to_numpy()
    assigned = np.full(len(work), None, dtype=object)
    distance = np.full(len(work), np.nan)
    second_distance = np.full(len(work), np.nan)
    n_candidates = np.zeros(len(work), dtype=int)
    ambiguous = np.zeros(len(work), dtype=bool)
    low_confidence = np.zeros(len(work), dtype=bool)
    started = time.time()
    point_geoms = shapely_points(points)
    for position, point in enumerate(point_geoms):
        candidates = tree.query(point.buffer(config.assignment_radius_m), predicate="intersects")
        n_candidates[position] = len(candidates)
        if len(candidates) == 0:
            continue
        distances = np.array([geometries[int(index)].distance(point) for index in candidates])
        keep = distances <= config.assignment_radius_m
        candidates = candidates[keep]
        distances = distances[keep]
        n_candidates[position] = len(candidates)
        if len(candidates) == 0:
            continue
        order = np.argsort(distances)
        best = int(order[0])
        assigned[position] = ids[int(candidates[best])]
        distance[position] = float(distances[best])
        if len(order) > 1:
            second_distance[position] = float(distances[order[1]])
            ambiguous[position] = bool(
                distances[order[1]] - distances[order[0]] <= config.ambiguity_margin_m
            )
        low_confidence[position] = bool(distances[best] > config.low_confidence_m)
        if (position + 1) % 100_000 == 0:
            LOGGER.info("assigned %s / %s crashes in %.0fs", position + 1, len(work), time.time() - started)
    result = pd.DataFrame(
        {
            "crash_record_id": work["crash_record_id"].to_numpy(),
            "month": work["month"].to_numpy(),
            "segment_id": assigned,
            "assignment_distance_m": distance,
            "second_distance_m": second_distance,
            "n_candidate_segments": n_candidates,
            "ambiguous_assignment": ambiguous,
            "low_confidence_assignment": low_confidence,
        }
    )
    result["assigned"] = result["segment_id"].notna()
    summary = {
        "crashes_in_window": len(result),
        "assigned": int(result["assigned"].sum()),
        "unassigned": int((~result["assigned"]).sum()),
        "assignment_rate": float(result["assigned"].mean()),
        "ambiguous_assignments": int(result["ambiguous_assignment"].sum()),
        "low_confidence_assignments": int(result["low_confidence_assignment"].sum()),
        "distance_m": {
            "median": float(result["assignment_distance_m"].median()),
            "p95": float(result["assignment_distance_m"].quantile(0.95)),
            "max": float(result["assignment_distance_m"].max()),
        },
        "n_candidate_segments": {
            "median": float(result["n_candidate_segments"].median()),
            "p95": float(result["n_candidate_segments"].quantile(0.95)),
            "max": int(result["n_candidate_segments"].max()),
        },
        "excluded": int(len(crashes) - len(result)),
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(config.processed_dir / "crash_road_assignments.parquet", index=False)
    (config.results_dir / "assignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return result, summary


def segment_month_counts(
    segments: pd.DataFrame,
    assignments: pd.DataFrame,
    config: RoadConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or RoadConfig()
    """Build full-network counts, a model sample, and segment-level summaries."""
    assigned = assignments.loc[assignments["assigned"]].copy()
    assigned["segment_id"] = assigned["segment_id"].astype(str)
    assigned["month"] = pd.to_datetime(assigned["month"])
    nonzero = (
        assigned.groupby(["segment_id", "month"], as_index=False)
        .size()
        .rename(columns={"size": "y_all"})
    )
    months = pd.date_range(
        pd.Timestamp("2017-09-01"), pd.Timestamp(config.target_end), freq="MS"
    )
    segment_summary = (
        assigned.groupby("segment_id", as_index=False)
        .agg(total_crashes=("crash_record_id", "size"), active_months=("month", "nunique"))
        .merge(
            segments[["segment_id", "major_road", "road_class", "length_m", "centroid_x", "centroid_y"]],
            on="segment_id",
            how="right",
        )
        .fillna({"total_crashes": 0, "active_months": 0})
    )
    segment_summary["mean_monthly_crashes"] = (
        segment_summary["total_crashes"] / len(months)
    )

    # Fixed sample: every major segment plus a seeded random sample of local
    # segments.  The sample is drawn from all local segments, not only those that
    # ever crashed, so zero-crash road segments are retained.
    local = segment_summary.loc[segment_summary["major_road"].eq(0)].sort_values("segment_id")
    rng = np.random.default_rng(config.seed)
    n_local = min(config.local_sample_size, len(local))
    chosen_local = set(rng.choice(local["segment_id"].to_numpy(), size=n_local, replace=False))
    selected_ids = set(segment_summary.loc[segment_summary["major_road"].eq(1), "segment_id"]) | chosen_local
    selected = segment_summary.loc[segment_summary["segment_id"].isin(selected_ids)].copy()
    selected = selected.merge(
        segments[
            [
                "segment_id",
                "name",
                "mtfcc",
                "from_degree",
                "to_degree",
                "node_degree_max",
                "is_intersection_edge",
                "geometry_wkt",
            ]
        ],
        on="segment_id",
        how="left",
    )
    counts = nonzero.merge(selected[["segment_id"]], on="segment_id", how="inner")
    panel = (
        pd.MultiIndex.from_product(
            [selected["segment_id"].to_numpy(), months.to_numpy()],
            names=["segment_id", "month"],
        )
        .to_frame(index=False)
        .merge(counts, on=["segment_id", "month"], how="left")
    )
    panel["y_all"] = panel["y_all"].fillna(0).astype(int)
    panel = panel.merge(
        selected[
            [
                "segment_id",
                "major_road",
                "road_class",
                "length_m",
                "centroid_x",
                "centroid_y",
                "name",
                "from_degree",
                "to_degree",
                "node_degree_max",
                "is_intersection_edge",
            ]
        ],
        on="segment_id",
        how="left",
    )
    panel["month"] = pd.to_datetime(panel["month"])
    panel["split"] = np.where(
        panel["month"] < pd.Timestamp(config.test_start), "train", "test"
    )
    selected_summary = {
        "n_segments_total": len(segment_summary),
        "n_segments_sampled": int(selected["segment_id"].nunique()),
        "n_major_sampled": int(selected["major_road"].sum()),
        "n_local_sampled": int((selected["major_road"] == 0).sum()),
        "n_local_total": len(local),
        "sample_fraction_all_segments": float(selected["segment_id"].nunique() / len(segment_summary)),
        "sample_fraction_local_segments": float(n_local / len(local)),
        "sample_total_crashes": int(selected["total_crashes"].sum()),
        "all_total_crashes": int(segment_summary["total_crashes"].sum()),
        "sample_crash_share": float(
            selected["total_crashes"].sum() / max(segment_summary["total_crashes"].sum(), 1)
        ),
        "zero_crash_segments_in_sample": int((selected["total_crashes"] == 0).sum()),
        "months": {"first": str(months[0].date()), "last": str(months[-1].date()), "n": len(months)},
        "note": (
            "All major segments plus a seeded random sample of local segments. "
            "The sample is not a citywide segment census; raw MAE is not comparable "
            "to the 250 m camera-zone task."
        ),
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    segment_summary.to_csv(config.results_dir / "segment_summary.csv", index=False)
    selected[["segment_id", "major_road", "road_class", "length_m", "total_crashes", "active_months"]].to_csv(
        config.results_dir / "model_sample_segments.csv", index=False
    )
    nonzero.to_parquet(config.processed_dir / "road_segment_month_counts.parquet", index=False)
    panel.to_parquet(config.processed_dir / "road_model_panel.parquet", index=False)
    (config.results_dir / "sample_summary.json").write_text(
        json.dumps(selected_summary, indent=2, sort_keys=True) + "\n"
    )
    return segment_summary, selected, panel


def add_road_features(panel: pd.DataFrame) -> pd.DataFrame:
    """History/calendar/static features for the road-segment task."""
    frame = panel.sort_values(["segment_id", "month"]).copy()
    group = frame.groupby("segment_id", sort=False)
    for lag in (1, 2, 3, 6, 12):
        frame[f"lag_{lag}"] = group["y_all"].shift(lag)
    for window, min_periods in ((3, 2), (6, 4), (12, 8)):
        frame[f"roll_mean_{window}"] = group["y_all"].transform(
            lambda series, w=window, m=min_periods: series.shift(1).rolling(w, min_periods=m).mean()
        )
        frame[f"roll_std_{window}"] = group["y_all"].transform(
            lambda series, w=window, m=min_periods: series.shift(1).rolling(w, min_periods=m).std()
        )
    frame["expanding_mean"] = group["y_all"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )
    frame["zero_share_12"] = group["y_all"].transform(
        lambda series: series.eq(0).astype(float).shift(1).rolling(12, min_periods=6).mean()
    )
    frame["month_index"] = (
        (frame["month"].dt.year - 2017) * 12 + frame["month"].dt.month
    )
    frame["year"] = frame["month"].dt.year
    frame["month_of_year"] = frame["month"].dt.month
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month_of_year"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month_of_year"] - 1) / 12)
    frame["days_in_month"] = frame["month"].dt.days_in_month
    frame["is_covid"] = frame["month"].between("2020-03-01", "2021-12-01").astype(int)
    frame["major_road"] = frame["major_road"].astype(int)
    frame["is_intersection_edge"] = frame["is_intersection_edge"].astype(int)
    return frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = RoadConfig()
    segments = extract_road_segments(cfg)
    crashes = pd.read_parquet(cfg.processed_dir / "crashes.parquet")
    assignments, summary = assign_crashes_to_segments(segments, crashes, cfg)
    segment_summary, selected, panel = segment_month_counts(segments, assignments, cfg)
    panel = add_road_features(panel)
    panel.to_parquet(cfg.processed_dir / "road_model_panel_features.parquet", index=False)
    print(json.dumps(summary, indent=2))
    print(panel.shape)
