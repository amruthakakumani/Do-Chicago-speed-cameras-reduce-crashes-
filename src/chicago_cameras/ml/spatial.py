"""Real spatial features from US Census TIGER/Line roads.

Road density, road classification, distance to major roads and intersection
density are computed from the 2024 TIGER/Line road shapefiles for Cook and
DuPage counties.  TIGER is public-domain, versioned, and reproducible; the raw
archives and SHA-256 checksums are preserved in ``data/raw/tiger/``.

No future crash information is used: all road features are static attributes of
the built environment.  A current AADT snapshot is deliberately **not** used
because it would leak future traffic exposure into historical training months.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapefile
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

LOGGER = logging.getLogger(__name__)

TIGER_COUNTIES = {"17031": "Cook", "17043": "DuPage"}
TIGER_VINTAGE = "2024"
TIGER_LAYER = "EDGES"  # topology edges carry TNIDF/TNIDT, so intersections are unambiguous
MAJOR_CLASSES = {"S1100", "S1200"}  # primary and secondary roads
LOCAL_CLASSES = {"S1400"}            # local neighborhood roads

ROAD_FEATURES = [
    "road_density_500m_km_per_km2",
    "major_road_density_500m_km_per_km2",
    "local_road_density_500m_km_per_km2",
    "road_length_500m_km",
    "major_road_share_500m",
    "n_road_segments_500m",
    "road_density_1000m_km_per_km2",
    "major_road_density_1000m_km_per_km2",
    "intersection_density_500m_per_km2",
    "n_intersections_500m",
    "intersection_density_1000m_per_km2",
    "dist_major_road_m",
    "dist_major_intersection_m",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_tiger(cache_dir: Path, refresh: bool = False) -> dict:
    """Download and unzip TIGER road archives with provenance."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = cache_dir / "provenance.json"
    existing = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    records = existing.get("counties", {})
    changed = False

    for county, name in TIGER_COUNTIES.items():
        zip_path = cache_dir / f"tl_{TIGER_VINTAGE}_{county}_edges.zip"
        extract_dir = cache_dir / f"tl_{TIGER_VINTAGE}_{county}_edges"
        url = f"https://www2.census.gov/geo/tiger/TIGER{TIGER_VINTAGE}/EDGES/tl_{TIGER_VINTAGE}_{county}_edges.zip"
        if zip_path.exists() and extract_dir.exists() and not refresh:
            continue
        LOGGER.info("downloading TIGER roads for %s", name)
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        records[county] = {
            "county_name": name,
            "url": url,
            "zip": str(zip_path),
            "sha256": _sha256(zip_path),
            "vintage": TIGER_VINTAGE,
            "license": "Public domain (U.S. Census Bureau)",
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
        }
        changed = True

    if not changed:
        for county, county_name in TIGER_COUNTIES.items():
            zip_path = cache_dir / f"tl_{TIGER_VINTAGE}_{county}_edges.zip"
            if zip_path.exists() and county not in records:
                records[county] = {
                    "county_name": county_name,
                    "url": f"https://www2.census.gov/geo/tiger/TIGER{TIGER_VINTAGE}/EDGES/tl_{TIGER_VINTAGE}_{county}_edges.zip",
                    "zip": str(zip_path),
                    "sha256": _sha256(zip_path),
                    "vintage": TIGER_VINTAGE,
                    "license": "Public domain (U.S. Census Bureau)",
                }
                changed = True

    if changed or not provenance_path.exists() or existing.get("source", "").split(",")[-1].strip() != "EDGES":
        provenance = {
            "source": "U.S. Census Bureau TIGER/Line Shapefiles, EDGES",
            "vintage": TIGER_VINTAGE,
            "counties": records,
            "note": (
                "Static built-environment features. Current Chicago AADT is not used "
                "because it would leak future exposure into historical training months."
            ),
        }
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return json.loads(provenance_path.read_text())


def _load_lines(
    cache_dir: Path,
    bounds: tuple[float, float, float, float],
    epsg: int,
) -> tuple[list[LineString], list[str], list[tuple[int, int]]]:
    """Read road shapes, project to metres and restrict to the city bounds."""
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    xmin, ymin, xmax, ymax = bounds
    # Inverse-project a 2 km margin for the shapefile prefilter.
    inverse = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon_min, lat_min = inverse.transform(xmin - 2000, ymin - 2000)
    lon_max, lat_max = inverse.transform(xmax + 2000, ymax + 2000)

    lines: list[LineString] = []
    classes: list[str] = []
    node_ids: list[tuple[int, int]] = []
    for county in TIGER_COUNTIES:
        base = cache_dir / f"tl_{TIGER_VINTAGE}_{county}_edges" / f"tl_{TIGER_VINTAGE}_{county}_edges"
        if not Path(str(base) + ".shp").exists():
            raise FileNotFoundError(
                f"TIGER edges missing for county {county}; run acquire_tiger() first"
            )
        reader = shapefile.Reader(str(base))
        fields = [field[0] for field in reader.fields[1:]]
        mtfcc_index = fields.index("MTFCC")
        road_index = fields.index("ROADFLG")
        from_index = fields.index("TNIDF")
        to_index = fields.index("TNIDT")
        for shape_record in reader.iterShapeRecords():
            record = shape_record.record
            mtfcc = record[mtfcc_index]
            if record[road_index] != "Y":
                continue
            if mtfcc not in MAJOR_CLASSES and mtfcc not in LOCAL_CLASSES:
                continue
            points = shape_record.shape.points
            parts = list(shape_record.shape.parts) + [len(points)]
            for start, end in itertools.pairwise(parts):
                part = points[start:end]
                if len(part) < 2:
                    continue
                lon = [p[0] for p in part]
                lat = [p[1] for p in part]
                if max(lon) < lon_min or min(lon) > lon_max or max(lat) < lat_min or min(lat) > lat_max:
                    continue
                x, y = transformer.transform(lon, lat)
                lines.append(LineString(np.column_stack([x, y])))
                classes.append(mtfcc)
                node_ids.append((int(record[from_index]), int(record[to_index])))
    return lines, classes, node_ids


def _build_intersections(
    lines: list[LineString],
    classes: list[str],
    node_ids: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive junctions from TIGER topology node ids.

    Each edge carries a from-node and a to-node.  A node incident to three or
    more road edges is a junction; a node with one or two edges is an endpoint or
    a mid-block continuation.  Major junctions have at least one incident
    primary/secondary edge.
    """
    nodes: dict[int, dict] = {}
    for line, mtfcc, (from_id, to_id) in zip(lines, classes, node_ids):
        for node_id, coordinate in ((from_id, line.coords[0]), (to_id, line.coords[-1])):
            entry = nodes.setdefault(
                node_id,
                {"x": coordinate[0], "y": coordinate[1], "degree": 0, "major": False},
            )
            entry["degree"] += 1
            if mtfcc in MAJOR_CLASSES:
                entry["major"] = True

    points: list[tuple[float, float]] = []
    is_major: list[bool] = []
    for entry in nodes.values():
        if entry["degree"] < 3:
            continue
        points.append((entry["x"], entry["y"]))
        is_major.append(bool(entry["major"]))
    return np.asarray(points), np.asarray(is_major, dtype=bool)


@dataclass
class RoadFeatureResult:
    features: pd.DataFrame
    metadata: dict


def build_road_features(
    locations: pd.DataFrame,
    cache_dir: Path,
    epsg: int = 26916,
    refresh: bool = False,
) -> RoadFeatureResult:
    """Compute static road features for every location.

    ``locations`` must contain ``location_id``, ``x`` and ``y`` in the target
    metric CRS.  The computation is cached to ``data/processed`` by the caller.
    """
    from scipy.spatial import cKDTree

    acquire_tiger(cache_dir, refresh=refresh)
    bounds = (
        float(locations["x"].min()),
        float(locations["y"].min()),
        float(locations["x"].max()),
        float(locations["y"].max()),
    )
    lines, classes, node_ids = _load_lines(cache_dir, bounds, epsg)
    if not lines:
        raise RuntimeError("no TIGER road lines found inside the city bounds")

    classes_array = np.asarray(classes)
    major_mask = np.isin(classes_array, list(MAJOR_CLASSES))
    tree = STRtree(lines)
    major_tree = STRtree([line for line, major in zip(lines, major_mask) if major])
    major_lines = [line for line, major in zip(lines, major_mask) if major]

    intersection_points, intersection_major = _build_intersections(lines, classes, node_ids)
    intersection_tree = cKDTree(intersection_points) if len(intersection_points) else None
    major_intersection_tree = (
        cKDTree(intersection_points[intersection_major])
        if len(intersection_points) and intersection_major.any()
        else None
    )

    rows: list[dict] = []
    for location in locations.itertuples():
        point = Point(float(location.x), float(location.y))
        record = {"location_id": location.location_id}
        for radius, suffix in ((500.0, "500m"), (1000.0, "1000m")):
            buffer = point.buffer(radius)
            area_km2 = np.pi * (radius / 1000.0) ** 2
            indices = tree.query(buffer)
            length_major = 0.0
            length_local = 0.0
            n_segments = 0
            for index in indices:
                clipped = lines[int(index)].intersection(buffer)
                if clipped.is_empty:
                    continue
                length = float(clipped.length)
                n_segments += 1
                if classes_array[int(index)] in MAJOR_CLASSES:
                    length_major += length
                else:
                    length_local += length
            total = length_major + length_local
            record[f"road_density_{suffix}_km_per_km2"] = (total / 1000.0) / area_km2
            if suffix == "500m":
                record["major_road_density_500m_km_per_km2"] = (length_major / 1000.0) / area_km2
                record["local_road_density_500m_km_per_km2"] = (length_local / 1000.0) / area_km2
                record["road_length_500m_km"] = total / 1000.0
                record["major_road_share_500m"] = length_major / total if total > 0 else 0.0
                record["n_road_segments_500m"] = n_segments
            else:
                record["major_road_density_1000m_km_per_km2"] = (length_major / 1000.0) / area_km2

            if intersection_tree is not None:
                n_intersections = len(intersection_tree.query_ball_point([point.x, point.y], radius))
            else:
                n_intersections = 0
            record[f"n_intersections_{suffix}"] = n_intersections
            record[f"intersection_density_{suffix}_per_km2"] = n_intersections / area_km2

        if major_lines:
            nearest = int(major_tree.nearest(point))
            record["dist_major_road_m"] = float(major_lines[nearest].distance(point))
        else:
            record["dist_major_road_m"] = np.nan
        if major_intersection_tree is not None:
            distance, _ = major_intersection_tree.query([point.x, point.y])
            record["dist_major_intersection_m"] = float(distance)
        else:
            record["dist_major_intersection_m"] = np.nan
        rows.append(record)

    frame = pd.DataFrame(rows)
    metadata = {
        "source": "US Census TIGER/Line 2024 EDGES",
        "counties": list(TIGER_COUNTIES.values()),
        "vintage": TIGER_VINTAGE,
        "n_road_lines": len(lines),
        "n_intersections": len(intersection_points),
        "n_major_intersections": int(intersection_major.sum()),
        "feature_columns": ROAD_FEATURES,
    }
    return RoadFeatureResult(features=frame, metadata=metadata)
