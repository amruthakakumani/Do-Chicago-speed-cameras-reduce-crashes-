"""Road assignment and segment-panel tests with synthetic geometry."""

from __future__ import annotations

import json

import pandas as pd

from chicago_cameras.road_network import RoadConfig, assign_crashes_to_segments, segment_month_counts


def test_road_assignment_is_unique_and_tracks_ambiguity(tmp_path):
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
                        },
                    }
                ],
            }
        )
    )
    segments = pd.DataFrame(
        [
            {
                "segment_id": "a",
                "county_fips": "17031",
                "mtfcc": "S1200",
                "road_class": "secondary",
                "major_road": 1,
                "name": "A",
                "from_node": 1,
                "to_node": 2,
                "length_m": 20.0,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "geometry_wkt": "LINESTRING (-10 0, 10 0)",
                "node_degree_max": 3,
                "is_intersection_edge": 1,
                "from_degree": 3,
                "to_degree": 1,
            },
            {
                "segment_id": "b",
                "county_fips": "17031",
                "mtfcc": "S1400",
                "road_class": "local",
                "major_road": 0,
                "name": "B",
                "from_node": 3,
                "to_node": 4,
                "length_m": 20.0,
                "centroid_x": 0.0,
                "centroid_y": 5.0,
                "geometry_wkt": "LINESTRING (-10 5, 10 5)",
                "node_degree_max": 2,
                "is_intersection_edge": 0,
                "from_degree": 2,
                "to_degree": 1,
            },
        ]
    )
    crashes = pd.DataFrame(
        {
            "crash_record_id": ["c1", "c2"],
            "month": pd.to_datetime(["2018-01-01", "2018-01-01"]),
            "x": [0.0, 0.0],
            "y": [10.0, 1.0],
            "lon": [0.0, 0.0],
            "lat": [0.0, 0.0],
        }
    )
    config = RoadConfig(
        boundary_path=boundary,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        local_sample_size=1,
    )
    assignments, summary = assign_crashes_to_segments(segments, crashes, config)
    assert len(assignments) == 2
    assert assignments["crash_record_id"].is_unique
    assert assignments["assigned"].all()
    assert (assignments["assignment_distance_m"] <= 30).all()
    assert int(assignments["ambiguous_assignment"].sum()) == 2
    assert summary["assigned"] == 2
    _, _, panel = segment_month_counts(segments, assignments, config)
    assert panel["y_all"].sum() == 2
    assert len(panel) == 2 * 108
    # Both segments are in the sample: one major and one local.
    assert panel["segment_id"].nunique() == 2
