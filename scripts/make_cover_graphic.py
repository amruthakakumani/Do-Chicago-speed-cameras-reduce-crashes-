#!/usr/bin/env python
"""Generate a subtle, data-driven Chicago street-grid mark for the cover.

The mark is intentionally abstract: no labels, axes or crash symbols.  It is a
crop of the TIGER road network used in the research, so the cover visual is
grounded in the study rather than being decorative stock art.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "cover_street_grid.png"
SEGMENTS = ROOT / "data" / "processed" / "road_segments.parquet"


def main() -> int:
    if SEGMENTS.exists():
        segments = pd.read_parquet(SEGMENTS)
        segments = segments.loc[
            segments["centroid_x"].between(437_000, 449_000)
            & segments["centroid_y"].between(4_635_000, 4_645_000)
        ].head(4_000)
        local: list[list[tuple[float, float]]] = []
        major: list[list[tuple[float, float]]] = []
        for row in segments.itertuples():
            geometry = wkt.loads(row.geometry_wkt).simplify(5.0)
            parts = [geometry] if geometry.geom_type == "LineString" else list(geometry.geoms)
            for part in parts:
                coords = [(x / 1000.0, y / 1000.0) for x, y in part.coords]
                (major if int(row.major_road) else local).append(coords)
    else:
        # Reproducible fallback for a clean clone without processed data.
        local = [[(i, 0), (i, 10)] for i in range(12)] + [[(0, i), (12, i)] for i in range(11)]
        major = []

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    if local:
        ax.add_collection(LineCollection(local, colors="#d7e1ea", linewidths=0.45, alpha=0.85))
    if major:
        ax.add_collection(LineCollection(major, colors="#8fa6bf", linewidths=0.8, alpha=0.9))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.02)
    fig.subplots_adjust(0, 0, 1, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, transparent=True, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
