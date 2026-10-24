#!/usr/bin/env python
"""Download the three Chicago Data Portal datasets and preserve provenance."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chicago_cameras.acquisition import acquire, specs_from_config  # noqa: E402
from chicago_cameras.config import load_config, path_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download even when a raw file already exists.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    raw_dir = path_for(config, "raw")
    specs = specs_from_config(config)
    datasets = acquire(specs, raw_dir, refresh=args.refresh)

    for name, frame in datasets.items():
        print(f"{name:10s} rows={len(frame):>10,}  columns={len(frame.columns):>2}")
    print(f"Raw files and provenance: {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
