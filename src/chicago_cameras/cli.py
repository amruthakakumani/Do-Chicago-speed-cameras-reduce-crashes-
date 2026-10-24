"""Command-line interface.

Examples
--------
    chicago-cameras download
    chicago-cameras analyze --config config/analysis.yaml
    chicago-cameras verify-replication
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _download(args: argparse.Namespace) -> int:
    from .acquisition import acquire, specs_from_config
    from .config import load_config, path_for

    config = load_config(args.config)
    datasets = acquire(specs_from_config(config), path_for(config, "raw"), refresh=args.refresh)
    for name, frame in datasets.items():
        print(f"{name:10s} rows={len(frame):>10,}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from .config import load_config
    from .pipeline import run

    run(load_config(args.config), refresh=args.refresh, quick=args.quick)
    return 0


def _verify(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "verify_replication.py"
    if not script.exists():  # pragma: no cover - installed package without repo
        print(f"verify_replication.py not found at {script}", file=sys.stderr)
        return 2
    import runpy

    sys.argv = [str(script), "--config", args.config]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chicago-cameras", description=__doc__)
    parser.add_argument("--config", default="config/analysis.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download the three portal datasets")
    download.add_argument("--refresh", action="store_true")
    download.set_defaults(func=_download)

    analyze = subparsers.add_parser("analyze", help="Run the full analysis pipeline")
    analyze.add_argument("--refresh", action="store_true", help="Re-download raw data")
    analyze.add_argument("--quick", action="store_true", help="Fewer bootstrap/placebo draws")
    analyze.set_defaults(func=_analyze)

    verify = subparsers.add_parser(
        "verify-replication", help="Reproduce the original notebook's estimates"
    )
    verify.set_defaults(func=_verify)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
