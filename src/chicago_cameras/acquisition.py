"""Download Chicago Data Portal (Socrata) datasets reproducibly.

The module implements the acquisition requirements of the research upgrade:

* paginated downloads against the documented Socrata endpoints;
* bounded retries with exponential backoff for transient failures;
* an explicit, recorded query (dataset id, ``$select``, ``$where``, ordering);
* acquisition timestamps, page/row counts and SHA-256 checksums;
* raw data preserved on disk as gzip-compressed CSV;
* a hard failure instead of any substitution when the real data are unavailable.

Nothing in this module fabricates, imputes or simulates observations.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://data.cityofchicago.org/resource"
PORTAL_URL = "https://data.cityofchicago.org"
DEFAULT_PAGE_SIZE = 50_000


class AcquisitionError(RuntimeError):
    """Raised when the portal cannot be read or returns an invalid response."""


@dataclass(frozen=True)
class DatasetSpec:
    """A single portal dataset and the exact query used to read it."""

    name: str
    dataset_id: str
    select: str | None = None
    where: str | None = None
    order: str = ":id"
    page_size: int = DEFAULT_PAGE_SIZE

    @property
    def clean_select(self) -> str | None:
        """Collapse a folded YAML string into a valid Socrata column list."""
        if self.select is None:
            return None
        columns = [part.strip() for part in self.select.split(",") if part.strip()]
        return ",".join(columns)

    def query_params(self, limit: int, offset: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "$limit": limit,
            "$offset": offset,
            "$order": self.order,
        }
        if self.clean_select:
            params["$select"] = self.clean_select
        if self.where:
            params["$where"] = self.where
        return params

    @property
    def human_url(self) -> str:
        return f"{PORTAL_URL}/d/{self.dataset_id}"


def build_session(
    total_retries: int = 6,
    backoff_factor: float = 1.5,
    timeout: float = 180.0,
    user_agent: str = "chicago-cameras-research/0.2 (reproducible research)",
) -> requests.Session:
    """Return a requests session with retries for transient portal failures."""
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent, "Accept": "text/csv"})
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def _get(
    session: requests.Session,
    spec: DatasetSpec,
    params: dict[str, Any],
    retry_on_empty_body: bool = True,
) -> requests.Response:
    url = f"{BASE_URL}/{spec.dataset_id}.csv"
    timeout = getattr(session, "request_timeout", 180.0)
    attempts = 3
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:  # network layer
            last_error = exc
            LOGGER.warning("request failed (%s/%s): %s", attempt + 1, attempts, exc)
            time.sleep(2.0 * (attempt + 1))
            continue
        if response.status_code in (200, 206):
            if not response.content and retry_on_empty_body and attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
                continue
            return response
        if response.status_code in (400, 403, 404):
            raise AcquisitionError(
                f"{spec.name}: HTTP {response.status_code} for {url} "
                f"with params {params}. Body: {response.text[:500]}"
            )
        if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
            wait = float(response.headers.get("Retry-After", 5 * (attempt + 1)))
            LOGGER.warning("portal returned %s; sleeping %.1fs", response.status_code, wait)
            time.sleep(wait)
            continue
        raise AcquisitionError(
            f"{spec.name}: HTTP {response.status_code} for {url}. Body: {response.text[:500]}"
        )
    raise AcquisitionError(f"{spec.name}: download failed after {attempts} attempts: {last_error}")


def download_dataset(
    spec: DatasetSpec,
    session: requests.Session | None = None,
    page_size: int | None = None,
    max_rows: int | None = None,
    pause: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download every page of a dataset and return it with provenance metadata.

    Values are read as strings so that the raw layer preserves exactly what the
    portal returned; typing happens in :mod:`chicago_cameras.cleaning`.
    """
    session = session or build_session()
    page_size = page_size or spec.page_size
    frames: list[pd.DataFrame] = []
    page_rows: list[int] = []
    offset = 0
    started = datetime.now(timezone.utc)

    while True:
        params = spec.query_params(limit=page_size, offset=offset)
        response = _get(session, spec, params)
        text = response.content.decode("utf-8", errors="strict")
        if not text.strip():
            break
        chunk = pd.read_csv(
            io.StringIO(text),
            dtype=str,
            keep_default_na=False,
            na_values=[""],
        )
        if chunk.empty:
            break
        frames.append(chunk)
        page_rows.append(len(chunk))
        offset += len(chunk)
        LOGGER.info("%s: %s rows downloaded", spec.name, f"{offset:,}")
        if len(chunk) < page_size:
            break
        if max_rows is not None and offset >= max_rows:
            break
        time.sleep(pause)

    if not frames:
        raise AcquisitionError(
            f"{spec.name}: the portal returned no rows for the configured query "
            f"({spec.clean_select=}, {spec.where=}). Refusing to continue with an "
            "empty dataset; no substitute data are created."
        )

    frame = pd.concat(frames, ignore_index=True)
    finished = datetime.now(timezone.utc)
    provenance = {
        "name": spec.name,
        "dataset_id": spec.dataset_id,
        "portal_url": spec.human_url,
        "api_endpoint": f"{BASE_URL}/{spec.dataset_id}.csv",
        "query": {
            "select": spec.clean_select,
            "where": spec.where,
            "order": spec.order,
            "page_size": page_size,
        },
        "pages": len(page_rows),
        "rows_per_page": page_rows,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "downloaded_at_utc": started.isoformat(),
        "completed_at_utc": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 1),
        "note": "Raw values are preserved verbatim as strings; see data/raw/.",
    }
    return frame, provenance


def save_raw(frame: pd.DataFrame, path: Path) -> str:
    """Write a gzip CSV and return its SHA-256 checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=6) as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], compression="gzip")


def update_provenance(path: Path, **entries: Any) -> dict[str, Any]:
    """Merge provenance entries into ``provenance.json`` deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update(entries)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return existing


def acquire(
    specs: Iterable[DatasetSpec],
    raw_dir: Path,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download (or reuse) each dataset, preserving raw files and provenance."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = raw_dir / "provenance.json"
    session = build_session()
    datasets: dict[str, pd.DataFrame] = {}

    for spec in specs:
        raw_path = raw_dir / f"{spec.name}.csv.gz"
        if raw_path.exists() and not refresh:
            LOGGER.info("%s: reusing %s", spec.name, raw_path)
            datasets[spec.name] = read_raw(raw_path)
            continue

        LOGGER.info("%s: downloading dataset %s", spec.name, spec.dataset_id)
        frame, provenance = download_dataset(spec, session=session)
        checksum = save_raw(frame, raw_path)
        provenance.update(
            {
                "raw_file": str(raw_path),
                "raw_sha256": checksum,
                "raw_bytes": raw_path.stat().st_size,
            }
        )
        update_provenance(provenance_path, **{spec.name: provenance})
        datasets[spec.name] = frame

    return datasets


def specs_from_config(config: dict[str, Any]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for name, entry in config["data"]["datasets"].items():
        specs.append(
            DatasetSpec(
                name=name,
                dataset_id=str(entry["dataset_id"]),
                select=entry.get("select"),
                where=entry.get("where"),
            )
        )
    return specs
