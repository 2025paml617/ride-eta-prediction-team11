"""Pipeline configuration.

Every knob that affects the contents of the output dataset lives here, so the
config can be serialised into the manifest and hashed into the version
fingerprint. If a value changes the data, it belongs in `PipelineConfig`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Repo root, resolved from this file: src/eta_pipeline/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE = REPO_ROOT / "NYC Taxi Trip Duration.zip"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed"
DEFAULT_EXTERNAL_DIR = REPO_ROOT / "data" / "external"

# Columns the raw CSV must provide. The header gate in ingest.py checks these.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "trip_duration",
)

TARGET_COLUMN = "trip_duration"
IDENTIFIER_COLUMNS: tuple[str, ...] = ("id",)


@dataclass(frozen=True)
class PipelineConfig:
    """Parameters controlling ingestion, validation and feature engineering."""

    # --- Sources and sinks -------------------------------------------------
    source_path: Path = DEFAULT_SOURCE
    csv_member: str = "NYC.csv"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    external_dir: Path = DEFAULT_EXTERNAL_DIR

    # --- Ingestion ---------------------------------------------------------
    chunksize: int = 200_000
    sample_rows: int | None = None

    # --- Validation: geography --------------------------------------------
    # Bounding box covering the five boroughs plus a margin for Newark/Yonkers.
    min_longitude: float = -74.5
    max_longitude: float = -73.5
    min_latitude: float = 40.3
    max_latitude: float = 41.1
    # A ping this close to (0, 0) is a dead GPS unit, not a trip to the Atlantic.
    null_island_tolerance_deg: float = 0.01

    # --- Validation: time and duration ------------------------------------
    min_pickup_date: str = "2016-01-01"
    max_pickup_date: str = "2017-01-01"  # exclusive upper bound
    min_duration_s: int = 10
    max_duration_s: int = 6 * 3600
    # trip_duration should equal dropoff - pickup; allow a second of rounding.
    duration_mismatch_tolerance_s: int = 1

    # --- Validation: plausibility -----------------------------------------
    max_avg_speed_kmh: float = 150.0
    max_passenger_count: int = 8

    # --- Feature engineering ----------------------------------------------
    # Decimal places used to bucket coordinates into coarse zone ids.
    zone_precision: int = 3
    rush_hour_morning: tuple[int, int] = (7, 10)
    rush_hour_evening: tuple[int, int] = (16, 20)
    night_start_hour: int = 22
    night_end_hour: int = 5

    # --- Weather -----------------------------------------------------------
    # One of: "none", "csv", "fetch". See weather.py.
    weather_mode: str = "none"
    weather_csv: Path | None = None
    weather_latitude: float = 40.7128
    weather_longitude: float = -74.0060

    # --- Run behaviour (excluded from the fingerprint) ---------------------
    force: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.weather_mode not in {"none", "csv", "fetch"}:
            raise ValueError(f"unknown weather_mode: {self.weather_mode!r}")
        if self.weather_mode == "csv" and self.weather_csv is None:
            raise ValueError("weather_mode='csv' requires weather_csv")
        if self.chunksize <= 0:
            raise ValueError("chunksize must be positive")
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be positive when set")

    def to_dict(self) -> dict:
        """JSON-safe view of the config, for the manifest."""
        out = {}
        for key, value in asdict(self).items():
            out[key] = str(value) if isinstance(value, Path) else value
        return out

    def fingerprint_payload(self) -> str:
        """Canonical JSON of the data-affecting settings.

        `force` is a run-time switch and `chunksize` only changes how the work
        is batched, so neither may influence the fingerprint -- otherwise an
        identical dataset produced with a different batch size would look like
        a new version.
        """
        payload = self.to_dict()
        for volatile in ("force", "chunksize", "output_root", "external_dir"):
            payload.pop(volatile, None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
