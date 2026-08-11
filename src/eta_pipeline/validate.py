"""Validation: type coercion, schema checks and row-level quality rules.

Rows are never dropped silently. Every rejected row is written to
`rejects.csv.gz` alongside the pipe-separated list of reason codes that
disqualified it, and every code is counted in `validation_report.json`.

Two severities:

* **hard** -- the row is removed from the feature table (`HARD_CODES`).
* **soft** -- the row is kept but marked with a `flag_*` column (`SOFT_CODES`),
  so downstream modelling can decide for itself.

A row may trip several codes at once, so the per-code counts sum to more than
the number of rejected rows. `rows_rejected` is the distinct-row figure.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import REQUIRED_COLUMNS, PipelineConfig
from .geo import average_speed_kmh, haversine_km

logger = logging.getLogger(__name__)

COORD_COLUMNS = (
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)
NUMERIC_COLUMNS = ("vendor_id", "passenger_count", "trip_duration", *COORD_COLUMNS)
TIMESTAMP_COLUMNS = ("pickup_datetime", "dropoff_datetime")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

HARD_CODES = (
    "MISSING_REQUIRED_FIELD",
    "INVALID_NUMERIC",
    "INVALID_TIMESTAMP",
    "TIMESTAMP_OUT_OF_RANGE",
    "NON_POSITIVE_DURATION",
    "DURATION_MISMATCH",
    "DURATION_OUT_OF_RANGE",
    "MISSING_GPS",
    "GPS_OUT_OF_BOUNDS",
    "IMPLAUSIBLE_SPEED",
    "DUPLICATE_ID",
)

SOFT_CODES = (
    "ZERO_PASSENGERS",
    "HIGH_PASSENGERS",
    "ZERO_DISTANCE",
    "STORE_AND_FWD",
)

# Required columns whose absence is reported as MISSING_REQUIRED_FIELD.
# The coordinates are excluded because they get the more specific MISSING_GPS.
_NON_COORD_REQUIRED = tuple(c for c in REQUIRED_COLUMNS if c not in COORD_COLUMNS)


@dataclass
class ValidationReport:
    """Counters accumulated across every chunk of a run."""

    rows_read: int = 0
    rows_kept: int = 0
    rows_rejected: int = 0
    reject_counts: Counter = field(default_factory=Counter)
    flag_counts: Counter = field(default_factory=Counter)
    null_counts: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        rejected = self.rows_rejected or 1  # guard the percentage only
        return {
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "rows_rejected": self.rows_rejected,
            "reject_rate": round(self.rows_rejected / max(self.rows_read, 1), 6),
            "reject_counts": {
                code: self.reject_counts.get(code, 0) for code in HARD_CODES
            },
            "reject_share_of_rejected": {
                code: round(self.reject_counts.get(code, 0) / rejected, 6)
                for code in HARD_CODES
            },
            "flag_counts": {code: self.flag_counts.get(code, 0) for code in SOFT_CODES},
            "raw_null_counts": dict(sorted(self.null_counts.items())),
        }


def coerce_types(chunk: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy of a string-typed raw chunk.

    Unparseable values become NaN/NaT rather than raising, so that bad data is
    reported as a finding instead of crashing the run.
    """
    typed = pd.DataFrame(index=chunk.index)
    typed["id"] = chunk["id"].astype("string")
    typed["store_and_fwd_flag"] = chunk["store_and_fwd_flag"].astype("string")

    for column in NUMERIC_COLUMNS:
        typed[column] = pd.to_numeric(chunk[column], errors="coerce")

    for column in TIMESTAMP_COLUMNS:
        parsed = pd.to_datetime(chunk[column], errors="coerce", format=TIMESTAMP_FORMAT)
        # Retry only the failures with format inference, so a legitimate but
        # differently-formatted timestamp is not reported as invalid.
        unparsed = parsed.isna() & chunk[column].notna()
        if unparsed.any():
            parsed.loc[unparsed] = pd.to_datetime(
                chunk.loc[unparsed, column], errors="coerce", format="mixed"
            )
        typed[column] = parsed

    return typed


def _in_bbox(lat: pd.Series, lon: pd.Series, config: PipelineConfig) -> pd.Series:
    return (
        lat.between(config.min_latitude, config.max_latitude)
        & lon.between(config.min_longitude, config.max_longitude)
    )


def validate_chunk(
    raw: pd.DataFrame,
    typed: pd.DataFrame,
    config: PipelineConfig,
    report: ValidationReport,
    seen_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a chunk into (clean, rejected).

    `seen_ids` is mutated so duplicate ids are caught across chunk boundaries,
    not just within one.
    """
    n = len(raw)
    report.rows_read += n
    for column in REQUIRED_COLUMNS:
        report.null_counts[column] += int(raw[column].isna().sum())

    codes: dict[str, pd.Series] = {}
    false_mask = pd.Series(False, index=raw.index)

    # --- presence -----------------------------------------------------------
    codes["MISSING_REQUIRED_FIELD"] = raw[list(_NON_COORD_REQUIRED)].isna().any(axis=1)

    # Present in the source but not parseable as a number.
    non_coord_numeric = [c for c in NUMERIC_COLUMNS if c not in COORD_COLUMNS]
    invalid_numeric = false_mask.copy()
    for column in non_coord_numeric:
        invalid_numeric |= raw[column].notna() & typed[column].isna()
    codes["INVALID_NUMERIC"] = invalid_numeric

    # --- GPS ----------------------------------------------------------------
    coords_missing = typed[list(COORD_COLUMNS)].isna().any(axis=1)
    tol = config.null_island_tolerance_deg
    null_island = (
        (typed["pickup_latitude"].abs() < tol) & (typed["pickup_longitude"].abs() < tol)
    ) | (
        (typed["dropoff_latitude"].abs() < tol)
        & (typed["dropoff_longitude"].abs() < tol)
    )
    codes["MISSING_GPS"] = coords_missing | null_island.fillna(False)

    coords_usable = ~coords_missing
    in_bounds = _in_bbox(
        typed["pickup_latitude"], typed["pickup_longitude"], config
    ) & _in_bbox(typed["dropoff_latitude"], typed["dropoff_longitude"], config)
    codes["GPS_OUT_OF_BOUNDS"] = coords_usable & ~null_island.fillna(False) & ~in_bounds

    # --- timestamps ---------------------------------------------------------
    pickup = typed["pickup_datetime"]
    dropoff = typed["dropoff_datetime"]
    unparseable = false_mask.copy()
    for column in TIMESTAMP_COLUMNS:
        unparseable |= raw[column].notna() & typed[column].isna()
    codes["INVALID_TIMESTAMP"] = unparseable

    min_date = pd.Timestamp(config.min_pickup_date)
    max_date = pd.Timestamp(config.max_pickup_date)
    codes["TIMESTAMP_OUT_OF_RANGE"] = pickup.notna() & (
        (pickup < min_date) | (pickup >= max_date)
    )

    both_ts = pickup.notna() & dropoff.notna()
    elapsed_s = (dropoff - pickup).dt.total_seconds()
    codes["NON_POSITIVE_DURATION"] = both_ts & (elapsed_s <= 0)

    # --- duration -----------------------------------------------------------
    duration = typed["trip_duration"]
    codes["DURATION_MISMATCH"] = (
        both_ts
        & duration.notna()
        & ((elapsed_s - duration).abs() > config.duration_mismatch_tolerance_s)
    )
    codes["DURATION_OUT_OF_RANGE"] = duration.notna() & (
        (duration < config.min_duration_s) | (duration > config.max_duration_s)
    )

    # --- plausibility -------------------------------------------------------
    distance_km = pd.Series(
        haversine_km(
            typed["pickup_latitude"],
            typed["pickup_longitude"],
            typed["dropoff_latitude"],
            typed["dropoff_longitude"],
        ),
        index=typed.index,
    )
    speed_kmh = pd.Series(
        average_speed_kmh(distance_km, duration), index=typed.index
    )
    codes["IMPLAUSIBLE_SPEED"] = speed_kmh.notna() & (
        speed_kmh > config.max_avg_speed_kmh
    )

    # --- duplicates ---------------------------------------------------------
    ids = typed["id"]
    within_chunk = ids.duplicated(keep="first")
    across_chunks = ids.isin(seen_ids)
    codes["DUPLICATE_ID"] = (within_chunk | across_chunks).fillna(False)

    # Register the ids we are about to accept as seen.
    hard_mask = false_mask.copy()
    for code in HARD_CODES:
        mask = codes[code].fillna(False).astype(bool)
        codes[code] = mask
        report.reject_counts[code] += int(mask.sum())
        hard_mask |= mask
    seen_ids.update(ids[~hard_mask].dropna().tolist())

    # --- soft flags ---------------------------------------------------------
    passengers = typed["passenger_count"]
    flags = {
        "ZERO_PASSENGERS": passengers.notna() & (passengers <= 0),
        "HIGH_PASSENGERS": passengers.notna()
        & (passengers > config.max_passenger_count),
        "ZERO_DISTANCE": distance_km.notna() & np.isclose(distance_km, 0.0),
        "STORE_AND_FWD": typed["store_and_fwd_flag"].fillna("N").str.upper().eq("Y"),
    }

    clean = typed.loc[~hard_mask].copy()
    for code, mask in flags.items():
        kept = mask.fillna(False).astype(bool).loc[~hard_mask]
        report.flag_counts[code] += int(kept.sum())
        clean[f"flag_{code.lower()}"] = kept.to_numpy(dtype="int8")

    rejects = raw.loc[hard_mask].copy()
    if len(rejects):
        hits = np.column_stack(
            [np.where(codes[code].loc[hard_mask], code, "") for code in HARD_CODES]
        )
        rejects["reject_reasons"] = ["|".join(filter(None, row)) for row in hits]

    report.rows_kept += len(clean)
    report.rows_rejected += int(hard_mask.sum())
    return clean, rejects
