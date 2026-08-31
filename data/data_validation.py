"""Validation utilities for the raw NYC taxi dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "id", "vendor_id", "pickup_datetime", "dropoff_datetime",
    "pickup_longitude", "pickup_latitude", "dropoff_longitude",
    "dropoff_latitude", "passenger_count", "store_and_fwd_flag",
    "trip_duration",
]
GPS_COLUMNS = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
]


class DataValidationError(ValueError):
    """Raised when the ingested dataset violates the schema contract."""


@dataclass
class ValidationResult:
    data: pd.DataFrame
    report: dict[str, Any]


def validate_ingested_data(df: pd.DataFrame) -> ValidationResult:
    """Validate and clean ingested data, returning an auditable row report."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")

    data = df.copy()
    report: dict[str, Any] = {
        "input_rows": int(len(data)),
        "duplicate_rows": int(data.duplicated().sum()),
    }
    for column in GPS_COLUMNS + ["passenger_count", "trip_duration", "vendor_id"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["pickup_datetime", "dropoff_datetime"]:
        data[column] = pd.to_datetime(data[column], errors="coerce", utc=True)

    missing_gps = data[GPS_COLUMNS].isna().any(axis=1)
    invalid_timestamps = (
        data["pickup_datetime"].isna()
        | data["dropoff_datetime"].isna()
        | (data["dropoff_datetime"] < data["pickup_datetime"])
    )
    invalid_coordinates = (
        ~data["pickup_latitude"].between(-90, 90)
        | ~data["dropoff_latitude"].between(-90, 90)
        | ~data["pickup_longitude"].between(-180, 180)
        | ~data["dropoff_longitude"].between(-180, 180)
    )
    invalid_values = (
        data["passenger_count"].isna() | (data["passenger_count"] < 1)
        | data["trip_duration"].isna() | (data["trip_duration"] < 0)
        | ~data["vendor_id"].isin([1, 2])
    )
    invalid_rows = missing_gps | invalid_timestamps | invalid_coordinates | invalid_values
    report.update({
        "missing_gps_rows": int(missing_gps.sum()),
        "invalid_timestamp_rows": int(invalid_timestamps.sum()),
        "invalid_coordinate_rows": int(invalid_coordinates.sum()),
        "invalid_value_rows": int(invalid_values.sum()),
        "removed_rows": int(invalid_rows.sum()),
    })
    data = data.loc[~invalid_rows].drop_duplicates().reset_index(drop=True)
    report["output_rows"] = int(len(data))
    report["status"] = "passed" if len(data) else "failed"
    if data.empty:
        raise DataValidationError("Validation removed every row from the dataset")
    return ValidationResult(data=data, report=report)


def load_and_validate_data(filepath: str) -> ValidationResult:
    """Read a CSV file and validate it immediately after ingestion."""
    return validate_ingested_data(pd.read_csv(filepath))
