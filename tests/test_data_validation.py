import pandas as pd
import pytest

from data.data_validation import DataValidationError, validate_ingested_data


def valid_row(**overrides):
    row = {
        "id": "id0001",
        "vendor_id": 1,
        "pickup_datetime": "2016-01-01 08:00:00",
        "dropoff_datetime": "2016-01-01 08:15:00",
        "pickup_longitude": -73.98,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.97,
        "dropoff_latitude": 40.76,
        "passenger_count": 1,
        "store_and_fwd_flag": "N",
        "trip_duration": 900,
    }
    row.update(overrides)
    return row


def test_validation_removes_missing_gps_and_invalid_timestamps():
    result = validate_ingested_data(pd.DataFrame([
        valid_row(),
        valid_row(id="id0002", pickup_longitude=None),
        valid_row(id="id0003", pickup_datetime="not-a-timestamp"),
    ]))

    assert len(result.data) == 1
    assert result.report["missing_gps_rows"] == 1
    assert result.report["invalid_timestamp_rows"] == 1


def test_validation_requires_schema_columns():
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_ingested_data(pd.DataFrame([{"id": "id0001"}]))
