"""Unit tests for the ETA prediction data pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.eta_pipeline.config import PipelineConfig
from src.eta_pipeline.features import add_temporal_features, add_geo_features
from src.eta_pipeline.validate import coerce_types, validate_chunk, ValidationReport
from src.eta_pipeline.geo import haversine_km, bearing_deg


class TestPipelineConfig(unittest.TestCase):
    """Tests for the PipelineConfig class."""

    def test_fingerprint_and_payload(self):
        """Test that data-affecting fields are fingerprintable and run-time ones are not."""
        config1 = PipelineConfig(force=False, chunksize=10000)
        config2 = PipelineConfig(force=True, chunksize=20000)
        self.assertEqual(config1.fingerprint_payload(), config2.fingerprint_payload())

        config3 = PipelineConfig(min_pickup_date="2016-02-01")
        self.assertNotEqual(config1.fingerprint_payload(), config3.fingerprint_payload())


class TestFeatureEngineering(unittest.TestCase):
    """Tests for temporal and geospatial feature building."""

    def test_temporal_features(self):
        """Test extraction of clock, holiday, and weekend features."""
        config = PipelineConfig()
        
        # Test dates:
        # 1. 2016-01-01 08:30:00 -> New Year's Day (US holiday), Friday, morning rush hour
        # 2. 2016-01-02 23:30:00 -> Saturday night, weekend
        data = pd.DataFrame({
            "pickup_datetime": pd.to_datetime([
                "2016-01-01 08:30:00",
                "2016-01-02 23:30:00",
            ])
        })
        
        feat = add_temporal_features(data, config)
        
        self.assertEqual(feat.loc[0, "pickup_hour"], 8)
        self.assertEqual(feat.loc[0, "pickup_dow"], 4)  # Friday
        self.assertEqual(feat.loc[0, "is_us_holiday"], 1)  # New Year's Day
        self.assertEqual(feat.loc[0, "is_rush_hour"], 1)
        self.assertEqual(feat.loc[0, "is_weekend"], 0)

        self.assertEqual(feat.loc[1, "pickup_hour"], 23)
        self.assertEqual(feat.loc[1, "pickup_dow"], 5)  # Saturday
        self.assertEqual(feat.loc[1, "is_us_holiday"], 0)
        self.assertEqual(feat.loc[1, "is_rush_hour"], 0)
        self.assertEqual(feat.loc[1, "is_weekend"], 1)
        self.assertEqual(feat.loc[1, "is_night"], 1)

    def test_geo_features(self):
        """Test haversine distance and bearing calculation."""
        config = PipelineConfig(zone_precision=2)
        
        # Simple coordinate pairs
        data = pd.DataFrame({
            "pickup_latitude": [40.7128],
            "pickup_longitude": [-74.0060],
            "dropoff_latitude": [40.7589],
            "dropoff_longitude": [-73.9851],
        })
        
        feat = add_geo_features(data, config)
        
        self.assertGreater(feat.loc[0, "haversine_km"], 0)
        self.assertGreater(feat.loc[0, "manhattan_km"], 0)
        self.assertTrue(0 <= feat.loc[0, "bearing_deg"] < 360)
        self.assertEqual(feat.loc[0, "pickup_lat_cell"], 40.71)
        self.assertEqual(feat.loc[0, "pickup_lon_cell"], -74.01)


class TestValidation(unittest.TestCase):
    """Tests for type coercion and chunk-level validation rules."""

    def test_coerce_types(self):
        """Test coercion from string representation to appropriate data types."""
        raw = pd.DataFrame({
            "id": ["1"],
            "vendor_id": ["2"],
            "passenger_count": ["5"],
            "trip_duration": ["600"],
            "pickup_latitude": ["40.7128"],
            "pickup_longitude": ["-74.0060"],
            "dropoff_latitude": ["40.7589"],
            "dropoff_longitude": ["-73.9851"],
            "pickup_datetime": ["2016-01-01 08:30:00"],
            "dropoff_datetime": ["2016-01-01 08:40:00"],
            "store_and_fwd_flag": ["N"],
        })
        
        typed = coerce_types(raw)
        self.assertEqual(typed.loc[0, "vendor_id"], 2)
        self.assertEqual(typed.loc[0, "passenger_count"], 5)
        self.assertEqual(typed.loc[0, "trip_duration"], 600)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(typed["pickup_datetime"]))

    def test_validate_chunk(self):
        """Test that bad records are rejected and soft flags are added."""
        config = PipelineConfig(
            min_latitude=40.0, max_latitude=42.0,
            min_longitude=-75.0, max_longitude=-73.0,
        )
        report = ValidationReport()
        seen_ids = set()

        # Row 0: Valid row
        # Row 1: Invalid GPS (out of bounds)
        # Row 2: Negative trip duration (hard reject)
        # Row 3: Soft flag (0 passengers)
        raw = pd.DataFrame({
            "id": ["r0", "r1", "r2", "r3"],
            "vendor_id": ["1", "1", "1", "1"],
            "passenger_count": ["2", "1", "1", "0"],
            "trip_duration": ["600", "600", "-50", "600"],
            "pickup_latitude": ["40.7128", "10.0", "40.7128", "40.7128"],
            "pickup_longitude": ["-74.0060", "-74.0060", "-74.0060", "-74.0060"],
            "dropoff_latitude": ["40.7589", "40.7589", "40.7589", "40.7589"],
            "dropoff_longitude": ["-73.9851", "-73.9851", "-73.9851", "-73.9851"],
            "pickup_datetime": ["2016-01-01 08:00:00", "2016-01-01 08:00:00", "2016-01-01 08:00:00", "2016-01-01 08:00:00"],
            "dropoff_datetime": ["2016-01-01 08:10:00", "2016-01-01 08:10:00", "2016-01-01 08:10:00", "2016-01-01 08:10:00"],
            "store_and_fwd_flag": ["N", "N", "N", "N"],
        })

        typed = coerce_types(raw)
        clean, rejects = validate_chunk(raw, typed, config, report, seen_ids)

        self.assertEqual(len(clean), 2)  # r0 and r3 should be kept
        self.assertEqual(len(rejects), 2)  # r1 and r2 are hard rejects
        
        self.assertIn("r0", clean["id"].values)
        self.assertIn("r3", clean["id"].values)
        self.assertIn("r1", rejects["id"].values)
        self.assertIn("r2", rejects["id"].values)

        # Check soft flag
        self.assertEqual(clean.loc[clean["id"] == "r3", "flag_zero_passengers"].values[0], 1)
        self.assertEqual(clean.loc[clean["id"] == "r0", "flag_zero_passengers"].values[0], 0)


if __name__ == "__main__":
    unittest.main()
