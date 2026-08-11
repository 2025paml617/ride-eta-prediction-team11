"""Feature engineering: temporal, geospatial and weather features.

Leakage rules enforced here:

* `dropoff_datetime` never reaches the feature table -- knowing when the trip
  ended trivially gives away its duration.
* implied average speed is computed in `validate` for the plausibility check
  and is deliberately not rebuilt here, for the same reason.
* `pickup_datetime` is kept but tagged as *metadata*, not a feature. It is
  there so training can make a chronological train/validation split, which is
  the only honest way to evaluate an ETA model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .geo import bearing_deg, haversine_km, manhattan_km
from .weather import WeatherTable

logger = logging.getLogger(__name__)

# Columns that exist for auditing/splitting rather than for the model.
METADATA_COLUMNS = ("pickup_datetime",)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """The nth (1-based) `weekday` of a month; n=-1 means the last one."""
    if n > 0:
        first = pd.Timestamp(year=year, month=month, day=1)
        offset = (weekday - first.dayofweek) % 7
        return first + pd.Timedelta(days=offset + 7 * (n - 1))
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    offset = (last.dayofweek - weekday) % 7
    return last - pd.Timedelta(days=offset)


def _observed(date: pd.Timestamp) -> list[pd.Timestamp]:
    """A fixed-date holiday plus its observed weekday, when it falls on a weekend."""
    if date.dayofweek == 5:  # Saturday -> observed the Friday before
        return [date, date - pd.Timedelta(days=1)]
    if date.dayofweek == 6:  # Sunday -> observed the Monday after
        return [date, date + pd.Timedelta(days=1)]
    return [date]


def us_federal_holidays(year: int) -> set[pd.Timestamp]:
    """US federal holidays for a year, computed from the statutory rules.

    Implemented inline rather than via the `holidays` package to keep the
    pipeline dependency-free (see the environment notes in the README).
    """
    days: list[pd.Timestamp] = []
    for month, day in ((1, 1), (6, 19), (7, 4), (11, 11), (12, 25)):
        days.extend(_observed(pd.Timestamp(year=year, month=month, day=day)))
    days.append(_nth_weekday(year, 1, weekday=0, n=3))   # MLK Day
    days.append(_nth_weekday(year, 2, weekday=0, n=3))   # Presidents' Day
    days.append(_nth_weekday(year, 5, weekday=0, n=-1))  # Memorial Day
    days.append(_nth_weekday(year, 9, weekday=0, n=1))   # Labor Day
    days.append(_nth_weekday(year, 10, weekday=0, n=2))  # Columbus Day
    days.append(_nth_weekday(year, 11, weekday=3, n=4))  # Thanksgiving
    return set(days)


def add_temporal_features(
    frame: pd.DataFrame, config: PipelineConfig
) -> pd.DataFrame:
    """Calendar and clock features derived from the pickup timestamp."""
    pickup = frame["pickup_datetime"]

    frame["pickup_hour"] = pickup.dt.hour.astype("int8")
    frame["pickup_minute"] = pickup.dt.minute.astype("int8")
    frame["pickup_dow"] = pickup.dt.dayofweek.astype("int8")  # 0 = Monday
    frame["pickup_month"] = pickup.dt.month.astype("int8")
    frame["pickup_day"] = pickup.dt.day.astype("int8")
    frame["pickup_dayofyear"] = pickup.dt.dayofyear.astype("int16")
    frame["pickup_weekofyear"] = (
        pickup.dt.isocalendar().week.to_numpy(dtype="int16")
    )

    frame["is_weekend"] = (frame["pickup_dow"] >= 5).astype("int8")

    hour = frame["pickup_hour"]
    morning_from, morning_to = config.rush_hour_morning
    evening_from, evening_to = config.rush_hour_evening
    rush_window = hour.between(morning_from, morning_to - 1) | hour.between(
        evening_from, evening_to - 1
    )
    frame["is_rush_hour"] = (rush_window & (frame["is_weekend"] == 0)).astype("int8")

    # The night window wraps past midnight.
    frame["is_night"] = (
        (hour >= config.night_start_hour) | (hour <= config.night_end_hour)
    ).astype("int8")

    holidays: set[pd.Timestamp] = set()
    for year in pickup.dt.year.dropna().unique():
        holidays |= us_federal_holidays(int(year))
    frame["is_us_holiday"] = (
        pickup.dt.normalize().isin(holidays).to_numpy(dtype="int8")
    )

    # Cyclical encodings so that hour 23 sits next to hour 0.
    for name, values, period in (
        ("hour", frame["pickup_hour"], 24),
        ("dow", frame["pickup_dow"], 7),
        ("month", frame["pickup_month"] - 1, 12),
    ):
        angle = 2.0 * np.pi * values.to_numpy(dtype="float32") / period
        frame[f"{name}_sin"] = np.sin(angle).astype("float32")
        frame[f"{name}_cos"] = np.cos(angle).astype("float32")

    return frame


def add_geo_features(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Distance, direction and coarse zone features from the coordinates."""
    lat1 = frame["pickup_latitude"]
    lon1 = frame["pickup_longitude"]
    lat2 = frame["dropoff_latitude"]
    lon2 = frame["dropoff_longitude"]

    frame["haversine_km"] = haversine_km(lat1, lon1, lat2, lon2).astype("float32")
    frame["manhattan_km"] = manhattan_km(lat1, lon1, lat2, lon2).astype("float32")
    frame["bearing_deg"] = bearing_deg(lat1, lon1, lat2, lon2).astype("float32")

    # Coarse grid cells act as a cheap stand-in for neighbourhood ids.
    precision = config.zone_precision
    for prefix, lat, lon in (
        ("pickup", lat1, lon1),
        ("dropoff", lat2, lon2),
    ):
        frame[f"{prefix}_lat_cell"] = lat.round(precision).astype("float32")
        frame[f"{prefix}_lon_cell"] = lon.round(precision).astype("float32")

    return frame


def add_weather_features(frame: pd.DataFrame, weather: WeatherTable) -> pd.DataFrame:
    """Join hourly weather onto the pickup timestamp."""
    joined = weather.join(frame["pickup_datetime"])
    for column in joined.columns:
        frame[column] = joined[column]
    return frame


def build_features(
    clean: pd.DataFrame, config: PipelineConfig, weather: WeatherTable
) -> pd.DataFrame:
    """Turn a validated chunk into the model-ready feature table."""
    frame = clean.drop(columns=["dropoff_datetime"])  # leakage: end time
    frame = add_temporal_features(frame, config)
    frame = add_geo_features(frame, config)
    frame = add_weather_features(frame, weather)

    # Compact dtypes -- this is 1.4M rows and the file gets written every run.
    frame["vendor_id"] = frame["vendor_id"].astype("int8")
    frame["passenger_count"] = frame["passenger_count"].astype("int8")
    frame["trip_duration"] = frame["trip_duration"].astype("int32")
    for column in (
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ):
        frame[column] = frame[column].astype("float32")

    # store_and_fwd_flag is already captured as flag_store_and_fwd.
    frame = frame.drop(columns=["store_and_fwd_flag"])
    return frame[column_order(frame)]


def column_order(frame: pd.DataFrame) -> list[str]:
    """Stable column ordering: ids, metadata, features, target last."""
    from .config import IDENTIFIER_COLUMNS, TARGET_COLUMN

    head = [c for c in (*IDENTIFIER_COLUMNS, *METADATA_COLUMNS) if c in frame.columns]
    tail = [TARGET_COLUMN] if TARGET_COLUMN in frame.columns else []
    middle = [c for c in frame.columns if c not in head and c not in tail]
    return [*head, *middle, *tail]
