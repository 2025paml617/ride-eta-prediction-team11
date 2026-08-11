"""Weather enrichment.

The trip dataset carries no weather, so it has to come from outside. Three
modes are supported and all three return the *same* canonical columns, so
downstream code never has to branch on which one was used:

``none``   no external data; the columns are present but null and
           ``weather_available`` is 0. This is the default: the pipeline must
           not silently depend on the network.
``csv``    a local file, auto-detected between the two common shapes (the
           Kaggle "KNYC Metars 2016" hourly export and NOAA GHCN-Daily).
``fetch``  one HTTPS request to the Open-Meteo historical archive (free, no
           API key), cached under ``data/external/`` and reused thereafter.

The join key is (date, hour) of pickup. Daily sources are broadcast across all
24 hours of their day.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)

WEATHER_COLUMNS = (
    "temp_c",
    "precip_mm",
    "snowfall_mm",
    "wind_kph",
    "is_raining",
    "is_snowing",
    "weather_available",
)

_OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
_OPEN_METEO_HOURLY = "temperature_2m,precipitation,snowfall,wind_speed_10m"

# Column-name signatures used to recognise a local weather CSV.
_KNYC_COLUMNS = {"pickup_datetime", "tempm", "precip"}
_GHCN_COLUMNS = {"DATE", "PRCP"}


class WeatherTable:
    """Hourly weather indexed by (date, hour), ready to join onto trips."""

    def __init__(self, frame: pd.DataFrame | None, source: str) -> None:
        self.frame = frame
        self.source = source

    @property
    def available(self) -> bool:
        return self.frame is not None and not self.frame.empty

    def join(self, pickup: pd.Series) -> pd.DataFrame:
        """Return weather columns aligned to a Series of pickup timestamps."""
        index = pickup.index
        if not self.available:
            out = pd.DataFrame(
                {c: np.full(len(index), np.nan) for c in WEATHER_COLUMNS[:-1]},
                index=index,
            )
            out["weather_available"] = np.int8(0)
            return out

        key = pd.DataFrame(
            {
                "_date": pickup.dt.normalize(),
                "_hour": pickup.dt.hour.astype("float64"),
            },
            index=index,
        )
        merged = key.merge(self.frame, on=["_date", "_hour"], how="left")
        merged.index = index

        out = merged[list(WEATHER_COLUMNS[:-1])].copy()
        out["weather_available"] = out["temp_c"].notna().to_numpy(dtype="int8")
        return out


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived flags and keep only the canonical join + value columns."""
    frame["is_raining"] = (frame["precip_mm"].fillna(0.0) > 0.0).astype("int8")
    frame["is_snowing"] = (frame["snowfall_mm"].fillna(0.0) > 0.0).astype("int8")
    columns = ["_date", "_hour", *WEATHER_COLUMNS[:-1]]
    frame = frame[columns].drop_duplicates(subset=["_date", "_hour"], keep="first")
    return frame.reset_index(drop=True)


def _broadcast_daily_to_hourly(daily: pd.DataFrame) -> pd.DataFrame:
    """Expand one row per day into 24 rows, one per hour."""
    hours = pd.DataFrame({"_hour": np.arange(24, dtype="float64")})
    return daily.merge(hours, how="cross")


def load_weather_csv(path: Path) -> pd.DataFrame:
    """Read a local weather CSV and normalise it to the canonical schema."""
    raw = pd.read_csv(path)
    columns = set(raw.columns)

    if _KNYC_COLUMNS <= columns:
        # Kaggle KNYC Metars 2016: hourly, metric, 'pickup_datetime' timestamp.
        stamp = pd.to_datetime(raw["pickup_datetime"], errors="coerce")
        frame = pd.DataFrame(
            {
                "_date": stamp.dt.normalize(),
                "_hour": stamp.dt.hour.astype("float64"),
                "temp_c": pd.to_numeric(raw["tempm"], errors="coerce"),
                "precip_mm": pd.to_numeric(raw["precip"], errors="coerce"),
                "snowfall_mm": pd.to_numeric(
                    raw.get("snow", pd.Series(np.nan, index=raw.index)),
                    errors="coerce",
                ),
                "wind_kph": pd.to_numeric(
                    raw.get("wspdm", pd.Series(np.nan, index=raw.index)),
                    errors="coerce",
                ),
            }
        ).dropna(subset=["_date"])
        logger.info("weather: parsed %s as KNYC hourly (%d rows)", path.name, len(frame))
        return _finalise(frame)

    if _GHCN_COLUMNS <= columns:
        # NOAA GHCN-Daily: tenths of mm / tenths of degC, one row per day.
        date = pd.to_datetime(raw["DATE"], errors="coerce")
        tmax = pd.to_numeric(raw.get("TMAX"), errors="coerce")
        tmin = pd.to_numeric(raw.get("TMIN"), errors="coerce")
        daily = pd.DataFrame(
            {
                "_date": date.dt.normalize(),
                "temp_c": (tmax + tmin) / 2.0 / 10.0,
                "precip_mm": pd.to_numeric(raw["PRCP"], errors="coerce") / 10.0,
                "snowfall_mm": pd.to_numeric(raw.get("SNOW"), errors="coerce"),
                "wind_kph": pd.to_numeric(raw.get("AWND"), errors="coerce") * 3.6 / 10.0,
            }
        ).dropna(subset=["_date"])
        logger.info("weather: parsed %s as GHCN-Daily (%d days)", path.name, len(daily))
        return _finalise(_broadcast_daily_to_hourly(daily))

    # Already canonical (e.g. our own cached Open-Meteo export).
    if {"time", "temperature_2m"} <= columns:
        return _parse_open_meteo_frame(raw)

    raise ValueError(
        f"unrecognised weather CSV schema in {path}; columns: {sorted(columns)}"
    )


def _parse_open_meteo_frame(raw: pd.DataFrame) -> pd.DataFrame:
    stamp = pd.to_datetime(raw["time"], errors="coerce")
    frame = pd.DataFrame(
        {
            "_date": stamp.dt.normalize(),
            "_hour": stamp.dt.hour.astype("float64"),
            "temp_c": pd.to_numeric(raw["temperature_2m"], errors="coerce"),
            "precip_mm": pd.to_numeric(raw["precipitation"], errors="coerce"),
            # Open-Meteo reports snowfall in cm.
            "snowfall_mm": pd.to_numeric(raw["snowfall"], errors="coerce") * 10.0,
            "wind_kph": pd.to_numeric(raw["wind_speed_10m"], errors="coerce"),
        }
    ).dropna(subset=["_date"])
    return _finalise(frame)


def fetch_weather(config: PipelineConfig) -> pd.DataFrame:
    """Download hourly NYC weather for the configured date range, with a cache."""
    import requests  # imported lazily so the offline path needs no network stack

    cache_dir = Path(config.external_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    start = config.min_pickup_date
    end = (pd.Timestamp(config.max_pickup_date) - pd.Timedelta(days=1)).date().isoformat()
    cache = cache_dir / f"nyc_weather_{start}_{end}.csv"

    if cache.exists():
        logger.info("weather: using cached download %s", cache)
        return _parse_open_meteo_frame(pd.read_csv(cache))

    params = {
        "latitude": config.weather_latitude,
        "longitude": config.weather_longitude,
        "start_date": start,
        "end_date": end,
        "hourly": _OPEN_METEO_HOURLY,
        "timezone": "America/New_York",
    }
    logger.info("weather: requesting %s %s..%s", _OPEN_METEO_URL, start, end)
    response = requests.get(_OPEN_METEO_URL, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    raw = pd.DataFrame(hourly)
    raw.to_csv(cache, index=False)
    logger.info("weather: cached %d hourly records to %s", len(raw), cache)
    return _parse_open_meteo_frame(raw)


def build_weather_table(config: PipelineConfig) -> WeatherTable:
    """Resolve `config.weather_mode` into a joinable table."""
    if config.weather_mode == "csv":
        path = Path(config.weather_csv)
        return WeatherTable(load_weather_csv(path), f"csv:{path.name}")

    if config.weather_mode == "fetch":
        return WeatherTable(fetch_weather(config), "open-meteo-archive")

    logger.warning(
        "weather: no source configured -- weather columns will be null and "
        "weather_available=0. Pass --weather-csv or --fetch-weather for real data."
    )
    return WeatherTable(None, "none")
