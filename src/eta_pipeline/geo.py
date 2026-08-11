"""Vectorised great-circle geometry.

Kept separate from `features` so that `validate` can reuse the distance
calculation for its implausible-speed check without importing the feature
builders.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two arrays of points."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype="float64")) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def manhattan_km(lat1, lon1, lat2, lon2):
    """Distance along a lat leg plus a lon leg.

    Manhattan's street grid means a taxi almost never travels the great-circle
    line, so this is usually the better distance proxy for ETA models.
    """
    lat_leg = haversine_km(lat1, lon1, lat2, lon1)
    lon_leg = haversine_km(lat1, lon1, lat1, lon2)
    return lat_leg + lon_leg


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing from point 1 to point 2, in degrees [0, 360)."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype="float64")) for v in (lat1, lon1, lat2, lon2)
    )
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x)) % 360.0


def average_speed_kmh(distance_km, duration_s):
    """Implied average speed. Zero/negative durations yield NaN, not infinity.

    This is a function of the target and must never reach the feature table --
    it exists only so validation can reject physically impossible trips.
    """
    duration_s = np.asarray(duration_s, dtype="float64")
    safe = np.where(duration_s > 0, duration_s, np.nan)
    return np.asarray(distance_km, dtype="float64") / (safe / 3600.0)
