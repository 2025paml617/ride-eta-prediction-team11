"""Simulate feature drift and produce a retraining decision report.

The reference data is sampled from the preprocessed feature dataset. A
deterministic shift is applied to operationally important features to emulate
an incoming production window. PSI and relative mean shift are reported for
each feature. This module intentionally uses only NumPy and pandas so it can
run in the existing project environment without SciPy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/NYC_Preprocessed.csv")
DEFAULT_OUTPUT = Path("monitoring/drift_report.json")
DEFAULT_METRICS = Path("monitoring/drift_metrics.csv")
DRIFT_FEATURES = [
    "num__distance_km",
    "num__pickup_hour",
    "num__pickup_weekday",
    "num__temperature_c",
    "num__precipitation_mm",
    "num__wind_speed_kmh",
    "num__visibility_km",
]


def population_stability_index(reference, current, bins=10):
    """Calculate PSI using reference quantile bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    epsilon = 1e-6
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def simulate_drift(frame, seed=42, drift_scale=1.0, sample_size=10000):
    """Return reference data, shifted current data, and per-feature metrics."""
    available = [column for column in DRIFT_FEATURES if column in frame.columns]
    if not available:
        raise ValueError("No drift features found in the input dataset")
    sample_size = min(sample_size, len(frame))
    reference = frame[available].sample(sample_size, random_state=seed).reset_index(drop=True)
    current = reference.copy()
    rng = np.random.default_rng(seed)

    # These shifts represent a plausible operational change: longer trips,
    # different pickup-time mix, and changed weather conditions.
    shifts = {
        "num__distance_km": 0.35,
        "num__pickup_hour": 0.45,
        "num__pickup_weekday": 0.25,
        "num__temperature_c": 0.60,
        "num__precipitation_mm": 0.80,
        "num__wind_speed_kmh": 0.45,
        "num__visibility_km": -0.50,
    }
    for column in available:
        if column in shifts:
            current[column] = current[column].astype(float) + shifts[column] * drift_scale
        # Small noise avoids creating an artificial constant distribution.
        current[column] = current[column].astype(float) + rng.normal(
            0, 0.01, size=len(current)
        )

    metrics = []
    for column in available:
        ref_mean = float(reference[column].mean())
        cur_mean = float(current[column].mean())
        denominator = max(abs(ref_mean), 1e-6)
        psi = population_stability_index(reference[column], current[column])
        metrics.append({
            "feature": column,
            "reference_mean": ref_mean,
            "current_mean": cur_mean,
            "relative_mean_shift": abs(cur_mean - ref_mean) / denominator,
            "psi": psi,
        })
    return reference, current, pd.DataFrame(metrics)


def run(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, metrics_path=DEFAULT_METRICS,
        seed=42, drift_scale=1.0, sample_size=10000, psi_threshold=0.20):
    """Run simulation and write JSON/CSV outputs."""
    frame = pd.read_csv(input_path)
    _, _, metrics = simulate_drift(
        frame, seed=seed, drift_scale=drift_scale, sample_size=sample_size
    )
    metrics["drift_detected"] = metrics["psi"] >= psi_threshold
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    report = {
        "input": str(input_path),
        "seed": seed,
        "sample_size": int(min(sample_size, len(frame))),
        "drift_scale": drift_scale,
        "psi_threshold": psi_threshold,
        "drifted_features": metrics.loc[metrics["drift_detected"], "feature"].tolist(),
        "retraining_triggered": bool(metrics["drift_detected"].any()),
        "metrics_file": str(metrics_path),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--psi-threshold", type=float, default=0.20)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    report, metrics = run(
        input_path=args.input,
        output_path=args.output,
        metrics_path=args.metrics,
        seed=args.seed,
        drift_scale=args.drift_scale,
        sample_size=args.sample_size,
        psi_threshold=args.psi_threshold,
    )
    print(json.dumps(report, indent=2))
    if args.fail_on_drift and report["retraining_triggered"]:
        raise SystemExit(2)
