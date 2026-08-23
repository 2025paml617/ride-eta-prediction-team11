"""Pipeline runner: orchestrates ingestion, validation, and feature engineering."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT, PipelineConfig
from .features import build_features
from .ingest import iter_chunks
from .validate import coerce_types, validate_chunk, ValidationReport
from .weather import build_weather_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eta_pipeline.run")


def run_pipeline(config: PipelineConfig) -> Path:
    """Run the ingestion, validation, and feature engineering pipeline."""
    start_time = time.time()

    # Calculate fingerprint and version
    payload = config.fingerprint_payload()
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    version_dir = config.output_root / f"v_{fingerprint}"

    logger.info("Starting pipeline run...")
    logger.info("Fingerprint: %s", fingerprint)
    logger.info("Version directory: %s", version_dir)

    if version_dir.exists() and not config.force:
        logger.info(
            "Version directory %s already exists. Use `force=True` to overwrite. Skipping execution.",
            version_dir,
        )
        return version_dir

    version_dir.mkdir(parents=True, exist_ok=True)

    # Resolve weather
    logger.info("Resolving weather table...")
    weather = build_weather_table(config)

    # Initialize report and seen IDs
    report = ValidationReport()
    seen_ids: set[str] = set()

    features_path = version_dir / "features.csv.gz"
    rejects_path = version_dir / "rejects.csv.gz"

    is_first_chunk = True
    is_first_reject = True

    try:
        # Open output files using gzip
        with gzip.open(features_path, "wt", encoding="utf-8", newline="") as f_feat, \
             gzip.open(rejects_path, "wt", encoding="utf-8", newline="") as f_rej:

            for chunk_idx, raw_chunk in enumerate(iter_chunks(config)):
                # Coerce types
                typed_chunk = coerce_types(raw_chunk)

                # Validate chunk
                clean_chunk, rejects_chunk = validate_chunk(
                    raw_chunk, typed_chunk, config, report, seen_ids
                )

                # Engineer features on clean chunk
                if not clean_chunk.empty:
                    features_chunk = build_features(clean_chunk, config, weather)
                    features_chunk.to_csv(
                        f_feat, index=False, header=is_first_chunk
                    )
                    is_first_chunk = False

                # Write rejects chunk
                if not rejects_chunk.empty:
                    rejects_chunk.to_csv(
                        f_rej, index=False, header=is_first_reject
                    )
                    is_first_reject = False

    except Exception as e:
        logger.exception("Pipeline run failed")
        # Clean up partial files to prevent corrupted cache
        if features_path.exists():
            try:
                features_path.unlink()
            except OSError:
                pass
        if rejects_path.exists():
            try:
                rejects_path.unlink()
            except OSError:
                pass
        raise e

    # If no rejects were written, we can remove the empty rejects file
    if is_first_reject and rejects_path.exists():
        try:
            rejects_path.unlink()
        except OSError:
            pass

    # Save validation report
    report_dict = report.to_dict()
    report_path = version_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Save manifest
    manifest = {
        "version": fingerprint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "config": config.to_dict(),
        "report": report_dict,
    }
    manifest_path = version_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Pipeline run completed successfully in %.2fs", time.time() - start_time)
    logger.info("Kept %d of %d rows (reject rate: %.4f%%)", 
                report.rows_kept, report.rows_read, report_dict["reject_rate"] * 100)

    return version_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the ride ETA data pipeline.")
    parser.add_argument(
        "--source-path",
        type=Path,
        help="Path to the raw taxi trip dataset zip or csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Root directory for the processed versioned output.",
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        help="Directory for external data like weather downloads.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        help="Number of rows per chunk during streaming.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        help="Limit the total number of rows read (for fast iteration).",
    )
    parser.add_argument(
        "--weather-mode",
        choices=["none", "csv", "fetch"],
        help="Weather integration mode.",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        help="Path to weather CSV (required if weather-mode is csv).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if fingerprint output directory exists.",
    )

    args = parser.parse_args()

    # Create config overrides dict, filtering out None values
    overrides = {k: v for k, v in vars(args).items() if v is not None}

    # Instantiate PipelineConfig with overrides
    config = PipelineConfig(**overrides)

    run_pipeline(config)


if __name__ == "__main__":
    main()
