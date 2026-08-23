"""Model comparison and selection script.

Queries MLflow runs in the active experiment, displays a summary table,
selects the run with the lowest validation RMSE, and registers it as the
production-grade active model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil

import mlflow
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eta_model.compare")


def compare_and_select(experiment_name: str, production_model_dir: Path) -> None:
    """Find the best model run and copy its joblib artifact to production_model_dir."""
    mlflow.set_experiment(experiment_name)

    logger.info("Querying MLflow runs for experiment '%s'...", experiment_name)
    runs = mlflow.search_runs(experiment_names=[experiment_name])

    if runs.empty:
        logger.error("No runs found for experiment '%s'. Cannot compare.", experiment_name)
        return

    # Check for required validation metrics
    if "metrics.val_rmse" not in runs.columns:
        logger.error(
            "None of the runs have 'val_rmse' metrics. Has training run successfully?"
        )
        return

    # Filter out inactive or failed runs if status is present
    if "status" in runs.columns:
        runs = runs[runs["status"] == "FINISHED"]

    # Sort runs by lowest val_rmse
    runs_sorted = runs.sort_values(by="metrics.val_rmse", ascending=True)

    # Print comparison table in markdown format
    print("\n### Experiment Comparison Table ###\n")
    headers = [
        "Run ID",
        "Run Name",
        "Model Type",
        "Train RMSE",
        "Val RMSE",
        "Val MAE",
        "Val R²",
        "Fit Time (s)",
    ]
    print(
        "| "
        + " | ".join(headers)
        + " |"
    )
    print("|" + "|".join([" :---: " for _ in headers]) + "|")

    for _, row in runs_sorted.iterrows():
        run_id = row.get("run_id", "N/A")[:8]
        run_name = row.get("tags.mlflow.runName", "N/A")
        model_type = row.get("params.model_type", "N/A")
        train_rmse = f"{row.get('metrics.train_rmse', 0.0):.2f}"
        val_rmse = f"{row.get('metrics.val_rmse', 0.0):.2f}"
        val_mae = f"{row.get('metrics.val_mae', 0.0):.2f}"
        val_r2 = f"{row.get('metrics.val_r2', 0.0):.4f}"
        fit_time = f"{row.get('metrics.fit_time_seconds', 0.0):.2f}"

        print(
            f"| {run_id} | {run_name} | {model_type} | {train_rmse} | {val_rmse} | {val_mae} | {val_r2} | {fit_time} |"
        )
    print("\n")

    # Select champion (top sorted row)
    champion = runs_sorted.iloc[0]
    champion_id = champion["run_id"]
    champion_name = champion.get("tags.mlflow.runName", "N/A")
    champion_type = champion.get("params.model_type", "N/A")
    champion_val_rmse = champion["metrics.val_rmse"]

    logger.info(
        "Champion Selected: Run ID %s (%s, Type: %s) with Val RMSE: %.2f",
        champion_id,
        champion_name,
        champion_type,
        champion_val_rmse,
    )

    # Download joblib artifact from MLflow
    logger.info("Fetching champion model.joblib artifact...")
    client = mlflow.tracking.MlflowClient()
    
    # Create production model dir if missing
    production_model_dir.mkdir(parents=True, exist_ok=True)
    dest_path = production_model_dir / "production_model.joblib"

    # Download model.joblib artifact
    local_path = client.download_artifacts(champion_id, "model.joblib")
    
    # Copy file to output directory
    shutil.copy2(local_path, dest_path)
    logger.info("Hot-swapped best model to %s", dest_path)

    # Write a small metadata file describing the active model
    metadata_path = production_model_dir / "production_model_metadata.json"
    import json
    metadata = {
        "run_id": champion_id,
        "run_name": champion_name,
        "model_type": champion_type,
        "metrics": {
            "val_rmse": float(champion["metrics.val_rmse"]),
            "val_mae": float(champion["metrics.val_mae"]),
            "val_r2": float(champion["metrics.val_r2"]),
        },
        "params": {
            k.replace("params.", ""): v
            for k, v in champion.items()
            if isinstance(k, str) and k.startswith("params.") and pd.notna(v)
        },
        "updated_at": champion.get("start_time", "").strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(champion.get("start_time"), pd.Timestamp)
        else str(champion.get("start_time")),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved active model metadata to %s", metadata_path)


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="Compare MLflow runs and select the best model.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ride-eta-prediction",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--production-model-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save the selected active champion model.",
    )

    args = parser.parse_args()
    compare_and_select(args.experiment_name, args.production_model_dir)


if __name__ == "__main__":
    main()
