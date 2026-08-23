"""Model training script with MLflow tracking.

Fits a Ridge Regression baseline and an advanced HistGradientBoostingRegressor
champion on chronologically split NYC taxi data. Tracks parameters and metrics
using MLflow, then serialises the training pipelines.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eta_model.train")

# Define features used in training
CONTINUOUS_FEATURES = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "pickup_lat_cell",
    "pickup_lon_cell",
    "dropoff_lat_cell",
    "dropoff_lon_cell",
    "haversine_km",
    "manhattan_km",
    "bearing_deg",
    "temp_c",
    "precip_mm",
    "snowfall_mm",
    "wind_kph",
]

CYCLICAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

BINARY_FEATURES = [
    "vendor_id",
    "passenger_count",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "is_us_holiday",
    "is_raining",
    "is_snowing",
    "weather_available",
    "flag_zero_passengers",
    "flag_high_passengers",
    "flag_zero_distance",
    "flag_store_and_fwd",
]

ALL_FEATURES = CONTINUOUS_FEATURES + CYCLICAL_FEATURES + BINARY_FEATURES
TARGET = "trip_duration"


def create_preprocessor() -> ColumnTransformer:
    """Create a column transformer for pipeline preprocessing."""
    # Continuous variables: Impute missing (like weather) with median and scale
    continuous_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Cyclical and binary features: Impute missing with 0 and pass through
    passthrough_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", continuous_transformer, CONTINUOUS_FEATURES),
            ("passthrough", passthrough_transformer, CYCLICAL_FEATURES + BINARY_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def evaluate_model(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> dict[str, float]:
    """Calculate and return evaluation metrics."""
    logger.info("Computing predictions and evaluating...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Train metrics
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))

    # Val metrics
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    val_mae = float(mean_absolute_error(y_val, y_val_pred))
    val_r2 = float(r2_score(y_val, y_val_pred))

    metrics = {
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2,
    }
    return metrics


def plot_and_log_importance(model: Pipeline, model_type: str, run_dir: Path) -> Path | None:
    """Generate and return feature importance or coefficient plot."""
    try:
        plt.figure(figsize=(10, 8))
        if model_type == "ridge":
            coefs = model.named_steps["regressor"].coef_
            indices = np.argsort(np.abs(coefs))
            plt.barh(range(len(indices)), coefs[indices], align="center")
            plt.yticks(range(len(indices)), [ALL_FEATURES[i] for i in indices])
            plt.title("Ridge Regression Coefficients")
        else:
            importances = model.named_steps["regressor"].feature_importances_
            indices = np.argsort(importances)
            plt.barh(range(len(indices)), importances[indices], align="center")
            plt.yticks(range(len(indices)), [ALL_FEATURES[i] for i in indices])
            plt.title("HistGradientBoosting Feature Importances")

        plt.xlabel("Importance / Value")
        plt.tight_layout()
        plot_path = run_dir / f"feature_importance_{model_type}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        logger.warning("Could not generate feature importance plot: %s", e)
        return None


def run_experiment(
    df: pd.DataFrame,
    model_type: str,
    hyperparams: dict,
    experiment_name: str,
    output_model_dir: Path,
) -> str:
    """Run model training and log metrics to MLflow."""
    mlflow.set_experiment(experiment_name)

    # Perform chronological split
    logger.info("Performing chronological split (Jan-May for train, Jun for validation)...")
    train_mask = df["pickup_datetime"] < "2016-06-01"
    val_mask = df["pickup_datetime"] >= "2016-06-01"

    X_train = df.loc[train_mask, ALL_FEATURES]
    y_train = df.loc[train_mask, TARGET]
    X_val = df.loc[val_mask, ALL_FEATURES]
    y_val = df.loc[val_mask, TARGET]

    logger.info(
        "Train set: %s rows, Validation set: %s rows", len(X_train), len(X_val)
    )

    # Initialize model pipeline
    preprocessor = create_preprocessor()
    if model_type == "ridge":
        estimator = Ridge(**hyperparams)
    elif model_type == "gbt":
        estimator = HistGradientBoostingRegressor(**hyperparams)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", estimator)
    ])

    run_name = f"{model_type}_{time.strftime('%m%d_%H%M%S')}"

    # Output run directory for artifact caching
    run_dir = output_model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Starting MLflow run: %s", run.info.run_id)

        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_params(hyperparams)

        # Fit model
        logger.info("Fitting %s model...", model_type)
        start_fit = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_fit
        logger.info("Fitting completed in %.2fs", fit_time)
        mlflow.log_metric("fit_time_seconds", fit_time)

        # Evaluate
        metrics = evaluate_model(pipeline, X_train, y_train, X_val, y_val)
        for k, v in metrics.items():
            logger.info("Metric %s: %.4f", k, v)
            mlflow.log_metric(k, v)

        # Plot & log feature importance
        plot_path = plot_and_log_importance(pipeline, model_type, run_dir)
        if plot_path and plot_path.exists():
            mlflow.log_artifact(str(plot_path))

        # Save model pipeline
        model_path = run_dir / "model.joblib"
        joblib.dump(pipeline, model_path)
        logger.info("Saved serialized pipeline to %s", model_path)
        mlflow.log_artifact(str(model_path))

        # Log scikit-learn model natively to MLflow
        try:
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="pickle",
            )
        except Exception as e:
            logger.warning("Failed to log model natively with pickle serialization: %s. Retrying with skops_trusted_types=True", e)
            try:
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                    skops_trusted_types=True,
                )
            except Exception as e2:
                logger.error("Failed to log model natively to MLflow: %s", e2)

        logger.info("MLflow run %s completed successfully", run.info.run_id)
        return run.info.run_id


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="Train and track ride ETA regression models.")
    parser.add_argument(
        "--features-path",
        type=Path,
        required=True,
        help="Path to the versioned features.csv.gz file.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ride-eta-prediction",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--model-type",
        choices=["ridge", "gbt", "both"],
        default="both",
        help="Which models to train.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularisation strength for Ridge Regression.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for Gradient Boosting Regressor.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Max iterations (trees) for Gradient Boosting Regressor.",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=31,
        help="Max leaf nodes per tree for Gradient Boosting Regressor.",
    )
    parser.add_argument(
        "--output-model-dir",
        type=Path,
        default=Path("models"),
        help="Local directory to store model checkpoints.",
    )

    args = parser.parse_args()

    logger.info("Loading processed features dataset from %s...", args.features_path)
    df = pd.read_csv(args.features_path, parse_dates=["pickup_datetime"])

    # Extract hyperparams
    ridge_params = {"alpha": args.alpha}
    gbt_params = {
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
        "max_leaf_nodes": args.max_leaf_nodes,
        "random_state": 42,
    }

    # Run selected experiments
    if args.model_type in ("ridge", "both"):
        logger.info("=== Running Ridge Regression baseline ===")
        run_experiment(df, "ridge", ridge_params, args.experiment_name, args.output_model_dir)

    if args.model_type in ("gbt", "both"):
        logger.info("=== Running HistGradientBoostingRegressor champion ===")
        run_experiment(df, "gbt", gbt_params, args.experiment_name, args.output_model_dir)


if __name__ == "__main__":
    main()
