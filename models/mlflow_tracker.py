"""
MLflow Tracker Helper Module.
"""

from pathlib import Path
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost

logger = logging.getLogger(__name__)


class MLflowTracker:
    def __init__(self, experiment_name: str = "NYC_Taxi_Trip_Duration"):
        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        run_name: str,
        model,
        params: dict,
        metrics: dict,
        flavor: str = "sklearn",
        artifacts: list = None
    ):
        """Log a training run to MLflow."""
        logger.info("Logging MLflow run '%s' for model %s", run_name, type(model).__name__)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)

            if metrics:
                mlflow.log_metrics(metrics)

            if flavor == "xgboost":
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model")

            logger.info("MLflow run '%s' logged successfully", run_name)

            if artifacts:
                for file_path in artifacts:
                    if Path(file_path).exists():
                        mlflow.log_artifact(file_path)

    def log_summary_run(
        self,
        run_name: str,
        best_model_name: str,
        best_metrics: dict,
        best_model,
        artifacts: list
    ):
        """Log final evaluation and best model summary to MLflow."""
        logger.info("Logging selected model '%s' to MLflow", best_model_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("winning_model", best_model_name)
            mlflow.log_metrics(best_metrics)

            for file_path in artifacts:
                if Path(file_path).exists():
                    mlflow.log_artifact(file_path)

            # XGBRegressor contains xgboost-specific Booster state and must be
            # serialized with MLflow's XGBoost flavor. Using the sklearn flavor
            # triggers skops trusted-type validation for these internals.
            is_xgboost_model = (
                type(best_model).__module__.startswith("xgboost")
                or type(best_model).__name__.startswith("XGB")
            )
            if is_xgboost_model:
                mlflow.xgboost.log_model(best_model, name="selected_best_model")
            else:
                mlflow.sklearn.log_model(best_model, name="selected_best_model")
            logger.info("Selected model '%s' logged successfully", best_model_name)
