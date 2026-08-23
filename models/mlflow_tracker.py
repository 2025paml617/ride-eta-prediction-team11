"""
MLflow Tracker Helper Module.
"""

from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost


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
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)

            if metrics:
                mlflow.log_metrics(metrics)

            if flavor == "xgboost":
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model")

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
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("winning_model", best_model_name)
            mlflow.log_metrics(best_metrics)

            for file_path in artifacts:
                if Path(file_path).exists():
                    mlflow.log_artifact(file_path)

            mlflow.sklearn.log_model(best_model, name="selected_best_model")