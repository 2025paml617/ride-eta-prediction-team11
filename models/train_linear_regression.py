"""
Train baseline Linear Regression model directly from SQLite feature store.
Saves model and metadata as a pickle file.
"""

from pathlib import Path
import sqlite3
import sys
import warnings
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_HELPER_DIR = PROJECT_ROOT / "mlflow"
if str(MLFLOW_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(MLFLOW_HELPER_DIR))

from mlflow_tracker import MLflowTracker


warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "feature_store" / "feature_store.db"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "num__passenger_count",
    "num__distance_km",
    "num__pickup_hour",
    "num__pickup_day",
    "num__pickup_month",
    "num__pickup_weekday",
    "num__rush_hour",
    "num__is_weekend",
    "cat__vendor_id_1",
    "cat__vendor_id_2",
    "cat__store_and_fwd_flag_N",
    "cat__store_and_fwd_flag_Y",
]
TARGET = "trip_duration"


def load_features_from_store():
    if not DB_PATH.exists():
        logger.error("Feature store not found: %s", DB_PATH)
        raise FileNotFoundError("feature_store.db was not found.")

    columns = ", ".join(FEATURE_COLUMNS + [TARGET])
    query = f"SELECT {columns} FROM feature_store"

    with sqlite3.connect(DB_PATH) as conn:
        data = pd.read_sql_query(query, conn)

    logger.info("Loaded %d rows for linear regression", len(data))
    return data[FEATURE_COLUMNS], data[TARGET]


def main():
    tracker = MLflowTracker()
    logger.info("Starting Linear Regression training")

    print("Loading data for Linear Regression...")
    X, y = load_features_from_store()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Training Linear Regression...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    logger.info("Linear Regression training completed")

    predictions = linear_model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }

    params = {"model_type": "LinearRegression", "test_size": TEST_SIZE, "random_state": RANDOM_STATE}

    joblib.dump(linear_model, "linear_regression_model.pkl")
    logger.info("Saved linear_regression_model.pkl")
    print("Saved: linear_regression_model.pkl")

    tracker.log_run(
        run_name="Linear_Regression_Baseline",
        model=linear_model,
        params=params,
        metrics=metrics,
        flavor="sklearn",
        artifacts=["linear_regression_model.pkl"],
    )
    print("Logged run to MLflow.")
    logger.info("Logged Linear Regression run to MLflow")

    output_data = {
        "model": linear_model,
        "name": "Linear Regression",
        "metrics": metrics,
        "predictions": predictions,
        "X_test": X_test,
        "y_test": y_test,
    }

    joblib.dump(output_data, "linear_regression.pkl")
    logger.info("Saved linear_regression.pkl")
    print("Saved: linear_regression.pkl\n")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Linear Regression pipeline failed")
        raise
