"""
Train Random Forest, XGBoost, and perform hyperparameter tuning.
Saves model pickle files and hyperparameter search tuning object.
"""

from pathlib import Path
import sqlite3
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("Install XGBoost with: pip install xgboost") from exc

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "feature_store" / "feature_store.db"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TUNING_SAMPLE_SIZE = 100_000

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
    """Load model features and target from SQLite."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "feature_store.db was not found. Run build_features.py first."
        )

    columns = ", ".join(FEATURE_COLUMNS + [TARGET])
    query = f"SELECT {columns} FROM feature_store"

    with sqlite3.connect(DB_PATH) as conn:
        data = pd.read_sql_query(query, conn)

    return data[FEATURE_COLUMNS], data[TARGET]


def compute_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": (mean_squared_error(y_true, predictions)) ** 0.5,
        "R2": r2_score(y_true, predictions),
    }


def main():
    X, y = load_features_from_store()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ---------------------------------------------------------------
    # 1. Random Forest
    # ---------------------------------------------------------------
    print("\n--- 2. Random Forest ---")
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_scores = compute_metrics(y_test, rf_preds)
    print(f"Random Forest: MAE={rf_scores['MAE']:.4f}, RMSE={rf_scores['RMSE']:.4f}, R2={rf_scores['R2']:.4f}")

    joblib.dump(
        {
            "model": rf_model,
            "name": "Random Forest",
            "metrics": rf_scores,
            "predictions": rf_preds,
        },
        "random_forest.pkl",
    )
    print("Saved: random_forest.pkl")

    # ---------------------------------------------------------------
    # 2. XGBoost Baseline
    # ---------------------------------------------------------------
    print("\n--- 3. XGBoost Baseline ---")
    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_scores = compute_metrics(y_test, xgb_preds)
    print(f"XGBoost: MAE={xgb_scores['MAE']:.4f}, RMSE={xgb_scores['RMSE']:.4f}, R2={xgb_scores['R2']:.4f}")

    joblib.dump(
        {
            "model": xgb_model,
            "name": "XGBoost",
            "metrics": xgb_scores,
            "predictions": xgb_preds,
        },
        "xgboost.pkl",
    )
    print("Saved: xgboost.pkl")

    # ---------------------------------------------------------------
    # 3. XGBoost Hyperparameter Tuning
    # ---------------------------------------------------------------
    print("\n--- 4. Tuning XGBoost ---")
    tuning_size = min(TUNING_SAMPLE_SIZE, len(X_train))
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train, train_size=tuning_size, random_state=RANDOM_STATE
    )

    parameter_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.08, 0.10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
    }

    tuning_model = XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1
    )

    search = RandomizedSearchCV(
        tuning_model,
        parameter_grid,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_tune, y_tune)

    # Save RandomizedSearchCV search object
    joblib.dump(search, "xgboost_tuning_search.pkl")
    print("Saved Hyperparameter Tuning Object: xgboost_tuning_search.pkl")

    print("\nBest Parameters found:", search.best_params_)

    # Retrain tuned model on full training data
    tuned_model = XGBRegressor(
        **search.best_params_,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    tuned_model.fit(X_train, y_train)
    tuned_preds = tuned_model.predict(X_test)
    tuned_scores = compute_metrics(y_test, tuned_preds)

    print(f"Tuned XGBoost: MAE={tuned_scores['MAE']:.4f}, RMSE={tuned_scores['RMSE']:.4f}, R2={tuned_scores['R2']:.4f}")

    joblib.dump(
        {
            "model": tuned_model,
            "name": "Tuned XGBoost",
            "metrics": tuned_scores,
            "predictions": tuned_preds,
        },
        "tuned_xgboost.pkl",
    )
    print("Saved: tuned_xgboost.pkl\n")


if __name__ == "__main__":
    main()