"""
NYC Taxi Trip Duration - Model Training, Tuning and Evaluation

Tasks covered:
1. Baseline Linear Regression
2. Random Forest
3. XGBoost
4. Hyperparameter tuning
5. MLflow experiment tracking
6. Model metric comparison
7. Feature importance
8. Error analysis
9. Best-model selection

Input:
    NYC_Preprocessed.csv

The input is already preprocessed, so this script does not repeat encoding/scaling.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore") 


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "NYC_Preprocessed.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Tuning on a sample keeps the search practical for a large dataset.
# The final tuned model is then evaluated on the complete test set.
TUNING_SAMPLE_SIZE = 50_000

# Enable MLflow, but keep the workload small enough to avoid memory-heavy model
# artifact serialization. Only the best final model is logged to reduce Zip memory use.
USE_MLFLOW = True
MLFLOW_EXPERIMENT = "NYC_Taxi_Trip_Duration"
MLFLOW_LOG_ONLY_BEST_MODEL = True


# ---------------------------------------------------------------------------
# Optional MLflow setup
# ---------------------------------------------------------------------------
mlflow = None

if USE_MLFLOW:
    try:
        import mlflow
        import mlflow.sklearn
        import mlflow.xgboost
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except ImportError:
        print("MLflow is not installed. Continuing without MLflow tracking.")
        mlflow = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    """Return the main regression evaluation metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_model(name, model, X_test, y_test, results):
    """Generate predictions, calculate metrics and store the results."""
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    results.append({
        "Model": name,
        **metrics
    })

    print(f"\n{name}")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"R2   : {metrics['R2']:.4f}")

    return predictions, metrics


def log_model_to_mlflow(name, model, metrics, params=None):
    """Log model, parameters and metrics to MLflow when available."""
    if mlflow is None:
        return

    if MLFLOW_LOG_ONLY_BEST_MODEL and name != "Best_Model":
        return

    with mlflow.start_run(run_name=name):
        if params:
            # MLflow parameter values must be strings/numeric scalar values.
            safe_params = {
                key: str(value) for key, value in params.items()
            }
            mlflow.log_params(safe_params)

        mlflow.log_metrics(metrics)
        try:
            if type(model).__module__.startswith("xgboost") or type(model).__name__.startswith("XGB"):
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model")
        except Exception as exc:
            print(
                f"Warning: MLflow model logging failed for '{name}': "
                f"{type(exc).__name__}: {exc}"
            )


# ---------------------------------------------------------------------------
# 1. Load preprocessed data
# ---------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find '{DATA_PATH}'. "
        "Place the CSV file in the same folder as this script."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))


# ---------------------------------------------------------------------------
# 2. Separate features and target
# ---------------------------------------------------------------------------
TARGET = "trip_duration"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in the dataset.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Remove rows containing missing/infinite values before model training.
valid_rows = X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
valid_rows &= y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]

print("Rows used for modelling:", len(X))


# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ---------------------------------------------------------------------------
# 4. Baseline Linear Regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. BASELINE LINEAR REGRESSION")
print("=" * 70)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

results = []

linear_predictions, linear_metrics = evaluate_model(
    "Linear Regression",
    linear_model,
    X_test,
    y_test,
    results
)

log_model_to_mlflow(
    "Linear_Regression",
    linear_model,
    linear_metrics
)


# ---------------------------------------------------------------------------
# 5. Random Forest
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions, rf_metrics = evaluate_model(
    "Random Forest",
    rf_model,
    X_test,
    y_test,
    results
)

log_model_to_mlflow(
    "Random_Forest",
    rf_model,
    rf_metrics,
    rf_model.get_params()
)


# ---------------------------------------------------------------------------
# 6. XGBoost
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. XGBOOST")
print("=" * 70)

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "XGBoost is required for this script. Install it with:\n"
        "pip install xgboost"
    ) from exc

xgb_model = XGBRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_predictions, xgb_metrics = evaluate_model(
    "XGBoost",
    xgb_model,
    X_test,
    y_test,
    results
)

log_model_to_mlflow(
    "XGBoost",
    xgb_model,
    xgb_metrics,
    xgb_model.get_params()
)


# ---------------------------------------------------------------------------
# 7. Hyperparameter tuning - XGBoost
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. HYPERPARAMETER TUNING - XGBOOST")
print("=" * 70)

# Use a representative training sample for tuning because the dataset is large.
tuning_size = min(TUNING_SAMPLE_SIZE, len(X_train))

X_tune, _, y_tune, _ = train_test_split(
    X_train,
    y_train,
    train_size=tuning_size,
    random_state=RANDOM_STATE
)

print("Rows used for tuning:", len(X_tune))

param_distributions = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.08, 0.10],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
}

tuning_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=tuning_model,
    param_distributions=param_distributions,
    n_iter=5,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=1,
    verbose=1
)

search.fit(X_tune, y_tune)

print("\nBest XGBoost parameters:")
print(search.best_params_)
print(f"Best CV RMSE: {-search.best_score_:.4f}")


# Train the selected hyperparameters on the complete training set.
tuned_xgb = XGBRegressor(
    **search.best_params_,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

tuned_xgb.fit(X_train, y_train)

tuned_predictions, tuned_metrics = evaluate_model(
    "Tuned XGBoost",
    tuned_xgb,
    X_test,
    y_test,
    results
)

log_model_to_mlflow(
    "Tuned_XGBoost",
    tuned_xgb,
    tuned_metrics,
    search.best_params_
)


# ---------------------------------------------------------------------------
# 8. Compare model metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE").reset_index(drop=True)

print(results_df.to_string(index=False))

results_df.to_csv("models\\model_comparison.csv", index=False)


# ---------------------------------------------------------------------------
# 9. Feature importance analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. FEATURE IMPORTANCE")
print("=" * 70)

# Tree-based models provide feature_importances_.
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": tuned_xgb.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.to_string(index=False))

importance.to_csv("models\\feature_importance.csv", index=False)


# ---------------------------------------------------------------------------
# 10. Error analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("7. ERROR ANALYSIS - TUNED XGBOOST")
print("=" * 70)

error_analysis = X_test.copy()
error_analysis["Actual"] = y_test.values
error_analysis["Predicted"] = tuned_predictions
error_analysis["Error"] = (
    error_analysis["Actual"] - error_analysis["Predicted"]
)
error_analysis["Absolute_Error"] = error_analysis["Error"].abs()

print("\nError summary:")
print(error_analysis["Error"].describe())

print("\nLargest prediction errors:")
print(
    error_analysis[
        ["Actual", "Predicted", "Error", "Absolute_Error"]
    ]
    .sort_values("Absolute_Error", ascending=False)
    .head(10)
    .to_string(index=False)
)

# Save a sample of error records rather than writing the entire test set.
error_analysis.sort_values(
    "Absolute_Error", ascending=False
).head(1000).to_csv(
    "models\\top_1000_prediction_errors.csv",
    index=False
)


# ---------------------------------------------------------------------------
# 11. Select the best model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("8. BEST MODEL SELECTION")
print("=" * 70)

# For regression, lower RMSE is better. R2 is used as supporting evidence.
best_row = results_df.iloc[0]
best_model_name = best_row["Model"]

print(f"Best model: {best_model_name}")
print(f"RMSE: {best_row['RMSE']:.4f}")
print(f"MAE : {best_row['MAE']:.4f}")
print(f"R2  : {best_row['R2']:.4f}")

print(
    "\nJustification: the model is selected primarily using the lowest "
    "test RMSE, because RMSE penalizes large prediction errors. "
    "MAE and R2 are also reported to provide additional evidence."
)


# ---------------------------------------------------------------------------
# 12. Save the selected model
# ---------------------------------------------------------------------------
if best_model_name == "Linear Regression":
    best_model = linear_model
elif best_model_name == "Random Forest":
    best_model = rf_model
else:
    best_model = tuned_xgb

best_model_params = getattr(best_model, "get_params", lambda: {})()
log_model_to_mlflow(
    "Best_Model",
    best_model,
    {
        "RMSE": float(best_row["RMSE"]),
        "MAE": float(best_row["MAE"]),
        "R2": float(best_row["R2"]),
    },
    best_model_params
)

try:
    import joblib
    joblib.dump(best_model, "model_store\\best_model.pkl")
    print("\nSaved best model to: best_model.pkl")
except ImportError:
    print("\njoblib is not installed, so the model was not saved.")


print("\nFiles generated:")
print("- model_comparison.csv")
print("- feature_importance.csv")
print("- top_1000_prediction_errors.csv")
print("- best_model.pkl")
print("\nModel training and evaluation completed successfully.")
