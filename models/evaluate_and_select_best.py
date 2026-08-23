"""
Loads trained pickle models, compares test performance, extracts feature
importance, performs error analysis, and exports the top-performing model as best_model.pkl.
"""

from pathlib import Path
import joblib
import pandas as pd

from mlflow_tracker import MLflowTracker

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


def load_pickle(path):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required file '{path}' not found. Run previous training scripts first."
        )
    return joblib.load(path)


def main():
    tracker = MLflowTracker()
    print("--- Performance Evaluation & Model Selection ---")

    # Load artifacts and baseline dataset reference
    linear_data = load_pickle("linear_regression.pkl")
    rf_data = load_pickle("random_forest.pkl")
    xgb_data = load_pickle("xgboost.pkl")
    tuned_xgb_data = load_pickle("tuned_xgboost.pkl")

    all_runs = [linear_data, rf_data, xgb_data, tuned_xgb_data]

    # 1. Compare Metrics
    results = []
    for item in all_runs:
        results.append({"Model": item["name"], **item["metrics"]})

    comparison = pd.DataFrame(results).sort_values("RMSE")
    comparison.to_csv("model_comparison.csv", index=False)

    print("\nModel Comparison Table:")
    print(comparison.to_string(index=False))

    # 2. Select Best Fit Model
    best_run = min(all_runs, key=lambda x: x["metrics"]["RMSE"])
    best_model = best_run["model"]
    best_name = best_run["name"]

    joblib.dump(best_model, "best_model.pkl")

    print(f"\nSelected Best Model: {best_name}")
    print("Selection Criterion: Lowest Test RMSE")
    print("Exported Best Model: best_model.pkl")

    tracker.log_summary_run(
        run_name="Model_Selection_Summary",
        best_model_name=best_name,
        best_metrics=best_run,
        best_model=best_model,
        artifacts=best_run.get("artifacts", []),
    )
    print("Logged selection summary to MLflow.")
    
    # 3. Feature Importance Analysis
    if hasattr(best_model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": best_model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        importance.to_csv("feature_importance.csv", index=False)
        print("\nFeature Importance:")
        print(importance.to_string(index=False))

    # 4. Error Analysis
    X_test = linear_data["X_test"]
    y_test = linear_data["y_test"]

    errors = X_test.copy()
    errors["Actual"] = y_test.values
    errors["Predicted"] = best_run["predictions"]
    errors["Error"] = errors["Actual"] - errors["Predicted"]
    errors["Absolute_Error"] = errors["Error"].abs()

    errors.sort_values("Absolute_Error", ascending=False).head(1000).to_csv(
        "top_1000_prediction_errors.csv", index=False
    )
    print("\nSaved Top 1000 errors to: top_1000_prediction_errors.csv\n")


if __name__ == "__main__":
    main()