# ml-engg-ride-eta-prediction

A small machine learning project for predicting taxi ride estimated time of arrival (ETA) using trip and location features.

## Overview

This repository contains a dataset and starter information for building a ride ETA prediction model. The goal is to explore features, train a regression model, and evaluate performance in predicting trip duration or arrival time.

## Contents

- `NYC_Taxi_Trip_Duration_dataset.csv` — sample taxi trip data used for model training and evaluation.
- `README.md` — project overview and usage instructions.
- `LICENSE` — project license information.

## Dataset

The dataset includes typical taxi trip fields such as pickup and dropoff times, location coordinates, passenger count, and trip duration. Use this data to engineer features like distance, time of day, day of week, and traffic patterns.

## Suggested Workflow

1. Load the dataset into a data analysis environment.
2. Clean and preprocess the data:
   - parse timestamps
   - remove invalid or missing values
   - filter out outliers
3. Create feature columns:
   - pickup/dropoff latitude and longitude
   - distance metrics (Haversine, Manhattan, etc.)
   - request time features (hour, weekday, month)
   - passenger count and extra service flags
4. Split the data into training and validation sets.
5. Train one or more regression models:
   - linear regression
   - decision trees / random forest
   - gradient boosting
   - neural networks
6. Evaluate using metrics such as RMSE, MAE, and R².

## Example Usage

```bash
# Example commands, adapt to your environment
python train_model.py --data NYC_Taxi_Trip_Duration_dataset.csv
python evaluate_model.py --model model.pkl --data test.csv
```

## Notes

- This repository is intended as a starting point for experimentation and model development.
- Add your own scripts, notebooks, and model artifacts as needed.
- Ensure reproducibility by tracking preprocessing steps and model hyperparameters.

## Installation and execution

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

Run the pipeline in this order:

```bash
python features/build_features.py
python models/train_linear_regression.py
python models/train_advanced_models.py
python models/evaluate_and_select_best.py
```

The evaluation step writes `model_comparison.csv`, `feature_importance.csv`,
`top_1000_prediction_errors.csv`, and `best_model.pkl`.

## Logging

The feature-store, training, evaluation, and MLflow tracking scripts use the
standard Python logger. Messages are emitted at `INFO` level by default and
include timestamps, severity, module name, and message. Unexpected pipeline
failures are logged at `ERROR` level with a traceback before being re-raised.

To see more or less detail, configure logging before importing a script, or
change `level=logging.INFO` to `logging.DEBUG` or `logging.WARNING` in the
script entry point.

XGBoost models are logged with MLflow's XGBoost flavor. This is required for
`XGBRegressor` artifacts and avoids sklearn trusted-type errors during model
serialization.

## To perform the Exploratory data analysis run the below script
python EDA/eda_nyc_taxi.py

## Data preprocessing 
python preprocessing/preprocessing.py

1. Feature Store

python features/build_features.py → creates/updates feature_store.db

NYC_Preprocessed.csv
        ↓
build_features.py
        ↓
feature_store.db
2. Model Training

model_train_from_feature_store.py reads only from feature_store.db.

feature_store.db
        ↓
model_train_from_feature_store.py
        ↓
Linear Regression
Random Forest
XGBoost
        ↓
Hyperparameter tuning
        ↓
Model comparison
        ↓
Best model
3. MLflow — completely separate

Download mlflow_tracking.py

The MLflow script runs after model training and reads the generated model results:

model_train_from_feature_store.py
              ↓
     model_comparison.csv
     feature_importance.csv
     top_1000_prediction_errors.csv
     best_model.pkl
              ↓
       mlflow_tracking.py
              ↓
          MLflow

It logs:

Linear Regression metrics
Random Forest metrics
XGBoost metrics
Tuned XGBoost metrics
MAE
RMSE
R²
Selected best model
Best model artifact
Feature importance
Error-analysis results
Execution order

Install MLflow if required:

pip install mlflow

Then:

Step 1 — Build feature store

python features/build_features.py

step 2 - create a mlflow model tracker

python models/mlflow_tracker.py

Step 2 — Train models

python models/train_linear_regression.py
python models/train_advanced_models.py

Step 3 — Track experiments

python mlflow_tracking.py

Step 4 — Open MLflow UI

mlflow ui

Then open the local MLflow address shown in the terminal.

This separation is cleaner for your MLOps project because feature engineering, model training, and experiment tracking are now three independent components.


## License

This project is provided under the terms of the included `LICENSE` file.

