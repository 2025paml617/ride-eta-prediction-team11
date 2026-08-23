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

## intall python evnvironment
python -m venv venv

## Actiate the environment
venv\Scripts\Activate.bat

## To perform the Exploratory data analysis run the below script
python EDA/eda_nyc_taxi.py

## Data preprocessing 
python preprocessing/preprocessing.py

## model training
python models/NYC_Model_Training_MLflow.py


## License

This project is provided under the terms of the included `LICENSE` file.

