# ml-engg-ride-eta-prediction

Machine-learning project for predicting NYC taxi trip duration from trip,
distance, time, vendor, and service features.

## Repository workflow

```text
raw NYC.csv -> preprocessing -> feature store -> model training -> best model
                                                               |
                                                        Docker FastAPI API
```

The repository supports two alternative training workflows. Choose either the
manual commands or the DVC pipeline. Docker is used after training to serve
`best_model.pkl`.

## Installation

From the repository root, create a virtual environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The input file must be available at `data/raw/NYC.csv`.

## Option A: manual training

Run every stage explicitly, in this order:

```powershell
python preprocessing/preprocessing.py
python features/build_features.py
python models/train_linear_regression.py
python models/train_advanced_models.py
python models/evaluate_and_select_best.py
```

The final stage creates `best_model.pkl`, `model_comparison.csv`,
`feature_importance.csv`, and `top_1000_prediction_errors.csv`.

Manual execution does not use DVC to determine which stages need rerunning.

## Option B: DVC pipeline

`dvc.yaml` defines the same workflow as a reproducible dependency graph:

```text
raw CSV -> preprocessing -> feature store -> model training -> best-model selection
```

Initialize DVC once per clone, then reproduce the pipeline:

```powershell
dvc init
dvc repro
dvc status
```

If `.dvc/config` already exists, skip `dvc init`. After a successful run, Git
should track the DVC metadata:

```powershell
git add .gitignore feature_store/.gitignore data/processed/.gitignore dvc.lock
git commit -m "Track DVC pipeline state"
```

For shared data and artifacts, add the raw dataset and configure a DVC remote:

```powershell
dvc add data/raw/NYC.csv
dvc remote add -d storage <remote-url>
dvc push
```

After code or data changes, `dvc repro` reruns only affected stages. The
preprocessing transformer at `model_store/preprocessor.pkl` remains Git-managed
in this repository; processed data and trained model artifacts are DVC outputs.

## First-time Docker deployment

The FastAPI service in `deploy/api.py` loads `best_model.pkl` and exposes
`/health`, `/predict`, and `/docs`. Complete one of the training workflows above
before building the image.

### 1. Install and start Docker Desktop

Install Docker Desktop for Windows, start it, and select Linux containers.
Open a new PowerShell window as the same Windows user that runs Docker Desktop.

Verify that the Docker engine is reachable:

```powershell
docker version
docker info
```

If `docker info` cannot connect, restart Docker Desktop:

```powershell
docker desktop status
docker desktop restart
docker desktop engine use linux
```

Run `docker info` again before continuing.

### 2. Build and start the API

From the repository root:

```powershell
docker compose up --build
```

The first build downloads the base image and Python dependencies and may take
several minutes. Keep this terminal running while using the service.

To run the container in the background instead:

```powershell
docker compose up --build -d
docker compose ps
```

### 3. Verify the service

In a second PowerShell window:

```powershell
curl.exe http://localhost:8000/health
```

Open interactive API documentation at <http://localhost:8000/docs>.

### 4. Send a prediction

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"passenger_count":1,"distance_km":5.2,"pickup_hour":18,"pickup_day":15,"pickup_month":1,"pickup_weekday":3,"rush_hour":1,"is_weekend":0,"vendor_id":1,"store_and_fwd_flag":"N"}'
```

The response contains `trip_duration_seconds` and `trip_duration_minutes`.

### 5. Stop the service

Press `Ctrl+C` in the foreground terminal, or run:

```powershell
docker compose down
```

The image contains only the serving code, dependencies, and selected model.
Training data and the DVC cache are excluded through `.dockerignore`.

## Testing and Playwright mode

Run the local API contract tests from the repository root without Docker or a
trained model:

```powershell
pytest -m "not e2e"
```

The browser smoke test uses Playwright and checks that the FastAPI Swagger UI
loads from a running service. Install the browser once:

```powershell
playwright install chromium
```

Start the API in one terminal:

```powershell
docker compose up --build -d
```

Run the Playwright test in another terminal:

```powershell
pytest -m e2e --browser chromium --base-url http://127.0.0.1:8000
```

If the API is not running, the browser test is reported as skipped with the
Docker startup command rather than as an unexplained connection failure.

Run the complete suite with:

```powershell
pytest
```

The `e2e` marker keeps browser tests separate from fast unit/API tests. Stop
the deployment after testing with `docker compose down`.

## Exploration and MLflow

Run exploratory analysis independently:

```powershell
python EDA/eda_nyc_taxi.py
```

After training, open the local MLflow UI with:

```powershell
mlflow ui
```

XGBoost models are logged with MLflow's XGBoost flavor. This avoids sklearn
trusted-type errors when serializing `XGBRegressor` artifacts.

## Logging

Feature-store creation, training, evaluation, model selection, MLflow tracking,
and API inference emit standard Python logger messages at `INFO` level.
Unexpected failures are logged at `ERROR` level with a traceback before being
re-raised. The log format includes a timestamp, severity, module, and message.

## License

This project is provided under the terms of the included `LICENSE` file.
