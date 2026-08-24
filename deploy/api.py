"""FastAPI service for serving the selected NYC taxi duration model."""

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "best_model.pkl"))
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

app = FastAPI(title="NYC Taxi Duration API", version="1.0.0")


class PredictionRequest(BaseModel):
    passenger_count: float = Field(ge=1, le=8)
    distance_km: float = Field(gt=0, le=200)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_day: int = Field(ge=1, le=31)
    pickup_month: int = Field(ge=1, le=12)
    pickup_weekday: int = Field(ge=0, le=6)
    rush_hour: int = Field(ge=0, le=1)
    is_weekend: int = Field(ge=0, le=1)
    vendor_id: int = Field(ge=1, le=2)
    store_and_fwd_flag: str = Field(pattern="^[NY]$")


def load_model():
    if not MODEL_PATH.exists():
        logger.error("Model artifact not found: %s", MODEL_PATH)
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    logger.info("Loading model from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


model = None
try:
    model = load_model()
except FileNotFoundError:
    logger.warning("API started without a model; /predict will return 503")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    vendor_1 = int(request.vendor_id == 1)
    vendor_2 = int(request.vendor_id == 2)
    flag_n = int(request.store_and_fwd_flag == "N")
    flag_y = int(request.store_and_fwd_flag == "Y")
    features = pd.DataFrame([{
        "num__passenger_count": request.passenger_count,
        "num__distance_km": request.distance_km,
        "num__pickup_hour": request.pickup_hour,
        "num__pickup_day": request.pickup_day,
        "num__pickup_month": request.pickup_month,
        "num__pickup_weekday": request.pickup_weekday,
        "num__rush_hour": request.rush_hour,
        "num__is_weekend": request.is_weekend,
        "cat__vendor_id_1": vendor_1,
        "cat__vendor_id_2": vendor_2,
        "cat__store_and_fwd_flag_N": flag_n,
        "cat__store_and_fwd_flag_Y": flag_y,
    }])[FEATURE_COLUMNS]

    try:
        seconds = max(0.0, float(model.predict(features)[0]))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    logger.info("Prediction generated: %.2f seconds", seconds)
    return {"trip_duration_seconds": seconds, "trip_duration_minutes": round(seconds / 60, 2)}
