import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
import sys

from math import radians, sin, cos, sqrt, atan2

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Make imports work both from the repository root and from any working
# directory when this file is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import load_and_validate_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Load Dataset
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "NYC.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "weather.csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "validation_report.json"

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

validation_result = load_and_validate_data(str(DATA_PATH))
df = validation_result.data

VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
VALIDATION_REPORT_PATH.write_text(
    json.dumps(validation_result.report, indent=2), encoding="utf-8"
)
logger.info("Validation report written to %s", VALIDATION_REPORT_PATH)

print("\nOriginal Dataset Shape :", df.shape)
print("\nFirst 5 Rows")
print(df.head())

# ============================================================
# Remove Duplicate Rows
# ============================================================

df = df.drop_duplicates()

# ============================================================
# Convert Datetime
# ============================================================

# Validation has already parsed both pickup and dropoff timestamps and removed
# missing, malformed, and chronologically impossible records.


def add_weather_features(dataframe):
    """Left-join optional hourly weather observations by pickup time."""
    weather_features = [
        "temperature_c", "precipitation_mm", "wind_speed_kmh", "visibility_km"
    ]
    result = dataframe.copy()
    if not WEATHER_PATH.exists():
        for column in weather_features:
            result[column] = np.nan
        result["weather_data_available"] = 0
        logger.warning("No weather.csv found; weather features will be median-imputed")
        return result

    weather = pd.read_csv(WEATHER_PATH)
    # Accept the supplied NYC weather export as well as the normalized
    # contract documented in data/raw/README.md.
    weather_aliases = {
        "pickup_datetime": "weather_datetime",
        "tempm": "temperature_c",
        "precipm": "precipitation_mm",
        "wspdm": "wind_speed_kmh",
        "vism": "visibility_km",
    }
    weather = weather.rename(columns=weather_aliases)
    if "weather_datetime" not in weather.columns:
        raise ValueError(
            "weather.csv must contain weather_datetime or pickup_datetime"
        )
    weather["weather_datetime"] = pd.to_datetime(
        weather["weather_datetime"], errors="coerce", utc=True
    ).dt.floor("h")
    weather = weather.dropna(subset=["weather_datetime"])
    for column in weather_features:
        if column not in weather.columns:
            weather[column] = np.nan
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    # The supplied file has several observations per hour. Mean aggregation
    # prevents a many-to-many join and creates one deterministic row per hour.
    weather = weather.groupby("weather_datetime", as_index=False)[weather_features].mean()

    result["pickup_hour_timestamp"] = result["pickup_datetime"].dt.floor("h")
    result = result.merge(
        weather[["weather_datetime"] + weather_features],
        left_on="pickup_hour_timestamp",
        right_on="weather_datetime",
        how="left",
    ).drop(columns=["pickup_hour_timestamp", "weather_datetime"])
    result["weather_data_available"] = result[weather_features].notna().any(axis=1).astype(int)
    return result


df = add_weather_features(df)

# ============================================================
# Feature Engineering
# ============================================================

print("\nCreating Time Features...")

df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_day"] = df["pickup_datetime"].dt.day
df["pickup_month"] = df["pickup_datetime"].dt.month
df["pickup_weekday"] = df["pickup_datetime"].dt.dayofweek

df["is_weekend"] = (
    df["pickup_weekday"] >= 5
).astype(int)

df["rush_hour"] = (
    ((df["pickup_hour"] >= 7) & (df["pickup_hour"] <= 10))
    |
    ((df["pickup_hour"] >= 16) & (df["pickup_hour"] <= 20))
).astype(int)

# ============================================================
# Haversine Distance
# ============================================================

print("Calculating Distance...")


def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def remove_iqr_outliers(dataframe, column, multiplier=1.5):
    """Remove rows with values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
    if column not in dataframe.columns:
        return dataframe

    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    before = len(dataframe)
    dataframe = dataframe[dataframe[column].between(lower, upper)].copy()

    print(
        f"{column}: removed {before - len(dataframe):,} outliers "
        f"(valid range: {lower:.4f} to {upper:.4f})"
    )

    return dataframe


df["distance_km"] = df.apply(
    lambda row: haversine(
        row["pickup_latitude"],
        row["pickup_longitude"],
        row["dropoff_latitude"],
        row["dropoff_longitude"],
    ),
    axis=1,
)

# ============================================================
# Remove Invalid Distance
# ============================================================

df = df[df["distance_km"] > 0]

# ============================================================
# Target Column
# ============================================================

target_column = None

possible_targets = [
    "trip_duration",
    "duration",
    "ETA",
]

for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

# ============================================================
# Identify and Handle Outliers
# ============================================================

print("\nIdentifying and handling outliers...")

before_outliers = len(df)
for outlier_column in ["distance_km", "passenger_count"]:
    df = remove_iqr_outliers(df, outlier_column)

if target_column:
    df = remove_iqr_outliers(df, target_column)

print(
    f"Total rows removed by outlier filtering: {before_outliers - len(df):,}"
)

if target_column:
    y = df[target_column]
    X = df.drop(columns=[target_column])
else:
    X = df.copy()

# ============================================================
# Feature Lists
# ============================================================

numeric_features = [
    "passenger_count",
    "distance_km",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "pickup_weekday",
    "rush_hour",
    "is_weekend",
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kmh",
    "visibility_km",
    "weather_data_available",
]

categorical_features = []

if "vendor_id" in X.columns:
    categorical_features.append("vendor_id")

if "store_and_fwd_flag" in X.columns:
    categorical_features.append("store_and_fwd_flag")

# ============================================================
# Pipelines
# ============================================================

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

print("\nApplying Preprocessing Pipeline...")

processed = preprocessor.fit_transform(X)

feature_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    processed,
    columns=feature_names
)

# ============================================================
# Add Target Column Back
# ============================================================

if target_column:
    processed_df[target_column] = y.values

# ============================================================
# Display Results
# ============================================================

print("\n")
print("=" * 60)
print("PREPROCESSED DATA")
print("=" * 60)

print(processed_df.head(10))

print("\nProcessed Shape :", processed_df.shape)

print("\nColumns")
print(processed_df.columns.tolist())

print("\nMissing Values")
print(processed_df.isnull().sum())

print("\nData Types")
print(processed_df.dtypes)

print("\nStatistics")
print(processed_df.describe())

# ============================================================
# Save CSV
# ============================================================

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "NYC_Preprocessed.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

processed_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPreprocessed dataset saved as :", OUTPUT_FILE)

# ============================================================
# Save Pipeline
# ============================================================

joblib.dump(
    preprocessor,
    PROJECT_ROOT / "model_store" / "preprocessor.pkl"
)

print("Preprocessing Pipeline Saved : preprocessor.pkl")

print("\nProcessing Completed Successfully.")
