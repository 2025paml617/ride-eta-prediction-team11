import pandas as pd
import numpy as np
import joblib

from math import radians, sin, cos, sqrt, atan2

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================
# Load Dataset
# ============================================================

DATA_PATH = "NYC.csv"

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nOriginal Dataset Shape :", df.shape)
print("\nFirst 5 Rows")
print(df.head())

# ============================================================
# Schema Validation
# ============================================================

required_columns = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]

print("\nChecking Required Columns...")

missing_cols = [col for col in required_columns if col not in df.columns]

if len(missing_cols) > 0:
    raise Exception(f"Missing Columns : {missing_cols}")

print("Schema Validation Successful")

# ============================================================
# Remove Duplicate Rows
# ============================================================

df = df.drop_duplicates()

# ============================================================
# Convert Datetime
# ============================================================

df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(subset=["pickup_datetime"])

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

OUTPUT_FILE = "NYC_Preprocessed.csv"

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
    "preprocessor.pkl"
)

print("Preprocessing Pipeline Saved : preprocessor.pkl")

print("\nProcessing Completed Successfully.")