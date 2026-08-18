"""
Exploratory Data Analysis (EDA) - NYC Taxi Trip / ETA Prediction

Input:
    NYC.csv

Output:
    EDA_Results/
        missing_value_summary.csv
        numeric_summary.csv
        correlation_matrix.csv
        EDA plots (*.png)

The script performs EDA on the raw dataset before model training.
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "NYC.csv"
OUTPUT_DIR = PROJECT_ROOT / "EDA" / "EDA_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(filename):
    """Save the current plot and close it."""
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


# ================================================================
# STEP 1 - LOAD DATASET
# ================================================================

print("=" * 80)
print("STEP 1 - LOADING NYC TAXI DATASET")
print("=" * 80)

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find '{DATA_PATH}'. "
        "Place NYC.csv in the same folder as this script."
    )

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")


# ================================================================
# STEP 2 - BASIC DATASET INFORMATION
# ================================================================

print("\n" + "=" * 80)
print("STEP 2 - BASIC DATASET INFORMATION")
print("=" * 80)

print("\nColumn Names:")
for column in df.columns:
    print(f"  - {column}")

print("\nFirst 5 Rows:")
print(df.head().to_string())

print("\nData Types:")
print(df.dtypes.to_string())

print("\nDataset Shape:")
print(df.shape)


# ================================================================
# STEP 3 - DUPLICATE ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 3 - DUPLICATE RECORD ANALYSIS")
print("=" * 80)

duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count:,}")


# ================================================================
# STEP 4 - MISSING VALUE ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 4 - MISSING VALUE ANALYSIS")
print("=" * 80)

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df) * 100).round(2)

missing_summary = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": missing_percentage
})

print(missing_summary.to_string())

missing_summary.to_csv(
    os.path.join(OUTPUT_DIR, "missing_value_summary.csv")
)

missing_plot = missing_percentage[missing_percentage > 0].sort_values(
    ascending=False
)

if not missing_plot.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(missing_plot.index, missing_plot.values)
    plt.title("Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing Values (%)")
    plt.xticks(rotation=45)
    save_plot("01_missing_values.png")


# ================================================================
# STEP 5 - NUMERICAL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("STEP 5 - NUMERICAL SUMMARY STATISTICS")
print("=" * 80)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if numeric_columns:
    numeric_summary = df[numeric_columns].describe().T
    print(numeric_summary.to_string())
    numeric_summary.to_csv(
        os.path.join(OUTPUT_DIR, "numeric_summary.csv")
    )


# ================================================================
# STEP 6 - CATEGORICAL ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 6 - CATEGORICAL COLUMN ANALYSIS")
print("=" * 80)

categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).head(20).to_string())


# ================================================================
# STEP 7 - PARSE PICKUP DATETIME
# ================================================================

print("\n" + "=" * 80)
print("STEP 7 - PARSING PICKUP DATETIME")
print("=" * 80)

if "pickup_datetime" in df.columns:
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        errors="coerce"
    )

    print(
        "Invalid/missing timestamps:",
        df["pickup_datetime"].isna().sum()
    )

    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_weekday"] = df["pickup_datetime"].dt.dayofweek
    df["is_weekend"] = (df["pickup_weekday"] >= 5).astype(int)


# ================================================================
# STEP 8 - PICKUP HOUR ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 8 - PICKUP HOUR ANALYSIS")
print("=" * 80)

if "pickup_hour" in df.columns:
    hour_counts = df["pickup_hour"].value_counts().sort_index()
    print(hour_counts.to_string())

    plt.figure(figsize=(10, 6))
    plt.bar(hour_counts.index, hour_counts.values)
    plt.title("Number of Trips by Pickup Hour")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Number of Trips")
    plt.xticks(range(24))
    save_plot("02_trips_by_pickup_hour.png")


# ================================================================
# STEP 9 - WEEKDAY ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 9 - WEEKDAY ANALYSIS")
print("=" * 80)

if "pickup_weekday" in df.columns:
    weekday_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    weekday_counts = df["pickup_weekday"].value_counts().sort_index()

    for day_number, count in weekday_counts.items():
        if 0 <= day_number <= 6:
            print(f"{weekday_names[day_number]}: {count:,}")

    plt.figure(figsize=(10, 6))
    plt.bar(
        [weekday_names[i] for i in weekday_counts.index],
        weekday_counts.values
    )
    plt.title("Number of Trips by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Trips")
    plt.xticks(rotation=30)
    save_plot("03_trips_by_weekday.png")


# ================================================================
# STEP 10 - MONTHLY ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 10 - MONTHLY ANALYSIS")
print("=" * 80)

if "pickup_month" in df.columns:
    month_counts = df["pickup_month"].value_counts().sort_index()
    print(month_counts.to_string())

    plt.figure(figsize=(10, 6))
    plt.bar(month_counts.index, month_counts.values)
    plt.title("Number of Trips by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Trips")
    plt.xticks(range(1, 13))
    save_plot("04_trips_by_month.png")


# ================================================================
# STEP 11 - PASSENGER COUNT ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 11 - PASSENGER COUNT ANALYSIS")
print("=" * 80)

if "passenger_count" in df.columns:
    passenger_counts = (
        pd.to_numeric(df["passenger_count"], errors="coerce")
        .value_counts()
        .sort_index()
    )

    print(passenger_counts.to_string())

    plt.figure(figsize=(10, 6))
    plt.bar(
        passenger_counts.index.astype(str),
        passenger_counts.values
    )
    plt.title("Distribution of Passenger Count")
    plt.xlabel("Passenger Count")
    plt.ylabel("Number of Trips")
    save_plot("05_passenger_count.png")


# ================================================================
# STEP 12 - GPS LOCATION ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 12 - GPS LOCATION ANALYSIS")
print("=" * 80)

gps_columns = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude"
]

available_gps = [c for c in gps_columns if c in df.columns]

if available_gps:
    print(df[available_gps].describe().T.to_string())

if "pickup_longitude" in df.columns and "pickup_latitude" in df.columns:
    location_df = df[
        ["pickup_longitude", "pickup_latitude"]
    ].copy()

    location_df["pickup_longitude"] = pd.to_numeric(
        location_df["pickup_longitude"], errors="coerce"
    )
    location_df["pickup_latitude"] = pd.to_numeric(
        location_df["pickup_latitude"], errors="coerce"
    )

    location_df = location_df.dropna()

    if len(location_df) > 100000:
        location_df = location_df.sample(
            100000, random_state=42
        )

    plt.figure(figsize=(10, 8))
    plt.scatter(
        location_df["pickup_longitude"],
        location_df["pickup_latitude"],
        s=2,
        alpha=0.3
    )
    plt.title("Pickup Location Distribution")
    plt.xlabel("Pickup Longitude")
    plt.ylabel("Pickup Latitude")
    save_plot("06_pickup_locations.png")


# ================================================================
# STEP 13 - HAVERSINE DISTANCE ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 13 - DISTANCE ANALYSIS")
print("=" * 80)

distance_columns = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude"
]

if all(c in df.columns for c in distance_columns):

    distance_df = df[distance_columns].copy()

    for column in distance_columns:
        distance_df[column] = pd.to_numeric(
            distance_df[column], errors="coerce"
        )

    distance_df = distance_df.dropna()

    # NYC geographic sanity limits for EDA only.
    distance_df = distance_df[
        distance_df["pickup_latitude"].between(40, 42)
        & distance_df["dropoff_latitude"].between(40, 42)
        & distance_df["pickup_longitude"].between(-75, -72)
        & distance_df["dropoff_longitude"].between(-75, -72)
    ].copy()

    earth_radius_km = 6371.0

    lat1 = np.radians(distance_df["pickup_latitude"])
    lon1 = np.radians(distance_df["pickup_longitude"])
    lat2 = np.radians(distance_df["dropoff_latitude"])
    lon2 = np.radians(distance_df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    distance_df["distance_km"] = (
        2 * earth_radius_km
        * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    )

    print(distance_df["distance_km"].describe().to_string())

    upper_limit = distance_df["distance_km"].quantile(0.99)
    plot_distance = distance_df[
        distance_df["distance_km"] <= upper_limit
    ]["distance_km"]

    plt.figure(figsize=(10, 6))
    plt.hist(plot_distance, bins=50)
    plt.title("Trip Distance Distribution (up to 99th percentile)")
    plt.xlabel("Distance (km)")
    plt.ylabel("Number of Trips")
    save_plot("07_distance_distribution.png")


# ================================================================
# STEP 14 - TRIP DURATION / ETA ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 14 - TRIP DURATION / ETA ANALYSIS")
print("=" * 80)

target_column = None

for candidate in ["trip_duration", "duration", "ETA"]:
    if candidate in df.columns:
        target_column = candidate
        break

if target_column:
    print(f"Target column found: {target_column}")

    df[target_column] = pd.to_numeric(
        df[target_column], errors="coerce"
    )

    print(df[target_column].describe().to_string())

    if target_column == "trip_duration":
        duration_minutes = df[target_column].dropna() / 60
        upper_limit = duration_minutes.quantile(0.99)
        duration_plot = duration_minutes[
            duration_minutes <= upper_limit
        ]

        plt.figure(figsize=(10, 6))
        plt.hist(duration_plot, bins=50)
        plt.title(
            "Trip Duration Distribution (up to 99th percentile)"
        )
        plt.xlabel("Trip Duration (minutes)")
        plt.ylabel("Number of Trips")
        save_plot("08_trip_duration_distribution.png")


# ================================================================
# STEP 15 - DISTANCE VS TRIP DURATION
# ================================================================

print("\n" + "=" * 80)
print("STEP 15 - DISTANCE VS TRIP DURATION")
print("=" * 80)

if target_column == "trip_duration":

    relationship_df = df[
        ["pickup_latitude", "pickup_longitude",
         "dropoff_latitude", "dropoff_longitude",
         "trip_duration"]
    ].copy()

    for column in distance_columns:
        relationship_df[column] = pd.to_numeric(
            relationship_df[column], errors="coerce"
        )

    relationship_df["trip_duration"] = pd.to_numeric(
        relationship_df["trip_duration"], errors="coerce"
    )

    relationship_df = relationship_df.dropna()

    lat1 = np.radians(relationship_df["pickup_latitude"])
    lon1 = np.radians(relationship_df["pickup_longitude"])
    lat2 = np.radians(relationship_df["dropoff_latitude"])
    lon2 = np.radians(relationship_df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    relationship_df["distance_km"] = (
        2 * 6371.0
        * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    )

    relationship_df["duration_minutes"] = (
        relationship_df["trip_duration"] / 60
    )

    relationship_df = relationship_df[
        relationship_df["distance_km"] > 0
    ]

    distance_limit = relationship_df["distance_km"].quantile(0.99)
    duration_limit = relationship_df["duration_minutes"].quantile(0.99)

    relationship_df = relationship_df[
        (relationship_df["distance_km"] <= distance_limit)
        & (relationship_df["duration_minutes"] <= duration_limit)
    ]

    if len(relationship_df) > 100000:
        relationship_df = relationship_df.sample(
            100000, random_state=42
        )

    plt.figure(figsize=(10, 7))
    plt.scatter(
        relationship_df["distance_km"],
        relationship_df["duration_minutes"],
        s=3,
        alpha=0.3
    )
    plt.title("Distance vs Trip Duration")
    plt.xlabel("Distance (km)")
    plt.ylabel("Trip Duration (minutes)")
    save_plot("09_distance_vs_duration.png")

    correlation = relationship_df[
        ["distance_km", "duration_minutes"]
    ].corr().iloc[0, 1]

    print(
        f"Correlation between distance and duration: {correlation:.4f}"
    )


# ================================================================
# STEP 16 - DURATION BY PICKUP HOUR
# ================================================================

print("\n" + "=" * 80)
print("STEP 16 - TRIP DURATION BY PICKUP HOUR")
print("=" * 80)

if target_column == "trip_duration" and "pickup_hour" in df.columns:

    hour_duration = df[
        ["pickup_hour", "trip_duration"]
    ].dropna()

    hour_duration["duration_minutes"] = (
        hour_duration["trip_duration"] / 60
    )

    median_by_hour = (
        hour_duration.groupby("pickup_hour")["duration_minutes"]
        .median()
    )

    print(median_by_hour.to_string())

    plt.figure(figsize=(10, 6))
    plt.plot(
        median_by_hour.index,
        median_by_hour.values,
        marker="o"
    )
    plt.title("Median Trip Duration by Pickup Hour")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Median Trip Duration (minutes)")
    plt.xticks(range(24))
    save_plot("10_duration_by_pickup_hour.png")


# ================================================================
# STEP 17 - CORRELATION ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 17 - CORRELATION ANALYSIS")
print("=" * 80)

correlation_columns = []

for column in [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "pickup_weekday",
    "is_weekend",
    "trip_duration"
]:
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            correlation_columns.append(column)

if len(correlation_columns) >= 2:

    correlation_matrix = df[
        correlation_columns
    ].apply(pd.to_numeric, errors="coerce").corr()

    print(correlation_matrix.round(3).to_string())

    correlation_matrix.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "correlation_matrix.csv"
        )
    )

    plt.figure(figsize=(12, 10))
    plt.imshow(
        correlation_matrix,
        aspect="auto"
    )
    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90
    )
    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns
    )

    plt.title("Correlation Matrix")
    save_plot("11_correlation_matrix.png")


# ================================================================
# STEP 18 - RUSH HOUR ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("STEP 18 - RUSH HOUR ANALYSIS")
print("=" * 80)

if "pickup_hour" in df.columns:

    df["rush_hour"] = (
        df["pickup_hour"].between(7, 10)
        | df["pickup_hour"].between(16, 20)
    ).astype(int)

    if target_column == "trip_duration":

        rush_summary = (
            df.groupby("rush_hour")["trip_duration"]
            .agg(["count", "mean", "median"])
        )

        rush_summary["mean_minutes"] = (
            rush_summary["mean"] / 60
        )

        rush_summary["median_minutes"] = (
            rush_summary["median"] / 60
        )

        print(
            rush_summary[
                ["count", "mean_minutes", "median_minutes"]
            ].to_string()
        )


# ================================================================
# STEP 19 - FINAL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("STEP 19 - EDA COMPLETED")
print("=" * 80)

print(f"Dataset rows    : {len(df):,}")
print(f"Dataset columns : {df.shape[1]:,}")
print(f"Duplicate rows  : {duplicate_count:,}")
print(f"Output folder   : {OUTPUT_DIR}")

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {filename}")

print("\nEDA completed successfully.")
print("=" * 80)
