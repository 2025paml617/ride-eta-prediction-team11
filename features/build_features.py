"""
Build the NYC taxi feature store.

Input:
    NYC_Preprocessed.csv

Output:
    feature_store.db

The CSV is already preprocessed. This script only:
1. Selects the model features and target.
2. Stores them in SQLite.
3. Stores feature metadata.
4. Loads data in chunks to control memory usage.

Run:
    python build_features.py
"""

from pathlib import Path
import sqlite3
import logging
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "NYC_Preprocessed.csv"
DB_PATH = PROJECT_ROOT / "feature_store" / "feature_store.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

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
CHUNK_SIZE = 50_000


def create_database():
    """Create the feature-store tables."""
    if DB_PATH.exists():
        logger.info("Replacing existing feature store: %s", DB_PATH)
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    conn.executescript("""
        CREATE TABLE feature_store (
            feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
            num__passenger_count REAL,
            num__distance_km REAL,
            num__pickup_hour REAL,
            num__pickup_day REAL,
            num__pickup_month REAL,
            num__pickup_weekday REAL,
            num__rush_hour REAL,
            num__is_weekend REAL,
            cat__vendor_id_1 REAL,
            cat__vendor_id_2 REAL,
            cat__store_and_fwd_flag_N REAL,
            cat__store_and_fwd_flag_Y REAL,
            trip_duration REAL
        );

        CREATE TABLE feature_store_metadata (
            feature_name TEXT PRIMARY KEY,
            feature_type TEXT NOT NULL,
            description TEXT
        );
    """)

    metadata = [
        ("num__passenger_count", "numeric", "Passenger count"),
        ("num__distance_km", "numeric", "Trip distance in kilometres"),
        ("num__pickup_hour", "numeric", "Pickup hour"),
        ("num__pickup_day", "numeric", "Pickup day of month"),
        ("num__pickup_month", "numeric", "Pickup month"),
        ("num__pickup_weekday", "numeric", "Pickup weekday"),
        ("num__rush_hour", "numeric", "Rush-hour indicator"),
        ("num__is_weekend", "numeric", "Weekend indicator"),
        ("cat__vendor_id_1", "binary", "Vendor 1 indicator"),
        ("cat__vendor_id_2", "binary", "Vendor 2 indicator"),
        ("cat__store_and_fwd_flag_N", "binary", "Store-and-forward N indicator"),
        ("cat__store_and_fwd_flag_Y", "binary", "Store-and-forward Y indicator"),
        ("trip_duration", "target", "Trip duration in seconds"),
    ]

    conn.executemany(
        "INSERT INTO feature_store_metadata VALUES (?, ?, ?)",
        metadata
    )

    return conn


def build_feature_store():
    """Read the preprocessed data and populate SQLite."""
    if not DATA_PATH.exists():
        logger.error("Input data file not found: %s", DATA_PATH)
        raise FileNotFoundError(
            f"{DATA_PATH} was not found. "
            "Place NYC_Preprocessed.csv beside this script."
        )

    conn = create_database()
    required_columns = FEATURE_COLUMNS + [TARGET]
    total_rows = 0
    logger.info("Building feature store from %s", DATA_PATH)

    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
        missing = [
            column for column in required_columns
            if column not in chunk.columns
        ]

        if missing:
            conn.close()
            logger.error("Missing required columns: %s", missing)
            raise ValueError(f"Missing columns: {missing}")

        # Keep only features needed by the model.
        chunk = chunk[required_columns]

        # Remove invalid numeric values before storing features.
        chunk = chunk.replace([float("inf"), float("-inf")], pd.NA)
        chunk = chunk.dropna()

        chunk.to_sql(
            "feature_store",
            conn,
            if_exists="append",
            index=False
        )

        total_rows += len(chunk)

    # These indexes make future SQL reads more efficient.
    conn.execute(
        "CREATE INDEX idx_feature_store_id "
        "ON feature_store(feature_id)"
    )

    conn.execute(
        "CREATE INDEX idx_feature_store_target "
        "ON feature_store(trip_duration)"
    )

    conn.commit()
    conn.close()

    logger.info("Feature store created: %s", DB_PATH)
    logger.info("Rows stored: %d; features stored: %d; target: %s", total_rows, len(FEATURE_COLUMNS), TARGET)
    print(f"Feature store created: {DB_PATH}")
    print(f"Rows stored: {total_rows}")
    print(f"Features stored: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET}")


if __name__ == "__main__":
    try:
        build_feature_store()
    except Exception:
        logger.exception("Feature-store build failed")
        raise
