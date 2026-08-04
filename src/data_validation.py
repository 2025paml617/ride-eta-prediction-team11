import pandas as pd
import pandera as pa
from pandera import Column, Check

# Define the schema contract for the NYC Taxi dataset
taxi_schema = pa.DataFrameSchema({
    # Ensure ID is a string and starts with 'id'
    "id": Column(str, Check.str_startswith("id")),

    # Vendor ID is typically 1 or 2
    "vendor_id": Column(int, Check.isin([1, 2])),

    # Enforce strict datetime formats
    "pickup_datetime": Column(pa.DateTime),
    "dropoff_datetime": Column(pa.DateTime),

    # Passenger count must be 0 or greater
    "passenger_count": Column(int, Check.ge(0)),

    # GPS coordinates must fall within standard global boundaries
    "pickup_longitude": Column(float, Check.in_range(-180.0, 180.0)),
    "pickup_latitude": Column(float, Check.in_range(-90.0, 90.0)),
    "dropoff_longitude": Column(float, Check.in_range(-180.0, 180.0)),
    "dropoff_latitude": Column(float, Check.in_range(-90.0, 90.0)),

    # Flag must be exactly 'Y' or 'N'
    "store_and_fwd_flag": Column(str, Check.isin(["Y", "N"])),

    # Trip duration (in seconds) cannot be negative
    "trip_duration": Column(int, Check.ge(0))
})


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Loads the taxi dataset and validates it against the predefined schema.
    """
    print(f"Loading data from {filepath}...")

    # Read the data (using read_csv for actual large files, or read_excel for your sample)
    df = pd.read_excel(filepath)

    print("Validating schema...")
    # This will raise a pandera.errors.SchemaError if the data fails validation
    validated_df = taxi_schema.validate(df)

    print("Schema validation passed successfully!")
    return validated_df


# Execute the validation
if __name__ == "__main__":
    try:
        # Pass the exact name of your sample file
        clean_data = load_and_validate_data('Sample data.xlsx')
        print(f"Validated dataset shape: {clean_data.shape}")
    except pa.errors.SchemaError as e:
        print(f"Data validation failed: {e}")