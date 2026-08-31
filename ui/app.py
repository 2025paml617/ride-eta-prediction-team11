"""
Streamlit Web Application for NYC Taxi Trip Duration Prediction.

Run application:
    streamlit run app.py
"""

from pathlib import Path
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="NYC Taxi Duration Predictor",
    page_icon="🚕",
    layout="centered"
)

MODEL_PATH = Path("best_model.pkl")

# Exact feature order required by the model
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

@st.cache_resource
def load_model():
    """Load the trained model artifact."""
    if not MODEL_PATH.exists():
        st.error("Model file `best_model.pkl` not found. Please run the model comparison script first.")
        st.stop()
    return joblib.load(MODEL_PATH)


def main():
    st.title("🚕 NYC Taxi Trip Duration Predictor")
    st.markdown("Enter trip details below to estimate the total duration in minutes.")

    model = load_model()

    # Form UI Inputs
    with st.form("prediction_form"):
        st.subheader("Trip Configuration")

        col1, col2 = st.columns(2)

        with col1:
            passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1, step=1)
            distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=200.0, value=5.2, step=0.1)
            vendor_id = st.selectbox("Vendor", options=["Vendor 1", "Vendor 2"])

        with col2:
            pickup_date = st.date_input("Pickup Date", datetime.date(2026, 1, 15))
            pickup_time = st.time_input("Pickup Time", datetime.time(18, 30))
            store_flag = st.selectbox("Store and Forward Flag", options=["No (N)", "Yes (Y)"])

        submit_button = st.form_submit_button(label="Predict Duration")

    if submit_button:
        # Feature Extraction from Inputs
        pickup_dt = datetime.datetime.combine(pickup_date, pickup_time)
        pickup_hour = pickup_dt.hour
        pickup_day = pickup_dt.day
        pickup_month = pickup_dt.month
        pickup_weekday = pickup_dt.weekday()

        # Derived features
        is_weekend = 1 if pickup_weekday >= 5 else 0
        rush_hour = 1 if (7 <= pickup_hour <= 9 or 16 <= pickup_hour <= 19) and is_weekend == 0 else 0

        # One-Hot Encoded features
        cat_vendor_1 = 1 if vendor_id == "Vendor 1" else 0
        cat_vendor_2 = 1 if vendor_id == "Vendor 2" else 0
        cat_flag_N = 1 if "N" in store_flag else 0
        cat_flag_Y = 1 if "Y" in store_flag else 0

        # Construct DataFrame matching training schema
        input_data = pd.DataFrame([{
            "num__passenger_count": passenger_count,
            "num__distance_km": distance_km,
            "num__pickup_hour": pickup_hour,
            "num__pickup_day": pickup_day,
            "num__pickup_month": pickup_month,
            "num__pickup_weekday": pickup_weekday,
            "num__rush_hour": rush_hour,
            "num__is_weekend": is_weekend,
            "cat__vendor_id_1": cat_vendor_1,
            "cat__vendor_id_2": cat_vendor_2,
            "cat__store_and_fwd_flag_N": cat_flag_N,
            "cat__store_and_fwd_flag_Y": cat_flag_Y,
        }])[FEATURE_COLUMNS]

        # Model Inference
        predicted_seconds = model.predict(input_data)[0]
        predicted_minutes = max(0, predicted_seconds / 60.0)

        # Output Results
        st.success("### Prediction Result")
        st.metric(
            label="Estimated Trip Duration",
            value=f"{predicted_minutes:.1f} mins",
            delta=f"{predicted_seconds:.0f} seconds total"
        )


if __name__ == "__main__":
    main()