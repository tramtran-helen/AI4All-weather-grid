import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import requests
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import date

# ------------------ Weather + Location ------------------

def get_coordinates(location: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": "EnergyPredictionApp/1.0"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f"Location '{location}' not found.")
    return float(data[0]["lat"]), float(data[0]["lon"])

def get_weather_forecast(lat, lon):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["relative_humidity_2m"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max"],
        "forecast_days": 16,
        "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    current = response.Current()
    humidity = current.Variables(0).Value()

    daily = response.Daily()
    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
    temp_max = daily.Variables(0).ValuesAsNumpy()
    temp_min = daily.Variables(1).ValuesAsNumpy()
    wind_max = daily.Variables(2).ValuesAsNumpy()

    df = pd.DataFrame({
        "date": dates,
        "temperature_avg_C": (temp_max + temp_min) / 2,
        "humidity_percent": humidity,
        "wind_speed_kmh": wind_max
    })

    return df

# ------------------ Model Utilities ------------------

@st.cache_resource
def load_model_and_scaler():
    model = pickle.load(open('linear_regression_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

@st.cache_resource
def load_fault_model():
    model = joblib.load('rf_fault_model.pkl')
    preprocessor = joblib.load('rf_preprocessor.pkl')
    label_encoder = joblib.load('rf_label_encoder.pkl')
    return model, preprocessor, label_encoder

def predict_fault(voltage, current, power, frequency, energy, load_type):
    model, preprocessor, label_encoder = load_fault_model()
    features = pd.DataFrame([{
        "Voltage_V": voltage,
        "Current_A": current,
        "Power_kW": power,
        "Frequency_Hz": frequency,
        "Energy_Consumed_kWh": energy,
        "Load_Type": load_type
    }])
    X_processed = preprocessor.transform(features)
    pred_encoded = model.predict(X_processed)[0]
    return label_encoder.inverse_transform([pred_encoded])[0]

# ------------------ Streamlit UI ------------------

st.title("Grid Fault and Energy Predictor Among U.S. Health Facilities")

location = st.text_input("Enter U.S. health facility location (e.g., Seattle)", "Seattle")
avg_past_consumption = st.number_input("Average past electricity consumption (kWh)", min_value=0.0, value=500.0)

if st.button("Run Prediction"):
    try:
        lat, lon = get_coordinates(location)
        forecast_df = get_weather_forecast(lat, lon)
        model, scaler = load_model_and_scaler()

        results = []
        st.success(f"Showing forecast for **{location.title()}** ({lat:.2f}, {lon:.2f})")

        for idx, row in forecast_df.iterrows():
            # Linear regression features: weather + past consumption
            features = np.array([[row['temperature_avg_C'], row['humidity_percent'], row['wind_speed_kmh'], avg_past_consumption]])
            # Standardize using scaler.pkl
            features_scaled = scaler.transform(features)
            pred_consumption = model.predict(features_scaled)[0]

            # Simulate grid sensor data (these will be standardized inside the preprocessor)
            voltage = np.random.normal(235, 2)
            current = np.random.normal(10, 2)
            power = voltage * current / 1000
            frequency = np.random.normal(50, 0.2)
            load_type = np.random.choice(['Residential', 'Commercial', 'Industrial'])

            fault = predict_fault(voltage, current, power, frequency, pred_consumption, load_type)

            results.append({
                "Date": row["date"].date(),
                "Temperature (°C)": round(row['temperature_avg_C'], 1),
                "Humidity (%)": round(row['humidity_percent'], 1),
                "Wind Speed (km/h)": round(row['wind_speed_kmh'], 1),
                "Predicted Consumption (kWh)": round(pred_consumption, 2),
                "Fault Prediction": fault
            })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")