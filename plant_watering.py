import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

model = tf.keras.models.load_model("plant_watering_ann.keras",compile=False)

scaler = joblib.load("plant_scaler.pkl")


st.set_page_config(
    page_title="Smart Plant Watering Predictor",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Smart Plant Watering Predictor")
st.write("Predict whether your plant needs watering using an Artificial Neural Network.")

st.subheader("Enter Plant Conditions")

soil_moisture = st.slider(
    "Soil Moisture (%)",
    min_value=0,
    max_value=100,
    value=30
)

temperature = st.slider(
    "Temperature (°C)",
    min_value=0,
    max_value=50,
    value=30
)

humidity = st.slider(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    value=50
)



light_intensity = st.slider(
    "Light Intensity (Lux)",
    min_value=0,
    max_value=1000,
    value=600
)

days_since_watering = st.slider(
    "Days Since Last Watering",
    min_value=0,
    max_value=10,
    value=2
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Soil_Moisture": [soil_moisture],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Light_Intensity": [light_intensity],
        "Days_Since_Watering": [days_since_watering]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled, verbose=0)[0][0]

    water_prob = prediction * 100
    no_water_prob = (1 - prediction) * 100

    st.subheader("Prediction Result")
    if prediction >= 0.5:
        st.error("💧 Water Needed")
        confidence = water_prob
    else:
        st.success("🌿 No Water Needed")
        confidence = no_water_prob

    # Confidence Level
    st.subheader("Confidence Level")

    st.progress(int(confidence))

    st.write(f"**Model Confidence:** {confidence:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💧 Water Needed", f"{water_prob:.2f}%")

    with col2:
        st.metric("🌿 No Water Needed", f"{no_water_prob:.2f}%")
