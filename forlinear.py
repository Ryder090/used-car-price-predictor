import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os


model = joblib.load("linear_ford_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
columns = [col.strip() for col in columns]  

scaled_columns = ['engineSize', 'car_age', 'total_driven_km', 'kilometers_per_liter']

st.set_page_config(page_title="Used Car Price Predictor 🚗", page_icon="🚗", layout="centered")

image_path = "Ford-.png"

if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning("⚠️ Banner image not found. Please check the path or file name.")


    
st.markdown("<h1 style='text-align: center; color:#2c3e50;'>Used Car Price Predictor 🚘</h1>", unsafe_allow_html=True)
st.markdown("---")


car_age = st.slider("📅 How old is the car?", 6, 15, 10)
engine_size = st.selectbox("🔧 Engine Size (liters)", [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0])
total_driven = st.number_input("🛣️ Total Kilometers Driven", min_value=5000, max_value=120000, value=50000)
average = st.number_input("⛽ Fuel Average (km/l)", min_value=5.0, max_value=30.0, value=15.0)

fuel_type = st.selectbox("🛢️ Fuel Type", ['Petrol', 'Diesel'])
transmission = st.selectbox("⚙️ Transmission Type", ['Automatic', 'Manual', 'Semi-Auto'])
model_car = st.selectbox("Car Model", ['Edge', 'Fiesta', 'Focus', 'Galaxy', 'KA', 'Ka+',
                                       'Kuga', 'S-MAX'])  


if st.button("💸 Predict Price"):

    
    raw_input = {
        'engineSize': engine_size,
        'car_age': car_age,
        'total_driven_km': total_driven,
        'kilometers_per_liter': average,
        'fuelType_binary': 1 if fuel_type == 'Petrol' else 0,
        'transmission_Manual': 1 if transmission == 'Manual' else 0,
        'transmission_Semi-Auto': 1 if transmission == 'Semi-Auto' else 0,
    }


    for model_col in ['model_ Edge', 'model_ Fiesta', 'model_ Focus', 'model_ Galaxy',
                      'model_ KA', 'model_ Ka+', 'model_ Kuga', 'model_ S-MAX']:
        raw_input[model_col] = 1 if model_col.endswith(model_car) else 0
   
    input_df = pd.DataFrame([raw_input])

   
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    
    input_df = input_df[columns]

    
    input_df[scaled_columns] = scaler.transform(input_df[scaled_columns])

    prediction = model.predict(input_df)[0]


    if prediction < 0:
        st.warning("🚗 The car seems very old or uncommon. Unable to estimate an accurate price.")
    else:
        st.markdown(
            f"<div style='background-color:#e3f2fd; padding: 20px; border-radius: 10px; text-align:center;'>"
            f"<h2 style='color: #1e88e5;'>💰 Estimated Car Price: ₹{int(prediction):,}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Made by Shikhar ",
    unsafe_allow_html=True
)
