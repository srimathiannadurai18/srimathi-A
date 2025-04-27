import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load model
model = pickle.load(open('shelf_life_mlp.pkl', 'rb'))

# Input functions
def get_inputs():
    st.title("Shelf Life Prediction App (Smart Packaging)")
    storage_temp = st.number_input("Storage Temperature (°C)")
    storage_humid = st.number_input("Storage Humidity (%)")
    material_type = st.selectbox("Material Type", ["Glass", "HDPE", "PET", "PLA", "Paperboard", "Aluminum"])
    product_state = st.selectbox("Product State", ["Semisolid", "Solid", "Liquid"])
    storage_cond = st.selectbox("Storage Condition", ["Chilled", "Refrigerated", "Ambient"])
    return {
        'storage_temp': storage_temp,
        'storage_humid': storage_humid,
        'material_type': material_type,
        'product_state': product_state,
        'storage_cond': storage_cond
    }

# Prediction function
def predict(X_input):
    X_df = pd.DataFrame([X_input])
    prediction = model.predict(X_df)
    return prediction[0]

if __name__ == "__main__":
    X_input = get_inputs()
    if st.button("Predict Shelf Life"):
        result = predict(X_input)
        st.success(f"Predicted Shelf Life: {result:.0f} days")
