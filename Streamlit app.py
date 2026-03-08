import streamlit as st
import joblib
import numpy as np

model = joblib.load("../models/premium_model.pkl")

st.title("SmartPremium Insurance Predictor")

age = st.number_input("Age")
income = st.number_input("Annual Income")
health = st.number_input("Health Score")
claims = st.number_input("Previous Claims")

if st.button("Predict Premium"):

    data = np.array([[age, income, health, claims]])

    prediction = model.predict(data)

    st.success(f"Estimated Premium: ₹{prediction[0]:,.2f}")
