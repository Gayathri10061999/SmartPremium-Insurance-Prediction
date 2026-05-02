import streamlit as st
import pandas as pd
import joblib

model = joblib.load("C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/best_model.pkl")

st.title("Insurance Premium Prediction")

def user_input():
    id=st.number_input("id",1,100000)
    age = st.number_input("Age", 18, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Annual Income")
    maritalstatus=st.selectbox("Marital Status",["Married","Single","Divorced"])
    numberofdependents=st.number_input("Number of Dependents")
    educationlevel=st.selectbox("Education Level",["High School","Bachelor's","Master's","PhD"])
    occupation=st.selectbox("Occupation",["Employed","Self-Employed","Unemployed","Blank"])
    health = st.number_input("Health Score")
    location=st.text_input("Location")
    policy = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    claims = st.number_input("Previous Claims")
    vehicleage=st.number_input("Vehicle Age")
    creditscore=st.number_input("Credit Score", 100, 1000)
    insuranceduration=st.number_input("Insurance Duration")
    policystartdate=st.date_input("Policy Start Date")
    customerfeedback=st.selectbox("Customer Feedback",["Good","Average","Poor"])
    smokingstatus=st.selectbox("Smoking Status",["Yes","No"])
    exercisefrequency=st.selectbox("Exercise Frequency",["Daily","Weekly","Monthly","Rarely"])
    propertytype=st.selectbox("Property Type",["House","Condo","Apartment"])

    
    data = {
        "id":id,
        "Age": age,
        "Gender": gender,
        "Annual Income": income,
        "Marital Status":maritalstatus,
        "Number of Dependents":numberofdependents,
        "Education Level":educationlevel,
        "Occupation":occupation,
        "Health Score": health,
        "Location":location,
        "Policy Type": policy,
        "Previous Claims": claims,
        "Vehicle Age": vehicleage,
        "Credit Score": creditscore,
        "Insurance Duration": insuranceduration,
        "Policy Start Date":policystartdate,
        "Customer Feedback":customerfeedback,
        "Smoking Status": smokingstatus,
        "Exercise Frequency": exercisefrequency,
        "Property Type": propertytype
        
    }

    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict Premium"):
    prediction = model.predict(input_df)
    st.success(f"Estimated Premium: ₹{prediction[0]:.2f}")
    
