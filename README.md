# SmartPremium: Insurance Premium Prediction using Machine Learning

## Overview
SmartPremium is a machine learning project that predicts insurance premiums based on customer demographic, financial, and health-related information.

## Technologies Used
Python
Pandas
NumPy
Scikit-Learn
XGBoost
MLflow
Streamlit

## Dataset
The dataset contains 200k+ records with 20 features including:

Age
Gender
Annual Income
Health Score
Previous Claims
Policy Type
Location
Credit Score
Insurance Duration

Target Variable:
Premium Amount

## Project Workflow

1. Data Preprocessing
Handling missing values
Encoding categorical variables
Feature scaling

2. Exploratory Data Analysis
Distribution plots
Correlation analysis
Feature relationships

3. Model Training
Linear Regression
Random Forest
XGBoost

4. Model Evaluation
RMSE
MAE
R² Score
RMSLE

5. MLflow Experiment Tracking

6. Streamlit Deployment

## Run the Project

Install dependencies

pip install -r requirements.txt

Train the model

python src/train_model.py

Run Streamlit app

streamlit run streamlit_app/app.py

## Output
The application predicts insurance premium based on user inputs.

## Author
Gayathri M
