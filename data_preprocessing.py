import pandas as pd
from sklearn.impute import SimpleImputer

path='C:/Users/gayat/AppData/Local/Programs/Python/Python313/insurance_data.csv'

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Fix date
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
    df['policy_year'] = df['Policy Start Date'].dt.year

    # Drop raw date & text
    df.drop(['Policy Start Date', 'Customer Feedback'], axis=1, inplace=True)

    # Separate features
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Imputation
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df
