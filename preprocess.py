import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    target = "Premium Amount"

    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(target, axis=1)
    y = df[target]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
