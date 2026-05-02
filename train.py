import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def load_data(path):
    df = pd.read_csv(path)

    # Fix date
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
    df['policy_year'] = df['Policy Start Date'].dt.year

    # Drop unused columns
    df.drop(['Policy Start Date', 'Customer Feedback'], axis=1, inplace=True)

    return df


def build_pipeline(model, num_cols, cat_cols):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2


def train_model(data_path):
    df = load_data(data_path)

    X = df.drop("Premium Amount", axis=1)
    y = df["Premium Amount"]

    # Column separation
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    best_model = None
    best_rmse = float("inf")

    mlflow.set_experiment("Insurance Premium Prediction")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            pipeline = build_pipeline(model, num_cols, cat_cols)

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            rmse, mae, r2 = evaluate(y_test, preds)

            # MLflow logging
            mlflow.log_param("model", name)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            # FIXED (no deprecation warning)
            mlflow.sklearn.log_model(pipeline, name=name)

            print(f"{name} -> RMSE: {rmse:.2f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = pipeline

    # Save final pipeline
    joblib.dump(best_model, "C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/best_model.pkl")

    print("/n✅ Best model saved as models/best_model.pkl")


if __name__ == "__main__":
    train_model("C:/Users/gayat/AppData/Local/Programs/Python/Python313/insurance_data.csv")
