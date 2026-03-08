import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import preprocess_data

df = pd.read_csv("../data/insurance_data.csv")

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()

mlflow.start_run()

model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

mlflow.log_param("model", "RandomForest")

mlflow.log_metric("RMSE", rmse)
mlflow.log_metric("MAE", mae)
mlflow.log_metric("R2", r2)

mlflow.sklearn.log_model(model, "model")

mlflow.end_run()

joblib.dump(model, "../models/premium_model.pkl")
