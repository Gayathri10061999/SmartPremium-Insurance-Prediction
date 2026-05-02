from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def build_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor())
    ])
    return pipeline
