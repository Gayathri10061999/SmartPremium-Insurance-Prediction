import pandas as pd

def encode_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df
