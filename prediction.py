import pickle
import numpy as np
import pandas as pd

def load_artifacts():
    with open("svr_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    return model, scaler, feature_cols

def predict_movie(features_dict):
    model, scaler, feature_cols = load_artifacts()

    row = pd.DataFrame([features_dict], columns=feature_cols)

    row_scaled = scaler.transform(row)
    y_log = model.predict(row_scaled)
    y_pred = np.expm1(y_log)

    return y_pred[0]