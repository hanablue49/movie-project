import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# LOAD DATA
def load_data():
    df = pd.read_csv("data/clean_movies.csv")
    return df

# TRAIN MODEL
def train_model():
    df = load_data()

    # Load feature list
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    df = df.dropna(subset=["gross_log"])
    X = df[feature_cols]
    y = df["gross_log"]

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svr = SVR(kernel="rbf")
    param_grid = {
        "C": [10, 100],
        "gamma": ["scale", 0.01],
        "epsilon": [0.1, 0.2]
    }

    grid = GridSearchCV(svr, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_scaled, y)

    best_model = grid.best_estimator_

    with open("svr_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return best_model, scaler

if __name__ == "__main__":
    print("Training model...")
    train_model()
    print("Model saved!")