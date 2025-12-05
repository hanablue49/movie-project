import streamlit as st
import numpy as np
import pandas as pd
import pickle

from prediction import predict_movie

# Load list of features
with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

st.set_page_config(page_title="Movie Gross Prediction – SVR", layout="centered")

# TITLE & DESCRIPTION
st.title("🎬 Movie Gross Prediction — SVR Model")
st.write(
    "Prediksi pendapatan worldwide berdasarkan informasi film. "
    "Model menggunakan Support Vector Regression (Final Project)."
)

st.markdown("---")

# Numeric Inputs
st.subheader("🎛 Film Basic Information")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Duration (minutes)", min_value=0, value=110)

with col2:
    rating = st.number_input("Rating (0–10)", min_value=0.0, max_value=10.0, value=7.5)

col3, col4 = st.columns(2)

with col3:
    budget = st.number_input("Budget (USD)", min_value=0.0, value=50000000.0)

with col4:
    votes = st.number_input("Votes Count", min_value=0, value=200000)

year = st.number_input("Release Year", min_value=1900, max_value=2100, value=2020)

st.markdown("---")

# Dropdown Inputs
st.subheader("🎭 Movie Metadata")

# Auto-extract encoded metadata options
genre_list = sorted([c.replace("genre_", "") for c in feature_cols if c.startswith("genre_")])
writers_list = sorted([c.replace("writers_", "") for c in feature_cols if c.startswith("writers_")])
directors_list = sorted([c.replace("directors_", "") for c in feature_cols if c.startswith("directors_")])
stars_list = sorted([c.replace("stars_", "") for c in feature_cols if c.startswith("stars_")])
production_list = sorted([c.replace("production_companies_", "") for c in feature_cols if c.startswith("production_companies_")])

main_genre = st.selectbox("Main Genre", ["None"] + genre_list)
writer = st.selectbox("Writer", ["None"] + writers_list)
director = st.selectbox("Director", ["None"] + directors_list)
star = st.selectbox("Main Actor", ["None"] + stars_list)
production = st.selectbox("Production Company", ["None"] + production_list)

st.markdown("---")

# BUILD FEATURE INPUT

# Start with all zero
feature_input = {col: 0 for col in feature_cols}

# Inject numeric fields
feature_input["duration"] = duration
feature_input["rating"] = rating
feature_input["budget_log"] = np.log1p(budget)
feature_input["votes_log"] = np.log1p(votes)
feature_input["year"] = int(year)
feature_input["profit_log"] = 0.0

# Inject categorical choices
if main_genre != "None":
    colname = f"genre_{main_genre}"
    if colname in feature_input:
        feature_input[colname] = 1

if writer != "None":
    colname = f"writers_{writer}"
    if colname in feature_input:
        feature_input[colname] = 1

if director != "None":
    colname = f"directors_{director}"
    if colname in feature_input:
        feature_input[colname] = 1

if star != "None":
    colname = f"stars_{star}"
    if colname in feature_input:
        feature_input[colname] = 1

if production != "None":
    colname = f"production_companies_{production}"
    if colname in feature_input:
        feature_input[colname] = 1

# PREDICTION BUTTON
st.markdown("### 🚀 Predict Movie Gross")

if st.button("Predict"):
    prediction = predict_movie(feature_input)
    st.success(f"💰 Predicted Worldwide Gross: **${prediction:,.2f}**")
