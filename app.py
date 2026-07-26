import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Algerian Real Estate Price Predictor", page_icon="🏠", layout="centered")

# ---------- Load model and metadata ----------
@st.cache_resource
def load_artifacts():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return model, feature_names, explainer

model, feature_names, explainer = load_artifacts()

# ---------- Extract available cities and property types from feature names ----------
cities = sorted([f.replace("store_adress_", "") for f in feature_names if f.startswith("store_adress_")])
property_types = sorted([f.replace("property_type_", "") for f in feature_names if f.startswith("property_type_")])

# ---------- UI ----------
st.title("🏠 Algerian Real Estate Price Predictor")
st.write("Estimate property prices in Algeria using a machine learning model trained on real Ouedkniss listings.")

col1, col2 = st.columns(2)

with col1:
    listing_type = st.selectbox("Listing type", ["vente", "location", "echange", "other"])
    property_type = st.selectbox("Property type", property_types)
    num_rooms = st.slider("Number of rooms (F-type)", 0, 10, 3)

with col2:
    city = st.selectbox("City (wilaya)", cities)
    year = st.selectbox("Listing year", [2020, 2021, 2022, 2023, 2024, 2025], index=4)
    has_rooms_info = st.checkbox("Room count specified in listing", value=True)

predict_btn = st.button("Predict price", type="primary")

if predict_btn:
    # build input row matching training feature format
    input_data = {feat: 0 for feat in feature_names}
    input_data["num_rooms"] = num_rooms
    input_data["year"] = year
    input_data["has_rooms_info"] = int(has_rooms_info)

    listing_col = f"listing_type_{listing_type}"
    if listing_col in input_data:
        input_data[listing_col] = 1

    property_col = f"property_type_{property_type}"
    if property_col in input_data:
        input_data[property_col] = 1

    city_col = f"store_adress_{city}"
    if city_col in input_data:
        input_data[city_col] = 1

    X_input = pd.DataFrame([input_data])[feature_names]

    # predict (model was trained on log price)
    log_pred = model.predict(X_input)[0]
    price_pred = np.expm1(log_pred)

    st.success(f"### Estimated price: {price_pred:,.0f} DA")

    # SHAP explanation for this specific prediction
    st.subheader("Why this price?")
    shap_values = explainer.shap_values(X_input)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[0],
        feature_names=feature_names, max_display=8, show=False
    )
    st.pyplot(fig)

    st.caption("Red bars push the price up, blue bars push it down, relative to the average prediction.")

st.divider()
st.caption("Model: XGBoost · R² = 0.624 · Trained on 15,532 cleaned listings from Ouedkniss")