import streamlit as st
import joblib
import json
import numpy as np

st.title("🛡️ Malicious Traffic Detection - IDS")

@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    with open("models/selected_features.json") as f:
        features = json.load(f)
    return model, scaler, encoder, features

model, scaler, encoder, features = load_model()

st.write(f"Modèle chargé avec {len(features)} features")
st.write("Features:", features)

st.subheader("Prédiction manuelle")
inputs = []
for feat in features:
    val = st.number_input(feat, value=0.0)
    inputs.append(val)

if st.button("Prédire"):
    X = scaler.transform([inputs])
    pred = model.predict(X)[0]
    label = encoder.inverse_transform([pred])[0]
    st.success(f"Prédiction : **{label}**")
