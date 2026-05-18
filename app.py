import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

st.title("🛡️ Malicious Traffic Detection - IDS")


@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")

    # The scaler is the source of truth for feature count and order.
    # Fall back to the JSON only if the scaler doesn't carry names.
    if hasattr(scaler, "feature_names_in_"):
        features = list(scaler.feature_names_in_)
    else:
        with open("models/selected_features.json") as f:
            features = json.load(f)

    return model, scaler, encoder, features


model, scaler, encoder, features = load_model()

# --- Sanity check: surface mismatches instead of crashing later ---
expected = scaler.n_features_in_
if len(features) != expected:
    st.error(
        f"Feature mismatch: feature list has {len(features)} entries "
        f"but the scaler was fitted on {expected}. "
        f"Check that scaler.pkl and selected_features.json come from the same training run."
    )
    st.stop()

if hasattr(model, "n_features_in_") and model.n_features_in_ != expected:
    st.error(
        f"Model/scaler mismatch: model expects {model.n_features_in_} features, "
        f"scaler expects {expected}."
    )
    st.stop()

st.write(f"Modèle chargé avec {len(features)} features")
with st.expander("Voir la liste des features"):
    st.write(features)

st.subheader("Prédiction manuelle")

inputs = []
for feat in features:
    val = st.number_input(feat, value=0.0)
    inputs.append(val)

if st.button("Prédire"):
    # Build a DataFrame so sklearn aligns by column name, not position.
    X_df = pd.DataFrame([inputs], columns=features)
    X = scaler.transform(X_df)
    pred = model.predict(X)[0]
    label = encoder.inverse_transform([pred])[0]
    st.success(f"Prédiction : **{label}**")
