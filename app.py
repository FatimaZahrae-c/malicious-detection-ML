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
    with open("models/selected_features.json") as f:
        selected_features = json.load(f)

    # Full feature list the scaler was fitted on (69 cols).
    if hasattr(scaler, "feature_names_in_"):
        all_features = list(scaler.feature_names_in_)
    else:
        # Fallback: needs a JSON with all 69 names in the original order
        with open("models/all_features.json") as f:
            all_features = json.load(f)

    return model, scaler, encoder, selected_features, all_features


model, scaler, encoder, selected_features, all_features = load_model()

# Sanity checks
if scaler.n_features_in_ != len(all_features):
    st.error(
        f"Scaler expects {scaler.n_features_in_} features but the full feature list has {len(all_features)}. "
        f"Provide the correct full-feature list (or ensure scaler.feature_names_in_ is set)."
    )
    st.stop()

missing = [f for f in selected_features if f not in all_features]
if missing:
    st.error(f"Selected features missing from scaler's feature list: {missing}")
    st.stop()

if hasattr(model, "n_features_in_") and model.n_features_in_ != len(selected_features):
    st.error(
        f"Model expects {model.n_features_in_} features but selected_features.json has {len(selected_features)}."
    )
    st.stop()

st.write(f"Modèle entraîné sur {len(selected_features)} features (sur {len(all_features)} au total)")
with st.expander("Voir la liste des features utilisées"):
    st.write(selected_features)

st.subheader("Prédiction manuelle")

inputs = {}
for feat in selected_features:
    inputs[feat] = st.number_input(feat, value=0.0)

if st.button("Prédire"):
    # Build a full 69-column row, filling unused features with 0.0
    full_row = {f: 0.0 for f in all_features}
    for f, v in inputs.items():
        full_row[f] = v

    X_full = pd.DataFrame([full_row], columns=all_features)

    # Scale all 69, then select the 20 the model expects
    X_scaled_full = scaler.transform(X_full)
    X_scaled_df = pd.DataFrame(X_scaled_full, columns=all_features)
    X_selected = X_scaled_df[selected_features].values

    pred = model.predict(X_selected)[0]
    label = encoder.inverse_transform([pred])[0]
    st.success(f"Prédiction : **{label}**")
