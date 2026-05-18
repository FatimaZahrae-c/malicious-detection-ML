import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

st.title("Malicious Traffic Detection - IDS")


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
    full_row = {f: 0.0 for f in all_features}
    for f, v in inputs.items():
        full_row[f] = v
    X_full = pd.DataFrame([full_row], columns=all_features)
    X_scaled_full = scaler.transform(X_full)
    X_scaled_df   = pd.DataFrame(
        X_scaled_full, columns=all_features
    )
    X_selected = X_scaled_df[selected_features].values
    pred  = model.predict(X_selected)[0]
    label = encoder.inverse_transform([pred])[0]
    proba = model.predict_proba(X_selected)[0]
    conf  = round(float(np.max(proba)) * 100, 1)

    if label == "BENIGN":
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#002d0a,#001a05);
                    border-left:4px solid #00cc44;
                    border-radius:6px; padding:16px;
                    color:#44ff88; font-size:1.1rem;
                    font-weight:600;">
            ✅ TRAFIC NORMAL — BENIGN<br>
            <span style="font-size:0.85rem;
                         color:#7fb3d3;">
                Confiance : {conf}%
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#2d0a0a,#1a0505);
                    border-left:4px solid #ff3333;
                    border-radius:6px; padding:16px;
                    color:#ff4444; font-size:1.1rem;
                    font-weight:600;">
            🔴 ATTAQUE DÉTECTÉE : {label}<br>
            <span style="font-size:0.85rem;
                         color:#ffaaaa;">
                Confiance : {conf}%
            </span>
        </div>
        """, unsafe_allow_html=True)
