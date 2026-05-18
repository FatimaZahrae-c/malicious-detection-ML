import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="IDS - Detection Malicious Traffic",
    page_icon="🛡️",
    layout="wide"
)

st.markdown('''
<style>
    .stApp { background-color: #0a1628; }
    h1 { color: #00d4ff !important; }
    h2, h3 { color: #7fb3d3 !important; }
    .attack-box {
        background: #2d0a0a;
        border-left: 5px solid #ff3333;
        border-radius: 6px;
        padding: 16px;
        color: #ff4444;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .normal-box {
        background: #002d0a;
        border-left: 5px solid #00cc44;
        border-radius: 6px;
        padding: 16px;
        color: #44ff88;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 10px 0;
    }
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model   = joblib.load("models/random_forest.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    with open("models/selected_features.json") as f:
        selected_features = json.load(f)
    if hasattr(scaler, "feature_names_in_"):
        all_features = list(scaler.feature_names_in_)
    else:
        with open("models/all_features.json") as f:
            all_features = json.load(f)
    return model, scaler, encoder, selected_features, all_features

model, scaler, encoder, selected_features, all_features = load_model()

st.title("🛡️ Malicious Traffic Detection - IDS")
st.markdown(f"Modèle entraîné sur **{len(selected_features)}** features "
            f"(sur {len(all_features)} au total) | "
            f"Dataset : CICIDS2017 | F1 = 0.9824")

with st.expander("Voir la liste des features utilisées"):
    st.write(selected_features)

st.subheader("Prédiction manuelle")
inputs = {}
col1, col2 = st.columns(2)
for i, feat in enumerate(selected_features):
    col = col1 if i % 2 == 0 else col2
    with col:
        inputs[feat] = st.number_input(
            feat, value=0.0, format="%.2f"
        )

if st.button("🔍 Prédire", use_container_width=True):
    full_row = {f: 0.0 for f in all_features}
    for f, v in inputs.items():
        full_row[f] = v
    X_full       = pd.DataFrame([full_row], columns=all_features)
    X_scaled     = scaler.transform(X_full)
    X_scaled_df  = pd.DataFrame(X_scaled, columns=all_features)
    X_selected   = X_scaled_df[selected_features].values
    pred         = model.predict(X_selected)[0]
    label        = encoder.inverse_transform([pred])[0]
    proba        = model.predict_proba(X_selected)[0]
    conf         = round(float(np.max(proba)) * 100, 1)

    if label == "BENIGN":
        st.markdown(f'''
        <div class="normal-box">
            ✅ TRAFIC NORMAL — BENIGN<br>
            <span style="font-size:0.85rem; color:#aaffcc;">
                Confiance : {conf}%
            </span>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="attack-box">
            🔴 ATTAQUE DÉTECTÉE : {label}<br>
            <span style="font-size:0.85rem; color:#ffaaaa;">
                Confiance : {conf}%
            </span>
        </div>''', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Top 5 probabilités")
    proba_df = pd.DataFrame({
        'Classe'      : encoder.classes_,
        'Probabilité' : proba
    }).sort_values('Probabilité', ascending=False).head(5)
    st.dataframe(proba_df, use_container_width=True,
                 hide_index=True)
