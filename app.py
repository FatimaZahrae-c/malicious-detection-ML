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

st.markdown("""
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
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("models/random_forest.pkl")
    scaler   = joblib.load("models/scaler.pkl")
    encoder  = joblib.load("models/label_encoder.pkl")
    with open("models/selected_features.json") as f:
        selected = json.load(f)
    all_feat = list(scaler.feature_names_in_)
    return model, scaler, encoder, selected, all_feat


model, scaler, encoder, selected, all_feat = load_model()


# ─────────────────────────────────────────────────────────────
# Preset attack signatures (median values from CICIDS2017)
# These let you quickly test that the model recognizes each class.
# Replace these with your own median-computed values for best results.
# ─────────────────────────────────────────────────────────────
PRESETS = {
    "— Manuel (tout à zéro) —": {f: 0.0 for f in selected},

    "PortScan": {
        'Flow Duration': 5021059.0,
        'Total Length of Fwd Packets': 703.0,
        'Bwd Packet Length Max': 1050.0,
        'Bwd Packet Length Mean': 282.80,
        'Bwd Packet Length Std': 456.92,
        'Flow IAT Std': 1568379.16,
        'Flow IAT Max': 4965658.0,
        'Fwd IAT Total': 55401.0,
        'Fwd IAT Std': 17612.67,
        'Fwd IAT Max': 41863.0,
        'Max Packet Length': 1050.0,
        'Packet Length Mean': 176.42,
        'Packet Length Std': 317.47,
        'Packet Length Variance': 100787.90,
        'FIN Flag Count': 0.0,
        'Average Packet Size': 192.45,
        'Avg Bwd Segment Size': 282.80,
        'Idle Mean': 0.0,
        'Idle Max': 0.0,
        'Idle Min': 0.0,
    },

    "FTP-Patator": {
        # Values for a representative (non-empty) FTP-Patator flow.
        # The dataset's first row is degenerate — these are typical medians.
        'Flow Duration': 112000.0,
        'Total Length of Fwd Packets': 24.0,
        'Bwd Packet Length Max': 96.0,
        'Bwd Packet Length Mean': 48.0,
        'Bwd Packet Length Std': 35.0,
        'Flow IAT Std': 28000.0,
        'Flow IAT Max': 95000.0,
        'Fwd IAT Total': 110000.0,
        'Fwd IAT Std': 35000.0,
        'Fwd IAT Max': 95000.0,
        'Max Packet Length': 96.0,
        'Packet Length Mean': 35.0,
        'Packet Length Std': 38.0,
        'Packet Length Variance': 1450.0,
        'FIN Flag Count': 0.0,
        'Average Packet Size': 42.0,
        'Avg Bwd Segment Size': 48.0,
        'Idle Mean': 0.0,
        'Idle Max': 0.0,
        'Idle Min': 0.0,
    },

    "DoS Hulk": {
        'Flow Duration': 1878.0,
        'Total Length of Fwd Packets': 382.0,
        'Bwd Packet Length Max': 4355.0,
        'Bwd Packet Length Mean': 1932.50,
        'Bwd Packet Length Std': 2182.47,
        'Flow IAT Std': 229.13,
        'Flow IAT Max': 577.0,
        'Fwd IAT Total': 975.0,
        'Fwd IAT Std': 265.17,
        'Fwd IAT Max': 675.0,
        'Max Packet Length': 4355.0,
        'Packet Length Mean': 1197.70,
        'Packet Length Std': 1886.33,
        'Packet Length Variance': 3558249.79,
        'FIN Flag Count': 0.0,
        'Average Packet Size': 1330.78,
        'Avg Bwd Segment Size': 1932.50,
        'Idle Mean': 0.0,
        'Idle Max': 0.0,
        'Idle Min': 0.0,
    },

    "Bot": {
        'Flow Duration': 60202640.0,
        'Total Length of Fwd Packets': 322.0,
        'Bwd Packet Length Max': 256.0,
        'Bwd Packet Length Mean': 28.44,
        'Bwd Packet Length Std': 85.33,
        'Flow IAT Std': 4901981.33,
        'Flow IAT Max': 10200000.0,
        'Fwd IAT Total': 51200000.0,
        'Fwd IAT Std': 5268489.91,
        'Fwd IAT Max': 10200000.0,
        'Max Packet Length': 322.0,
        'Packet Length Mean': 30.42,
        'Packet Length Std': 91.78,
        'Packet Length Variance': 8424.26,
        'FIN Flag Count': 0.0,
        'Average Packet Size': 32.11,
        'Avg Bwd Segment Size': 28.44,
        'Idle Mean': 10200000.0,
        'Idle Max': 10200000.0,
        'Idle Min': 10100000.0,
    },
}


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🛡️ Malicious Traffic Detection - IDS")
st.markdown(
    f"Modèle : **Random Forest** | "
    f"Features : **{len(selected)}** | "
    f"Dataset : **CICIDS2017** | "
    f"F1 = **0.9824**"
)

with st.expander("Voir la liste des features utilisées"):
    st.write(selected)

st.subheader("Prédiction manuelle")
st.info("Entrez les valeurs brutes (non normalisées) de la connexion réseau")

# Preset selector
preset_name = st.selectbox(
    "Charger un preset d'attaque (optionnel)",
    list(PRESETS.keys())
)
preset = PRESETS[preset_name]

# Input fields — iterate in `selected` order so display matches training order
inputs = {}
col1, col2 = st.columns(2)
for i, feat in enumerate(selected):
    col = col1 if i % 2 == 0 else col2
    with col:
        default_val = float(preset.get(feat, 0.0))
        inputs[feat] = st.number_input(
            feat,
            value=default_val,
            format="%.2f",
            key=f"input_{feat}"
        )

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
if st.button("🔍 Prédire"):
    # 1. Build full 69-feature row in the EXACT order scaler expects
    full_row = {f: 0.0 for f in all_feat}
    for f, v in inputs.items():
        if f in full_row:
            full_row[f] = v

    X_df = pd.DataFrame([full_row], columns=all_feat)  # order guaranteed

    # 2. Scale
    X_scaled = scaler.transform(X_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_feat)

    # 3. Select the 20 features IN THE ORDER `selected` defines
    #    (this is critical — model was trained on this exact ordering)
    X_sel = X_scaled_df[selected].values

    # 4. Predict
    pred  = model.predict(X_sel)[0]
    label = encoder.inverse_transform([pred])[0]
    proba = model.predict_proba(X_sel)[0]
    conf  = round(float(np.max(proba)) * 100, 1)

    # 5. Display result
    if label == "BENIGN":
        st.markdown(f"""
        <div class="normal-box">
            ✅ TRAFIC NORMAL — BENIGN<br>
            <span style="font-size:0.85rem; color:#aaffcc;">
                Confiance : {conf}%
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="attack-box">
            🔴 ATTAQUE DÉTECTÉE : {label}<br>
            <span style="font-size:0.85rem; color:#ffaaaa;">
                Confiance : {conf}%
            </span>
        </div>""", unsafe_allow_html=True)

    # 6. Top 5 probabilities
    st.markdown("---")
    st.subheader("Top 5 probabilités")
    class_labels = encoder.inverse_transform(model.classes_)
    proba_df = pd.DataFrame({
        'Classe'      : class_labels,
        'Probabilité' : np.round(proba, 4)
    }).sort_values('Probabilité', ascending=False) \
     .head(5).reset_index(drop=True)
    st.dataframe(proba_df, hide_index=True, use_container_width=True)

    # 7. Debug info (optional — collapse if you don't want it)
    with st.expander("🔧 Debug (valeurs envoyées au modèle)"):
        debug_df = pd.DataFrame({
            'Feature': selected,
            'Raw value': [inputs[f] for f in selected],
            'Scaled value': X_sel[0]
        })
        st.dataframe(debug_df, hide_index=True, use_container_width=True)
