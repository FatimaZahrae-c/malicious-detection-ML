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
# Preset attack signatures (from CICIDS2017 dataset)
PRESETS = {
    "— Manuel (tout à zéro) —": {f: 0.0 for f in selected},

    "PortScan": {
        'Flow Duration': 48.00,
        'Total Length of Fwd Packets': 2.00,
        'Bwd Packet Length Max': 6.00,
        'Bwd Packet Length Mean': 6.00,
        'Bwd Packet Length Std': 0.00,
        'Flow IAT Std': 0.00,
        'Flow IAT Max': 48.00,
        'Fwd IAT Total': 0.00,
        'Fwd IAT Std': 0.00,
        'Fwd IAT Max': 0.00,
        'Max Packet Length': 6.00,
        'Packet Length Mean': 3.33,
        'Packet Length Std': 2.31,
        'Packet Length Variance': 5.33,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 5.00,
        'Avg Bwd Segment Size': 6.00,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },  # 78,614 rows used

    "FTP-Patator": {
        'Flow Duration': 6527390.50,
        'Total Length of Fwd Packets': 69.00,
        'Bwd Packet Length Max': 34.00,
        'Bwd Packet Length Mean': 12.53,
        'Bwd Packet Length Std': 14.55,
        'Flow IAT Std': 869828.86,
        'Flow IAT Max': 2661635.50,
        'Fwd IAT Total': 4661544.00,
        'Fwd IAT Std': 1119557.11,
        'Fwd IAT Max': 2502473.50,
        'Max Packet Length': 34.00,
        'Packet Length Mean': 10.58,
        'Packet Length Std': 12.23,
        'Packet Length Variance': 149.61,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 12.17,
        'Avg Bwd Segment Size': 12.53,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },  # 7,920 rows used

    "SSH-Patator": {
        'Flow Duration': 12113528.00,
        'Total Length of Fwd Packets': 2008.00,
        'Bwd Packet Length Max': 976.00,
        'Bwd Packet Length Mean': 85.78,
        'Bwd Packet Length Std': 220.24,
        'Flow IAT Std': 630656.76,
        'Flow IAT Max': 2384126.00,
        'Fwd IAT Total': 10200000.00,
        'Fwd IAT Std': 898939.77,
        'Fwd IAT Max': 2401297.00,
        'Max Packet Length': 976.00,
        'Packet Length Mean': 88.02,
        'Packet Length Std': 189.59,
        'Packet Length Variance': 35944.47,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 89.68,
        'Avg Bwd Segment Size': 85.78,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },  # 2,979 rows used

    "DoS Hulk": {
        'Flow Duration': 97415296.00,
        'Total Length of Fwd Packets': 356.00,
        'Bwd Packet Length Max': 5792.00,
        'Bwd Packet Length Mean': 1932.50,
        'Bwd Packet Length Std': 2179.55,
        'Flow IAT Std': 26100000.00,
        'Flow IAT Max': 97300000.00,
        'Fwd IAT Total': 97400000.00,
        'Fwd IAT Std': 37300000.00,
        'Fwd IAT Max': 97300000.00,
        'Max Packet Length': 5792.00,
        'Packet Length Mean': 853.29,
        'Packet Length Std': 1665.57,
        'Packet Length Variance': 2774137.61,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 918.92,
        'Avg Bwd Segment Size': 1932.50,
        'Idle Mean': 85900000.00,
        'Idle Max': 97300000.00,
        'Idle Min': 85900000.00,
    },  # 165,141 rows used

    "Bot": {
        'Flow Duration': 54398.00,
        'Total Length of Fwd Packets': 206.00,
        'Bwd Packet Length Max': 128.00,
        'Bwd Packet Length Mean': 7.37,
        'Bwd Packet Length Std': 13.30,
        'Flow IAT Std': 6076.75,
        'Flow IAT Max': 52220.00,
        'Fwd IAT Total': 54398.00,
        'Fwd IAT Std': 13565.21,
        'Fwd IAT Max': 53613.00,
        'Max Packet Length': 194.00,
        'Packet Length Mean': 31.14,
        'Packet Length Std': 68.12,
        'Packet Length Variance': 4640.46,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 32.78,
        'Avg Bwd Segment Size': 7.37,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },  # 1,474 rows used

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
st.info("Choisissez un preset OU entrez vos propres valeurs")

# Preset selector
preset_name = st.selectbox(
    "Charger un preset d'attaque (optionnel)",
    list(PRESETS.keys()),
    key="preset_selector"
)
preset = PRESETS[preset_name]

# ─────────────────────────────────────────────────────────────
# KEY FIX: embed preset_name in widget key.
# When the preset changes, the widget keys change, so Streamlit
# creates new widgets with the new default values instead of
# reusing the stale session_state values from the previous preset.
# ─────────────────────────────────────────────────────────────
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
            key=f"input_{preset_name}_{feat}"
        )

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
if st.button("🔍 Prédire"):
    full_row = {f: 0.0 for f in all_feat}
    for f, v in inputs.items():
        if f in full_row:
            full_row[f] = v

    X_df = pd.DataFrame([full_row], columns=all_feat)
    X_scaled = scaler.transform(X_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_feat)
    X_sel = X_scaled_df[selected].values

    pred  = model.predict(X_sel)[0]
    label = encoder.inverse_transform([pred])[0]
    proba = model.predict_proba(X_sel)[0]
    conf  = round(float(np.max(proba)) * 100, 1)

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

    st.markdown("---")
    st.subheader("Top 5 probabilités")
    class_labels = encoder.inverse_transform(model.classes_)
    proba_df = pd.DataFrame({
        'Classe'      : class_labels,
        'Probabilité' : np.round(proba, 4)
    }).sort_values('Probabilité', ascending=False) \
     .head(5).reset_index(drop=True)
    st.dataframe(proba_df, hide_index=True, use_container_width=True)

    with st.expander("🔧 Debug (valeurs envoyées au modèle)"):
        debug_df = pd.DataFrame({
            'Feature': selected,
            'Raw value': [inputs[f] for f in selected],
            'Scaled value': X_sel[0]
        })
        st.dataframe(debug_df, hide_index=True, use_container_width=True)
