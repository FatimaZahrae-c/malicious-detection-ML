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


# Presets derived from medians of the actual TRAINING data
# (inverse-transformed from scaled X_train, not raw CSVs)
PRESETS = {
    "— Manuel (tout à zéro) —": {f: 0.0 for f in selected},

    "PortScan": {
        'Flow Duration': 50.0,
        'Total Length of Fwd Packets': 0.0,
        'Bwd Packet Length Max': 6.0,
        'Bwd Packet Length Mean': 6.0,
        'Bwd Packet Length Std': 0.0,
        'Flow IAT Std': 0.0,
        'Flow IAT Max': 50.0,
        'Fwd IAT Total': 0.0,
        'Fwd IAT Std': 0.0,
        'Fwd IAT Max': 0.0,
        'Max Packet Length': 6.0,
        'Packet Length Mean': 2.40,
        'Packet Length Std': 2.31,
        'Packet Length Variance': 5.33,
        'FIN Flag Count': 0.0,
        'Average Packet Size': 3.0,
        'Avg Bwd Segment Size': 6.0,
        'Idle Mean': 0.0,
        'Idle Max': 0.0,
        'Idle Min': 0.0,
    },

    "FTP-Patator": {
        'Flow Duration': 8695781.0,
        'Total Length of Fwd Packets': 102.0,
        'Bwd Packet Length Max': 34.0,
        'Bwd Packet Length Mean': 12.53,
        'Bwd Packet Length Std': 14.55,
        'Flow IAT Std': 980766.88,
        'Flow IAT Max': 3108757.50,
        'Fwd IAT Total': 5731094.0,
        'Fwd IAT Std': 1320630.75,
        'Fwd IAT Max': 3035823.50,
        'Max Packet Length': 34.0,
        'Packet Length Mean': 11.60,
        'Packet Length Std': 12.52,
        'Packet Length Variance': 156.82,
        'FIN Flag Count': 0.0,
        'Average Packet Size': 12.12,
        'Avg Bwd Segment Size': 12.53,
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
        'Flow Duration': 71485.0,
        'Total Length of Fwd Packets': 6.0,
        'Bwd Packet Length Max': 6.0,
        'Bwd Packet Length Mean': 6.0,
        'Bwd Packet Length Std': 0.0,
        'Flow IAT Std': 27626.54,
        'Flow IAT Max': 69327.50,
        'Fwd IAT Total': 71485.0,
        'Fwd IAT Std': 1703.77,
        'Fwd IAT Max': 70689.0,
        'Max Packet Length': 6.0,
        'Packet Length Mean': 6.0,
        'Packet Length Std': 3.21,
        'Packet Length Variance': 10.29,
        'FIN Flag Count': 0.0,
        'Average Packet Size': 9.0,
        'Avg Bwd Segment Size': 6.0,
        'Idle Mean': 0.0,
        'Idle Max': 0.0,
        'Idle Min': 0.0,
    },
}


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

preset_name = st.selectbox(
    "Charger un preset d'attaque (optionnel)",
    list(PRESETS.keys()),
    key="preset_selector"
)
preset = PRESETS[preset_name]

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
