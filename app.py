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

    "BENIGN": {  # 1,492,717 training samples
        'Flow Duration': 34382.00,
        'Total Length of Fwd Packets': 68.00,
        'Bwd Packet Length Max': 91.00,
        'Bwd Packet Length Mean': 83.00,
        'Bwd Packet Length Std': 0.00,
        'Flow IAT Std': 158.48,
        'Flow IAT Max': 31233.00,
        'Fwd IAT Total': 4.00,
        'Fwd IAT Std': 0.00,
        'Fwd IAT Max': 4.00,
        'Max Packet Length': 98.00,
        'Packet Length Mean': 61.20,
        'Packet Length Std': 30.60,
        'Packet Length Variance': 936.33,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 77.50,
        'Avg Bwd Segment Size': 83.00,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },

    "Bot": {  # 1,364 training samples
        'Flow Duration': 71485.00,
        'Total Length of Fwd Packets': 6.00,
        'Bwd Packet Length Max': 6.00,
        'Bwd Packet Length Mean': 6.00,
        'Bwd Packet Length Std': 0.00,
        'Flow IAT Std': 27626.54,
        'Flow IAT Max': 69327.50,
        'Fwd IAT Total': 71485.00,
        'Fwd IAT Std': 1703.77,
        'Fwd IAT Max': 70689.00,
        'Max Packet Length': 6.00,
        'Packet Length Mean': 6.00,
        'Packet Length Std': 3.21,
        'Packet Length Variance': 10.29,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 9.00,
        'Avg Bwd Segment Size': 6.00,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },

    "DDoS": {  # 89,610 training samples
        'Flow Duration': 1875296.00,
        'Total Length of Fwd Packets': 26.00,
        'Bwd Packet Length Max': 5755.00,
        'Bwd Packet Length Mean': 1934.50,
        'Bwd Packet Length Std': 2189.77,
        'Flow IAT Std': 900336.55,
        'Flow IAT Max': 1872785.50,
        'Fwd IAT Total': 1709997.50,
        'Fwd IAT Std': 900336.55,
        'Fwd IAT Max': 1707070.50,
        'Max Packet Length': 5755.00,
        'Packet Length Mean': 833.93,
        'Packet Length Std': 1903.96,
        'Packet Length Variance': 3625073.79,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 898.08,
        'Avg Bwd Segment Size': 1934.50,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },

    "DoS GoldenEye": {  # 7,200 training samples
        'Flow Duration': 11580841.00,
        'Total Length of Fwd Packets': 378.00,
        'Bwd Packet Length Max': 3525.00,
        'Bwd Packet Length Mean': 1175.00,
        'Bwd Packet Length Std': 1762.50,
        'Flow IAT Std': 2068377.17,
        'Flow IAT Max': 6538672.00,
        'Fwd IAT Total': 6672179.50,
        'Fwd IAT Std': 2079852.17,
        'Fwd IAT Max': 6543110.50,
        'Max Packet Length': 3525.00,
        'Packet Length Mean': 470.55,
        'Packet Length Std': 1166.36,
        'Packet Length Variance': 1360400.26,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 516.09,
        'Avg Bwd Segment Size': 1175.00,
        'Idle Mean': 6538412.50,
        'Idle Max': 6538672.00,
        'Idle Min': 6535094.50,
    },

    "DoS Hulk": {  # 120,992 training samples
        'Flow Duration': 86371466.00,
        'Total Length of Fwd Packets': 354.00,
        'Bwd Packet Length Max': 5792.00,
        'Bwd Packet Length Mean': 1932.50,
        'Bwd Packet Length Std': 2178.09,
        'Flow IAT Std': 25700000.00,
        'Flow IAT Max': 85900000.00,
        'Fwd IAT Total': 86400000.00,
        'Fwd IAT Std': 37200000.00,
        'Fwd IAT Max': 85900000.00,
        'Max Packet Length': 5792.00,
        'Packet Length Mean': 852.57,
        'Packet Length Std': 1655.93,
        'Packet Length Variance': 2742091.04,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 918.15,
        'Avg Bwd Segment Size': 1932.50,
        'Idle Mean': 85600000.00,
        'Idle Max': 85900000.00,
        'Idle Min': 85600000.00,
    },

    "DoS Slowhttptest": {  # 3,660 training samples
        'Flow Duration': 63120978.50,
        'Total Length of Fwd Packets': 0.00,
        'Bwd Packet Length Max': 0.00,
        'Bwd Packet Length Mean': 0.00,
        'Bwd Packet Length Std': 0.00,
        'Flow IAT Std': 11900000.00,
        'Flow IAT Max': 32100000.00,
        'Fwd IAT Total': 63100000.00,
        'Fwd IAT Std': 11900000.00,
        'Fwd IAT Max': 32100000.00,
        'Max Packet Length': 0.00,
        'Packet Length Mean': 0.00,
        'Packet Length Std': 0.00,
        'Packet Length Variance': 0.00,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 0.00,
        'Avg Bwd Segment Size': 0.00,
        'Idle Mean': 18700000.00,
        'Idle Max': 32100000.00,
        'Idle Min': 8015915.00,
    },

    "DoS slowloris": {  # 3,770 training samples
        'Flow Duration': 99999349.50,
        'Total Length of Fwd Packets': 16.00,
        'Bwd Packet Length Max': 0.00,
        'Bwd Packet Length Mean': 0.00,
        'Bwd Packet Length Std': 0.00,
        'Flow IAT Std': 12500000.00,
        'Flow IAT Max': 51300000.00,
        'Fwd IAT Total': 100000000.00,
        'Fwd IAT Std': 711018.50,
        'Fwd IAT Max': 51300000.00,
        'Max Packet Length': 8.00,
        'Packet Length Mean': 4.80,
        'Packet Length Std': 4.62,
        'Packet Length Variance': 21.33,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 6.00,
        'Avg Bwd Segment Size': 0.00,
        'Idle Mean': 19200000.00,
        'Idle Max': 51300000.00,
        'Idle Min': 5769388.50,
    },

    "FTP-Patator": {  # 4,152 training samples
        'Flow Duration': 8695781.00,
        'Total Length of Fwd Packets': 102.00,
        'Bwd Packet Length Max': 34.00,
        'Bwd Packet Length Mean': 12.53,
        'Bwd Packet Length Std': 14.55,
        'Flow IAT Std': 980766.88,
        'Flow IAT Max': 3108757.50,
        'Fwd IAT Total': 5731094.00,
        'Fwd IAT Std': 1320630.75,
        'Fwd IAT Max': 3035823.50,
        'Max Packet Length': 34.00,
        'Packet Length Mean': 11.60,
        'Packet Length Std': 12.52,
        'Packet Length Variance': 156.82,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 12.12,
        'Avg Bwd Segment Size': 12.53,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },

    "Heartbleed": {  # 8 training samples
        'Flow Duration': 119260706.50,
        'Total Length of Fwd Packets': 12264.00,
        'Bwd Packet Length Max': 15204.00,
        'Bwd Packet Length Mean': 3761.08,
        'Bwd Packet Length Std': 2370.38,
        'Flow IAT Std': 152848.54,
        'Flow IAT Max': 995306.00,
        'Fwd IAT Total': 119000000.00,
        'Fwd IAT Std': 200846.91,
        'Fwd IAT Max': 996643.00,
        'Max Packet Length': 15204.00,
        'Packet Length Mean': 1615.42,
        'Packet Length Std': 2422.17,
        'Packet Length Variance': 5866961.82,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 1615.75,
        'Avg Bwd Segment Size': 3761.08,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },

    "Infiltration": {  # 25 training samples
        'Flow Duration': 69714525.00,
        'Total Length of Fwd Packets': 5821.00,
        'Bwd Packet Length Max': 6.00,
        'Bwd Packet Length Mean': 6.00,
        'Bwd Packet Length Std': 0.00,
        'Flow IAT Std': 3456029.56,
        'Flow IAT Max': 29500000.00,
        'Fwd IAT Total': 69700000.00,
        'Fwd IAT Std': 972656.12,
        'Fwd IAT Max': 24800000.00,
        'Max Packet Length': 1181.00,
        'Packet Length Mean': 137.50,
        'Packet Length Std': 271.31,
        'Packet Length Variance': 73607.76,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 137.80,
        'Avg Bwd Segment Size': 6.00,
        'Idle Mean': 14700000.00,
        'Idle Max': 24800000.00,
        'Idle Min': 9785751.00,
    },

    "PortScan": {  # 63,485 training samples
        'Flow Duration': 50.00,
        'Total Length of Fwd Packets': 0.00,
        'Bwd Packet Length Max': 6.00,
        'Bwd Packet Length Mean': 6.00,
        'Bwd Packet Length Std': 0.00,
        'Flow IAT Std': 0.00,
        'Flow IAT Max': 50.00,
        'Fwd IAT Total': 0.00,
        'Fwd IAT Std': 0.00,
        'Fwd IAT Max': 0.00,
        'Max Packet Length': 6.00,
        'Packet Length Mean': 2.40,
        'Packet Length Std': 2.31,
        'Packet Length Variance': 5.33,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 3.00,
        'Avg Bwd Segment Size': 6.00,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },

    "SSH-Patator": {  # 2,253 training samples
        'Flow Duration': 12021992.00,
        'Total Length of Fwd Packets': 2008.00,
        'Bwd Packet Length Max': 976.00,
        'Bwd Packet Length Mean': 85.78,
        'Bwd Packet Length Std': 220.24,
        'Flow IAT Std': 625820.81,
        'Flow IAT Max': 2355950.00,
        'Fwd IAT Total': 10100000.00,
        'Fwd IAT Std': 892094.44,
        'Fwd IAT Max': 2366399.00,
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
    },

    "Web Attack — Brute Force": {  # 1,029 training samples
        'Flow Duration': 5578909.00,
        'Total Length of Fwd Packets': 0.00,
        'Bwd Packet Length Max': 0.00,
        'Bwd Packet Length Mean': 0.00,
        'Bwd Packet Length Std': 0.00,
        'Flow IAT Std': 3131488.75,
        'Flow IAT Max': 5424342.00,
        'Fwd IAT Total': 5506513.00,
        'Fwd IAT Std': 3834960.50,
        'Fwd IAT Max': 5424342.00,
        'Max Packet Length': 0.00,
        'Packet Length Mean': 0.00,
        'Packet Length Std': 0.00,
        'Packet Length Variance': 0.00,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 0.00,
        'Avg Bwd Segment Size': 0.00,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },

    "Web Attack — Sql Injection": {  # 15 training samples
        'Flow Duration': 5006127.00,
        'Total Length of Fwd Packets': 447.00,
        'Bwd Packet Length Max': 530.00,
        'Bwd Packet Length Mean': 132.50,
        'Bwd Packet Length Std': 265.00,
        'Flow IAT Std': 1666210.00,
        'Flow IAT Max': 5000049.00,
        'Fwd IAT Total': 4355.00,
        'Fwd IAT Std': 1734.43,
        'Fwd IAT Max': 3391.00,
        'Max Packet Length': 530.00,
        'Packet Length Mean': 108.56,
        'Packet Length Std': 216.41,
        'Packet Length Variance': 46831.28,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 122.12,
        'Avg Bwd Segment Size': 132.50,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
    },

    "Web Attack — XSS": {  # 456 training samples
        'Flow Duration': 5387639.50,
        'Total Length of Fwd Packets': 0.00,
        'Bwd Packet Length Max': 0.00,
        'Bwd Packet Length Mean': 0.00,
        'Bwd Packet Length Std': 0.00,
        'Flow IAT Std': 3098500.38,
        'Flow IAT Max': 5367181.50,
        'Fwd IAT Total': 5381989.00,
        'Fwd IAT Std': 3794574.38,
        'Fwd IAT Max': 5367181.50,
        'Max Packet Length': 0.00,
        'Packet Length Mean': 0.00,
        'Packet Length Std': 0.00,
        'Packet Length Variance': 0.00,
        'FIN Flag Count': 0.00,
        'Average Packet Size': 0.00,
        'Avg Bwd Segment Size': 0.00,
        'Idle Mean': 0.00,
        'Idle Max': 0.00,
        'Idle Min': 0.00,
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
    "Charger un preset (BENIGN ou attaque)",
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
