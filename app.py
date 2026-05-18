import streamlit as st
import joblib, json
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
    model   = joblib.load("models/random_forest.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    with open("models/selected_features.json") as f:
        selected = json.load(f)
    all_feat = list(scaler.feature_names_in_)
    return model, scaler, encoder, selected, all_feat


model, scaler, encoder, selected, all_feat = load_model()

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

inputs = {}
col1, col2 = st.columns(2)
for i, feat in enumerate(selected):
    col = col1 if i % 2 == 0 else col2
    with col:
        inputs[feat] = st.number_input(
            feat, value=0.0, format="%.2f"
        )

if st.button("🔍 Prédire"):

    # Construire ligne complète 69 features
    full_row = {f: 0.0 for f in all_feat}
    for f, v in inputs.items():
        full_row[f] = v

    # Scaler sur 69 features
    X_df     = pd.DataFrame([full_row], columns=all_feat)
    X_scaled = scaler.transform(X_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_feat)

    # Sélectionner les 20 features
    X_sel = X_scaled_df[selected].values

    # Prédiction
    pred  = model.predict(X_sel)[0]
    label = encoder.inverse_transform([pred])[0]
    proba = model.predict_proba(X_sel)[0]
    conf  = round(float(np.max(proba)) * 100, 1)

    # Afficher résultat
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

    # Top 5 probabilités — CORRIGÉ
    st.markdown("---")
    st.subheader("Top 5 probabilités")

    # model.classes_ contient les indices encodés
    # encoder.inverse_transform les convertit en noms
    class_labels = encoder.inverse_transform(
        model.classes_
    )

    proba_df = pd.DataFrame({
        'Classe'      : class_labels,
        'Probabilité' : np.round(proba, 4)
    }).sort_values(
        'Probabilité', ascending=False
    ).head(5).reset_index(drop=True)

    st.dataframe(proba_df, hide_index=True)
