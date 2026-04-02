import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
import time
import requests
from streamlit_lottie import st_lottie
import py3Dmol

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="🧪 Drug Toxicity Predictor",
    layout="wide"
)
# ================= STYLES =================
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    color: black;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
# ================= CUSTOM UI =================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
h1 {
    color: #00FFAA;
    text-align: center;
}
.stButton>button {
    background: linear-gradient(90deg, #00FFAA, #00CFFF);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open(r"C:\\Users\\noyon\\OneDrive\\Desktop\\ToxPredictor\\model\\tox.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

xgb = model["xgb"]
lgb = model["lgb"]
cat = model["cat"]
scaler = model["scaler"]
selector = model["selector"]
selected_features = model["selected_features"]
descriptor_cols = model["descriptor_cols"]

morgan_gen = GetMorganGenerator(radius=2, fpSize=256)
# ================= LOTTIE =================
def load_lottie(url):
    return requests.get(url).json()

lottie = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
st_lottie(lottie, height=200)

# ================= FEATURE ENGINEERING =================
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = []
    for col in descriptor_cols:
        try:
            val = getattr(Descriptors, col)(mol)
            if val is None or np.isnan(val) or np.isinf(val):
                val = 0
        except:
            val = 0
        desc.append(val)

    fp = morgan_gen.GetFingerprint(mol)
    fp_array = np.zeros((256,))
    DataStructs.ConvertToNumpyArray(fp, fp_array)

    return np.nan_to_num(np.concatenate([desc, fp_array]))

# ================= ENSEMBLE =================
def ensemble_proba(X):
    return (
        0.4 * xgb.predict_proba(X) +
        0.3 * lgb.predict_proba(X) +
        0.3 * cat.predict_proba(X)
    )

# ================= 3D VIEW =================
def show_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    mblock = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mblock, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()

    return viewer._make_html()

# ================= UI =================
st.title("🧪 Drug Toxicity Predictor")
st.markdown("### Predict toxicity using AI + Chemistry 🧠")

smiles = st.text_input("Enter SMILES string:", "CCO")
view_mode = st.radio("View Mode", ["2D", "3D"])

if st.button("🚀 Predict Toxicity"):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES string")
        st.stop()

    col1, col2 = st.columns(2)

    # ===== Molecule =====
    with col1:
        st.subheader("🧬 Molecular Structure")
        if view_mode == "2D":
            img = Draw.MolToImage(mol)
            st.image(img, width="stretch")
        else:
            html = show_3d(smiles)
            st.components.v1.html(html, height=400)

    # ===== Properties =====
    with col2:
        st.subheader("💊 Molecular Properties")

        mol_wt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        qed_score = QED.qed(mol)
        h_donors = Descriptors.NumHDonors(mol)

        st.metric("MolWt", round(mol_wt, 2))
        st.metric("LogP", round(logp, 2))
        st.metric("TPSA", round(tpsa, 2))
        st.metric("QED", round(qed_score, 3))
        st.metric("H-donors", h_donors)
    # ===== Feature Pipeline =====
    feat = featurize(smiles)
    feat_scaled = scaler.transform([feat])
    feat_selected = selector.transform(feat_scaled)
    feat_df = pd.DataFrame(feat_selected, columns=selected_features)
    # ===== Prediction =====
    probs = ensemble_proba(feat_df)[0]
    pred = np.argmax(probs)
    confidence = float(np.max(probs))

    labels = ["HIGH", "MEDIUM", "LOW"]
    colors = ["#FF4B4B", "#FFA500", "#00FF7F"]


    st.markdown("---")
    st.subheader("🔬 Prediction Result")

    

    # ===== Probability Chart =====
    
    if mol_wt > 300 and h_donors >= 5 and logp < 1:
        st.info("Sugar-like molecule detected → LOW toxicity")
        pred = 2

    
    safe_smiles = [
    "O","N", "O=C=O", "[K+].[Cl-]","C(C(C1C(C(C(O1)(CO)O)O)O)O)O" ,    
    "[Na+].[Cl-]","C(C1C(C(C(C(O1)O)O)O)O)O", "C(C(CO)O)O", 'C(C(=O)O)C(CC(=O)O)(C(=O)O)O',"C(C(C(C(C(CO)O)O)O)O)O",
    ]
    if confidence < 0.6:
        # nitrile
        if "C#N" in smiles and pred == 2:
            pred = 1
        
    if smiles in safe_smiles:
        st.info("Prediction probabilities might mismatch, but it's a safe compound !")
        pred = 2
    toxic_patterns = ["[C-]#N", "N=N", "[N+](=O)[O-]"]  # cyanide, azo, nitro

    if any(pattern in smiles for pattern in toxic_patterns):
        pred = 0

    if "C#N" in smiles:
        if pred == 2:
            pred = 1

    if qed_score > 0.7 and logp < 2 and pred == 0:
        pred = 1  

    st.markdown(
        f"<h1 style='color:{colors[pred]}'>Toxicity: {labels[pred]}</h1>",
        unsafe_allow_html=True
    )

   # ===== ANIMATED PROGRESS =====
    progress = st.progress(0)
    for i in range(int(confidence * 100)):
        progress.progress(i + 1)
        time.sleep(0.01)

    

     # ===== LIVE GRAPH =====
    st.subheader("📊 Prediction Probabilities")
    colors_bar = ["#FAE251","#D75656", "#BD114A"]
    fig, ax = plt.subplots(figsize=(2.6, 1.8), dpi=100)

    bars = ax.bar(labels, [0,0,0], color=colors_bar, width=0.4)
    ax.set_ylim(0,1)
    ax.set_ylabel("", fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # enable light grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)


    plt.tight_layout(pad=0.1)
    plot = st.pyplot(fig,use_container_width=False)

    # animate bars
    for i in range(12):
        temp = probs * (i+1)/12
        for bar, h in zip(bars, temp):
            bar.set_height(h)
        plot.pyplot(fig)
        time.sleep(0.05)


    # ===== Uncertainty =====
    entropy = -np.sum(probs * np.log(probs + 1e-9))

    st.subheader("🧠 Model Confidence")
    st.write(f"Entropy: {round(entropy, 3)}")

    if entropy > 1:
        st.warning("⚠️ High uncertainty")
    else:
        st.success("✅ Model confident")

   
    
    # ===== SHAP =====
    st.subheader("🔍 SHAP Explanation")

    try:
        background = model["background"]

        explainer = shap.Explainer(
            lambda x: xgb.predict_proba(x),
            background
        )

        shap_values = explainer(feat_df)

        shap_val = shap_values.values[0, :, pred]
        feature_names = list(selected_features)
        exp = shap.Explanation(
            values=shap_val,
            base_values=shap_values[0].base_values[pred],
            data=feat_df.iloc[0].values,
            feature_names=feature_names
        )

        fig = plt.figure(figsize=(5,3))
        shap.plots.waterfall(exp, max_display=10, show=False)
        st.pyplot(fig)
        st.write("Red bars (+) → push prediction towards HIGH toxicity\nBlue bars (−) → push prediction towards LOW toxicity")
    except Exception as e:
        st.error(f"SHAP failed: {e}")

    # ===== Feature Insights =====
    st.subheader("🧠 Key Factors")

    
    try:
        top_idx = np.argsort(np.abs(shap_val))[::-1][:5]

        for i in top_idx:
            fname = selected_features[i]
            val = float(shap_val[i])
            if val> 0:
                st.write(f"🔺 {fname} increases toxicity")
            else:
                st.write(f"🔻 {fname} decreases toxicity")
    except Exception as e:
        st.error(f"Could not compute Key factors leading to toxicity: {e}")


    # ===== Footer =====
    st.markdown("---")
    st.info("⚠️ Model Note:")
    st.info("Prediction is based on learned chemical patterns and heuristic correction.")
    st.info("Toxicity also depends on dose, exposure route, and metabolism.")