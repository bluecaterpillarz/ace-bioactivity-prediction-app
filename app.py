import os
import pickle
import base64
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, SaltRemover

# ── Directory setup ───────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

remover = SaltRemover.SaltRemover()


def model_path(filename):
    return os.path.join(MODELS_DIR, filename)


# ── Helper Functions ──────────────────────────────────────────────────────────
def clean_smiles(smiles):
    """Removes salts from SMILES using RDKit SaltRemover."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    cleaned = remover.StripMol(mol, dontRemoveEverything=True)
    if cleaned is None:
        return None
    return Chem.MolToSmiles(cleaned)


def compute_maccs(smiles_list):
    """Computes MACCS fingerprints for a list of SMILES strings."""
    fingerprints = []
    valid_idx = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((167,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
            valid_idx.append(i)

    df_fp = pd.DataFrame(fingerprints, columns=[f"MACCS_{i}" for i in range(167)])
    return df_fp, valid_idx


def filedownload(df):
    """Generates a CSV download link."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="ace_predictions.csv">📥 Download Predictions</a>'


def predict(smiles_list, molecule_names):
    """Cleans SMILES, computes descriptors, applies model and returns predictions."""

    # Clean SMILES
    cleaned = [clean_smiles(s) for s in smiles_list]

    # Compute MACCS fingerprints
    df_fp, valid_idx = compute_maccs(cleaned)

    # Load descriptor list and align columns
    descriptor_list = pd.read_csv(model_path("ace_descriptor_list.csv")).columns.tolist()
    df_fp = df_fp.reindex(columns=descriptor_list, fill_value=0)

    # Load model and predict
    rf_model = pickle.load(open(model_path("ace_model.pkl"), "rb"))
    predictions = rf_model.predict(df_fp)

    # Build result dataframe
    valid_names = [molecule_names[i] for i in valid_idx]
    results = pd.DataFrame({
        "Molecule": valid_names,
        "SMILES":   [cleaned[i] for i in valid_idx],
        "pIC50":    predictions.round(3)
    })

    return results, df_fp


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACE Inhibitor Bioactivity Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ ACE Inhibitor Bioactivity Prediction App")
st.markdown("""
This app predicts the **pIC50** bioactivity of molecules against
**Angiotensin-Converting Enzyme (ACE)** (CHEMBL1808).

ACE is a key drug target for **hypertension and heart failure** treatment.
ACE inhibitors (e.g. enalapril, lisinopril) are among the most widely used cardiovascular drugs.

- Descriptors: **RDKit MACCS Keys** (167 bits)
- Model: **Random Forest Regressor** (R² = 0.70)

---
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("1. Input Molecules")

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Upload .txt file", "Enter SMILES manually"]
)

smiles_list = []
name_list = []

if input_method == "Upload .txt file":
    st.sidebar.markdown("""
**File format:** `SMILES molecule_name` (space-separated, no header)

Example:
CC(=O)Oc1ccccc1C(=O)O aspirin
CCO ethanol
""")
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["txt"])

    if uploaded_file:
        df_input = pd.read_table(uploaded_file, sep=" ", header=None)
        smiles_list = df_input.iloc[:, 0].tolist()
        name_list   = df_input.iloc[:, 1].tolist() if df_input.shape[1] > 1 else [f"Mol_{i+1}" for i in range(len(smiles_list))]

        st.subheader("📋 Uploaded Molecules")
        st.write(pd.DataFrame({"Name": name_list, "SMILES": smiles_list}))

else:
    raw_input = st.sidebar.text_area(
        "Enter SMILES (one per line):",
        placeholder="CC(=O)Oc1ccccc1C(=O)O\nCCO"
    )
    if raw_input.strip():
        lines = raw_input.strip().split("\n")
        smiles_list = [l.split()[0] for l in lines if l.strip()]
        name_list   = [f"Mol_{i+1}" for i in range(len(smiles_list))]

        st.subheader("📋 Input Molecules")
        st.write(pd.DataFrame({"Name": name_list, "SMILES": smiles_list}))

# ── Prediction ────────────────────────────────────────────────────────────────
if st.sidebar.button("🔬 Predict"):
    if not smiles_list:
        st.warning("Please provide at least one SMILES string.")
    else:
        if not os.path.exists(model_path("ace_model.pkl")) or \
           not os.path.exists(model_path("ace_descriptor_list.csv")):
            st.error("Model files not found in models/. Please run main.py first.")
        else:
            with st.spinner("Computing descriptors and predicting..."):
                results, df_fp = predict(smiles_list, name_list)

            st.subheader("🎯 Prediction Results")
            st.write(results)
            st.markdown(filedownload(results), unsafe_allow_html=True)

            st.subheader("🧪 MACCS Fingerprints (descriptor matrix)")
            st.write(df_fp)
            st.write(f"Shape: {df_fp.shape}")

else:
    st.info("👈 Add molecules in the sidebar and click **Predict** to start.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**Data source:** [ChEMBL Database](https://www.ebi.ac.uk/chembl/) | 
**Target:** ACE (CHEMBL1808) | 
**Descriptors:** RDKit MACCS Keys
""")