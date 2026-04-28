import os
import zipfile
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, SaltRemover
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold


# ── Directory setup ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


def data(filename):    return os.path.join(DATA_DIR, filename)
def model(filename):   return os.path.join(MODELS_DIR, filename)
def plot(filename):    return os.path.join(PLOTS_DIR, filename)
def result(filename):  return os.path.join(RESULTS_DIR, filename)


# ── Functions ─────────────────────────────────────────────────────────────────
def lipinski(smiles):
    """Calculates Lipinski descriptors for a list of SMILES strings."""
    moldata = [Chem.MolFromSmiles(elem) for elem in smiles]

    baseData = []
    for mol in moldata:
        if mol is not None:
            row = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol)
            ]
        else:
            row = [np.nan, np.nan, np.nan, np.nan]
        baseData.append(row)

    return pd.DataFrame(data=baseData, columns=["MW", "LogP", "NumHDonors", "NumHAcceptors"])


def classify_activity(value):
    """Assigns bioactivity class based on IC50 value in nM."""
    value = float(value)
    if value >= 10000:
        return "inactive"
    elif value <= 1000:
        return "active"
    return "intermediate"


def norm_value(input_df):
    """Normalizes standard_value by capping at 100,000,000."""
    input_df = input_df.copy()
    input_df["standard_value_norm"] = input_df["standard_value"].apply(
        lambda x: min(float(x), 100_000_000)
    )
    return input_df.drop("standard_value", axis=1)


def pIC50(input_df):
    """Converts normalized IC50 in nM to pIC50."""
    input_df = input_df.copy()
    input_df["pIC50"] = input_df["standard_value_norm"].apply(
        lambda x: -np.log10(x * 1e-9)
    )
    return input_df.drop("standard_value_norm", axis=1)


def mannwhitney(descriptor, df_2class, verbose=False):
    """Mann-Whitney U test for active vs inactive compounds."""
    active   = df_2class[df_2class.bioactivity_class == "active"][descriptor]
    inactive = df_2class[df_2class.bioactivity_class == "inactive"][descriptor]

    stat, p = mannwhitneyu(active, inactive)
    alpha = 0.05
    interpretation = (
        "Different distribution (reject H0)"
        if p <= alpha
        else "Same distribution (fail to reject H0)"
    )

    results = pd.DataFrame({
        "Descriptor":     descriptor,
        "Statistics":     stat,
        "p":              p,
        "alpha":          alpha,
        "Interpretation": interpretation
    }, index=[0])

    results.to_csv(data(f"ace_mannwhitneyu_{descriptor}.csv"), index=False)

    if verbose:
        print(results)

    return results


def boxplot_with_mannwhitney(descriptor, df_2class, ylabel=None):
    """Draws a boxplot and runs Mann-Whitney U test."""
    plt.figure(figsize=(5.5, 5.5))
    sns.boxplot(x="bioactivity_class", y=descriptor, data=df_2class)
    plt.xlabel("Bioactivity class", fontsize=14, fontweight="bold")
    plt.ylabel(ylabel or descriptor, fontsize=14, fontweight="bold")
    plt.savefig(plot(f"ace_plot_{descriptor}.pdf"))
    print(mannwhitney(descriptor, df_2class))


remover = SaltRemover.SaltRemover()

def clean_smiles_with_rdkit(smiles):
    """Removes salts from SMILES using RDKit SaltRemover."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    cleaned_mol = remover.StripMol(mol, dontRemoveEverything=True)
    if cleaned_mol is None:
        return None
    return Chem.MolToSmiles(cleaned_mol)


# ── 1. Target Search ─────────────────────────────────────────────────────────
target = new_client.target
targets = pd.DataFrame.from_dict(target.search("ACE"))
print(targets[["target_chembl_id", "pref_name", "organism"]])

selected_target = targets.target_chembl_id[5]  # Homo sapiens ACE (CHEMBL1808)
print(f"Selected target: {selected_target}")

# ── 2. Fetch IC50 Bioactivity Data ───────────────────────────────────────────
activity = new_client.activity
df = pd.DataFrame.from_dict(
    activity.filter(target_chembl_id=selected_target)
            .filter(standard_type="IC50")[:500]
)
print(f"Raw data shape: {df.shape}")

if df.empty:
    raise ValueError("No data fetched. Try a different target index.")

df.to_csv(data("ace_bioactivity_data.csv"), index=False)

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
df2 = df[df.standard_value.notna()].copy()
df2 = df2[df2.canonical_smiles.notna()].copy()
df2["bioactivity_class"] = df2["standard_value"].astype(float).map(classify_activity)

df3 = df2[["molecule_chembl_id", "canonical_smiles", "standard_value", "bioactivity_class"]]
df3 = df3.drop_duplicates(["canonical_smiles"])
df3.to_csv(data("ace_bioactivity_preprocessed_data.csv"), index=False)

print(f"Processed data: {df3.shape[0]} rows")
print(f"Class distribution:\n{df3['bioactivity_class'].value_counts()}")

# ── 4. SMILES Cleaning + Lipinski Descriptors + pIC50 ────────────────────────
df = pd.read_csv(data("ace_bioactivity_preprocessed_data.csv"))

df["canonical_smiles"] = df["canonical_smiles"].apply(clean_smiles_with_rdkit)
df = df[df["canonical_smiles"].notna()].reset_index(drop=True)

df_lipinski = lipinski(df.canonical_smiles)
df_combined = pd.concat([df, df_lipinski], axis=1)
df_final    = pIC50(norm_value(df_combined))

print(df_final)
print(df_final.describe())
df_final.to_csv(data("ace_bioactivity_final.csv"), index=False)

# ── 5. Exploratory Data Analysis ─────────────────────────────────────────────
df_2class = df_final[df_final.bioactivity_class != "intermediate"]
df_2class.to_csv(data("ace_bioactivity_2class.csv"), index=False)

sns.set(style="ticks")

# Plot 1: Bioactivity class distribution
plt.figure(figsize=(5.5, 5.5))
sns.countplot(x="bioactivity_class", data=df_2class, edgecolor="black")
plt.xlabel("Bioactivity class", fontsize=14, fontweight="bold")
plt.ylabel("Frequency", fontsize=14, fontweight="bold")
plt.savefig(plot("ace_plot_bioactivity_class.pdf"))

# Plot 2: MW vs LogP scatter plot
plt.figure(figsize=(5.5, 5.5))
sns.scatterplot(x="MW", y="LogP", data=df_2class, hue="bioactivity_class",
                size="pIC50", edgecolor="black", alpha=0.7)
plt.xlabel("MW", fontsize=14, fontweight="bold")
plt.ylabel("LogP", fontsize=14, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig(plot("ace_plot_MW_vs_LogP.pdf"))

plt.show()

# ── 6. Box Plots + Mann-Whitney U Test ───────────────────────────────────────
for descriptor in ["pIC50", "MW", "LogP", "NumHDonors", "NumHAcceptors"]:
    boxplot_with_mannwhitney(descriptor, df_2class)

plt.show()

# ── 7. Zip Results ────────────────────────────────────────────────────────────
all_files = (
    [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    + [os.path.join(PLOTS_DIR, f) for f in os.listdir(PLOTS_DIR) if f.endswith(".pdf")]
)

with zipfile.ZipFile(result("ace_results.zip"), "w") as zipf:
    for f in all_files:
        zipf.write(f, arcname=os.path.basename(f))

print(f"Zipped {len(all_files)} files → results/ace_results.zip")

# ── 8. Fingerprint Descriptor Calculation (RDKit MACCS Keys) ─────────────────
df3 = pd.read_csv(data("ace_bioactivity_final.csv"))

fingerprints = []
valid_idx = []

for i, smi in enumerate(df3.canonical_smiles):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is not None:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
        valid_idx.append(i)

df3_X = pd.DataFrame(fingerprints, columns=[f"MACCS_{i}" for i in range(167)])
df3_Y = df3["pIC50"].iloc[valid_idx].reset_index(drop=True)

print(f"X matrix: {df3_X.shape}")
print(f"Y variable: {df3_Y.shape}")

# ── 9. Prepare Final Dataset ──────────────────────────────────────────────────
dataset = pd.concat([df3_X, df3_Y], axis=1)
dataset.to_csv(data("ace_bioactivity_descriptors.csv"), index=False)
print(f"Final dataset saved: {dataset.shape}")

# ── 10. Random Forest Regression Model ───────────────────────────────────────
df_ml = pd.read_csv(data("ace_bioactivity_descriptors.csv"))

X = df_ml.drop("pIC50", axis=1)
Y = df_ml["pIC50"]

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# Remove low variance features
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = selection.fit_transform(X)
print(f"X shape after variance threshold: {X.shape}")

# Train/test split (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Build Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, Y_train)

r2 = model_rf.score(X_test, Y_test)
print(f"R² score (test set): {r2:.4f}")

Y_pred = model_rf.predict(X_test)

# ── 11. Scatter Plot: Experimental vs Predicted pIC50 ────────────────────────
sns.set(color_codes=True)
sns.set_style("white")

plt.figure(figsize=(5, 5))
ax = sns.regplot(x=Y_test, y=Y_pred, scatter_kws={"alpha": 0.4})
ax.set_xlabel("Experimental pIC50", fontsize=14, fontweight="bold")
ax.set_ylabel("Predicted pIC50", fontsize=14, fontweight="bold")
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
plt.savefig(plot("ace_plot_experimental_vs_predicted.pdf"))
plt.show()

# ── 12. Save Model and Descriptor List ───────────────────────────────────────
with open(model("ace_model.pkl"), "wb") as f:
    pickle.dump(model_rf, f)

selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
selector.fit(df_ml.drop("pIC50", axis=1))
selected_features = df_ml.drop("pIC50", axis=1).columns[selector.get_support()]
pd.DataFrame(columns=selected_features).to_csv(model("ace_descriptor_list.csv"), index=False)

print("Model and descriptor list saved.")
print("ace_model.pkl and ace_descriptor_list.csv are ready for app.py")