# ❤️ ACE Bioactivity Prediction App

This project predicts the **pIC50 bioactivity** of molecules against  
**Angiotensin-Converting Enzyme (ACE)** using machine learning.

It is an end-to-end cheminformatics pipeline including **data collection, preprocessing, feature engineering, model training, and deployment**.

---

## 🔬 Project Overview

- **Target:** ACE (Angiotensin-Converting Enzyme) – CHEMBL1808  
- **Data Source:** ChEMBL Database  
- **Descriptors:** RDKit MACCS Fingerprints (167 features)  
- **Model:** Random Forest Regressor  
- **Evaluation:** R² score on test set  
- **Deployment:** Streamlit web application  

---

## ⚙️ Pipeline

1. Fetch bioactivity data from ChEMBL  
2. Clean SMILES (salt removal using RDKit)  
3. Generate molecular descriptors (MACCS fingerprints)  
4. Perform exploratory data analysis (EDA)  
5. Apply statistical testing (Mann–Whitney U test)  
6. Train machine learning model (Random Forest)  
7. Deploy interactive web app with Streamlit  

---

## 📊 Features

- ✔ Automated data collection from ChEMBL  
- ✔ RDKit-based molecular preprocessing  
- ✔ Descriptor generation without external tools (no Java required)  
- ✔ Statistical analysis of bioactivity classes  
- ✔ Machine learning model for regression  
- ✔ Interactive prediction interface (Streamlit)  

---

## 🧪 Example Outputs

- Bioactivity class distribution  
- MW vs LogP scatter plots  
- Experimental vs Predicted pIC50 comparison  
- Descriptor matrix (MACCS fingerprints)  

---

## ▶️ How to Run

### 1. Clone the repository

    git clone https://github.com/your-username/ace-bioactivity-prediction-app.git
    cd ace-bioactivity-prediction-app

---

### 2. Install dependencies

    pip install -r requirements.txt

⚠️ RDKit may require Conda installation:

    conda install -c conda-forge rdkit

---

### 3. Train the model

    python main.py

---

### 4. Run the app

    streamlit run app.py

---

## 📁 Project Structure

    ├── app.py                  # Streamlit web app
    ├── main.py                 # Full ML pipeline
    ├── README.md
    ├── requirements.txt
    ├── .gitignore

    ├── data/                   # Processed datasets
    ├── models/                 # Trained model + descriptor list
    ├── plots/                  # Generated visualizations
    ├── results/                # Exported results

---

## 🧠 Technical Highlights

- RDKit for cheminformatics and molecular descriptors  
- MACCS fingerprints (167-bit representation)  
- Random Forest regression for bioactivity prediction  
- Feature selection using Variance Threshold  
- Reproducible and modular pipeline  

---

## 🙏 Acknowledgement

This project was inspired by:

https://github.com/dataprofessor/bioactivity-prediction-app

The original idea was extended with:

- RDKit-based descriptor generation (MACCS keys)  
- Improved SMILES preprocessing (salt removal)  
- Custom machine learning pipeline  
- Random Forest regression model  
- Streamlit-based interactive deployment  

---

## 👩‍💻 Author

Built as a complete end-to-end machine learning project  
for bioactivity prediction and cheminformatics applications.

---

## ⭐ Notes

- This project demonstrates a full ML pipeline, not just model training  
- Designed for portfolio and real-world application demonstration  
- Easily extendable to other biological targets  

