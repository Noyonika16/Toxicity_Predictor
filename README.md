# 🧪 Drug Toxicity Predictor Model

An end-to-end machine learning project that predicts **chemical toxicity from SMILES strings** using advanced ensemble models and explainable AI.

---

## Project Overview

This project predicts the toxicity level of chemical compounds (**HIGH / MEDIUM / LOW**) using molecular structure data.

It combines:

* 🧠 Machine Learning (XGBoost + LightGBM + CatBoost)
* 🧬 Chemical feature engineering (RDKit)
* 🔍 Explainable AI (SHAP)
* 🌐 Interactive UI (Streamlit)

Users can input any molecule (SMILES) and get:

* Toxicity prediction
* Confidence score
* Molecular visualization (2D + 3D)
* Key features affecting toxicity
* Model explanations

---

## ✨ Key Features

* 🔬 Predict toxicity from SMILES input
* 🧠 Ensemble model (XGBoost + LightGBM + CatBoost)
* 🧪 RDKit-based feature engineering (descriptors + fingerprints)
* 📊 Feature selection (SelectKBest)
* ⚖️ Class imbalance handling
* 🔍 SHAP-based explainability
* 📈 Confidence estimation (entropy)
* 🎨 Interactive Streamlit UI
* 🧬 2D + 3D molecule visualization

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Feature Engineering

* Molecular descriptors (MolWt, LogP, TPSA, etc.)
* Morgan Fingerprints (256-bit)
* RDKit chemical properties

### 2️⃣ Preprocessing

* Missing value handling
* Feature scaling (StandardScaler)
* Feature selection (SelectKBest)

### 3️⃣ Models

* XGBoost
* LightGBM
* CatBoost
* Ensemble averaging

### 4️⃣ Explainability

* SHAP values for feature-level explanations
* Human-readable insights

---

## 📊 Output Example

* Toxicity Level: HIGH / MEDIUM / LOW
* Confidence Score
* Probability Distribution
* Molecular Properties
* SHAP Explanation
* Key contributing features

---

## 📁 Project Structure

```
toxicity-predictor/
│
├── app/                # Streamlit app
│   └── app.py
│
├── assets/             # UI screenshots / demo
│
├── data/               # Dataset
│
├── models/             # Trained model
│   └── tox.pkl
│
├── notebook/           # Model training
│   └── model.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run This Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/toxicity-predictor.git
cd toxicity-predictor
```

---

### 2️⃣ Create environment (recommended)

```bash
conda create -n tox python=3.10
conda activate tox
```

OR using venv:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the app

```bash
cd app
streamlit run app.py
```

---

## 🧪 Technologies Used

* Python
* RDKit
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* SHAP
* Streamlit
* Matplotlib

---

## 🌍 Real-World Applications

* Drug discovery & screening
* Toxicity prediction before lab testing
* Chemical safety analysis
* Reducing experimental cost & time

---

## 🧠 Key Insight

This project demonstrates how **AI + Chemistry + Explainability** can be combined to build practical tools for real-world scientific problems.

---

## 👨‍💻 Author

**Noyonika Mukherjee**

📧 Email: [noyonikacarmel@gmail.com](mailto:noyonikacarmel@gmail.com)
🔗 LinkedIn: https://www.linkedin.com/in/noyonika16-mukherjee

---

## ⚠️ Disclaimer

This model provides predictive insights based on data patterns.
Actual toxicity depends on:

* Dose
* Exposure route
* Biological conditions

---

⭐ If you like this project, consider giving it a star!
