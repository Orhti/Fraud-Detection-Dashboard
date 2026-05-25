# 🛡️ Financial Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-008000)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end **Machine Learning system** that detects fraudulent financial transactions using 6 ML algorithms, with an interactive Streamlit dashboard for exploration and live prediction.

---

## 📌 Project Overview

Financial fraud costs billions annually. This project builds a complete fraud detection pipeline:

- **Dataset:** 50,000 synthetic transactions with 14 features (transaction type, amount, device, location, card details, account history)
- **Challenge:** Severe class imbalance — only ~32% fraud cases → solved with **SMOTE**
- **Approach:** Train 6 models, compare on F1/AUC-ROC, deploy the best as an interactive dashboard

---

## 🚀 Live Demo

> **[▶ Open Dashboard on Streamlit Cloud](#)** ← *(link added after deployment)*

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Project Overview** | KPI cards, pipeline diagram, model summary |
| 📊 **EDA & Patterns** | Interactive charts with filters, correlation heatmap, fraud patterns |
| 🤖 **Model Results** | Ranked comparison table, radar chart, metric breakdown |
| 🚨 **Live Predictor** | Enter any transaction → instant fraud/legitimate prediction |

---

## 🤖 Models Trained

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Supervised | Baseline linear model |
| Decision Tree | Supervised | Interpretable rule-based |
| **Random Forest** | Supervised | Best single model |
| XGBoost | Supervised | Gradient boosting |
| Isolation Forest | Unsupervised | Anomaly detection |
| **Voting Ensemble** | Combined | LR + RF + XGB → best overall |

---

## 📁 Repository Structure

```
├── data/
│   ├── synthetic_fraud_dataset1.csv      ← Primary dataset (50K rows)
│   ├── credit_card_fraud_dataset.csv     ← Secondary dataset
│   └── financial_fraud_detection_dataset.csv
├── dashboard/
│   ├── streamlit_app.py                  ← Streamlit dashboard (4 pages)
│   └── requirements.txt
├── Fraud_Detection_Notebook.ipynb        ← Full ML pipeline notebook
├── requirements.txt                      ← Root-level (for Streamlit Cloud)
└── README.md
```

---

## ⚙️ Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/Orhti/Fraud-Detection-Dashboard.git
cd Fraud-Detection-Dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the dashboard**
```bash
streamlit run dashboard/streamlit_app.py
```

Opens at **http://localhost:8501**

> **Note:** No need to run the notebook first — the dashboard automatically trains a model from the CSV on first launch (takes ~30 seconds).

---

## 📓 Running the Full Notebook

To train all 6 models and generate evaluation charts:

1. Open `Fraud_Detection_Notebook.ipynb` in Jupyter
2. Click **Kernel → Restart & Run All**
3. Results saved to `models/` and `model_comparison.csv`

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — visualisation
- **scikit-learn** — ML models & preprocessing
- **imbalanced-learn** — SMOTE for class balancing
- **XGBoost** — gradient boosting
- **Streamlit** — interactive dashboard

---

## 📈 Key Results

> *(Updated after running the notebook)*

| Model | F1 Score | AUC-ROC |
|-------|----------|---------|
| Voting Ensemble | — | — |
| Random Forest   | — | — |
| XGBoost         | — | — |

---

## 👤 Author

**Rohit** — Data Science Project, 2026

---

## 📄 License

This project is for educational purposes as part of a Data Science training program.
