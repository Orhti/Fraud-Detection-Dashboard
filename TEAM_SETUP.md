# 🛡️ Fraud Detection Dashboard — Team Setup Guide

Hi! Follow these steps to run the project on your own computer. No prior setup needed — this guide covers everything from scratch.

**Total time: ~5 minutes**

---

## ✅ What You Need First

| Requirement | Check | Download Link |
|-------------|-------|---------------|
| Python 3.10+ | Open terminal → type `python --version` | https://www.python.org/downloads/ |
| Git | Type `git --version` | https://git-scm.com/download/win |

> **Note:** During Python install, tick ✅ **"Add Python to PATH"** before clicking Install.

---

## Step 1 — Clone the Repository

Open **Command Prompt** or **PowerShell** and run:

```bash
git clone https://github.com/Orhti/Fraud-Detection-Dashboard.git
cd Fraud-Detection-Dashboard
```

This downloads all project files to your computer.

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required Python libraries (pandas, scikit-learn, streamlit, etc.).
It takes about 1–2 minutes on first run.

---

## Step 3 — Run the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Your browser will open automatically at **http://localhost:8501**

> If your browser doesn't open, copy `http://localhost:8501` and paste it manually.

---

## 🖥️ What You'll See

The dashboard has **4 pages** (navigate using the left sidebar):

| Page | What It Shows |
|------|---------------|
| 🏠 **Project Overview** | KPI cards, dataset stats, project pipeline |
| 📊 **EDA & Patterns** | Interactive charts — fraud patterns by type, location, device |
| 🤖 **Model Results** | All 6 models compared by F1 Score and AUC-ROC |
| 🚨 **Live Predictor** | Enter any transaction details → get instant fraud prediction |

---

## 📓 Optional: Run the Full Notebook

If you want to see the complete ML pipeline (data prep → model training → evaluation):

1. Install Jupyter:
   ```bash
   pip install notebook
   ```
2. Launch it:
   ```bash
   jupyter notebook
   ```
3. Open `Fraud_Detection_Notebook.ipynb` in your browser
4. Click **Kernel → Restart & Run All**

The notebook trains all 6 models and saves results — takes about 3–5 minutes.

---

## ❓ Common Issues

| Problem | Fix |
|---------|-----|
| `pip` not recognised | Use `python -m pip install -r requirements.txt` instead |
| `streamlit` not found | Run `pip install streamlit` then try again |
| Browser doesn't open | Go to `http://localhost:8501` manually |
| Port already in use | Run `streamlit run dashboard/streamlit_app.py --server.port 8502` |
| Module not found error | Make sure you ran `pip install -r requirements.txt` first |

---

## 🔗 Links

- **Live Dashboard:** https://fraud-detection-dashboard-ecaysbzh92x.streamlit.app/
- **GitHub Repo:** https://github.com/Orhti/Fraud-Detection-Dashboard

---

*Built by Rohit — Data Science Project, 2026*
