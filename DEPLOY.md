# 🚀 GitHub + Streamlit Cloud Deployment Guide

Follow these steps exactly. Total time: ~10 minutes.

---

## PART 1 — Push to GitHub

### Step 1: Check Git is installed
Open a terminal (Command Prompt or PowerShell) and run:
```
git --version
```
If you see `git version 2.x.x` → you're good.  
If you get "not recognized" → download Git from https://git-scm.com/download/win and install it.

---

### Step 2: Open terminal IN the project folder
```
cd "C:\Users\user\OneDrive\Documents\Data Science Project\Data Science Project\Project_1_Fraud_Detection"
```

---

### Step 3: Initialize Git
```
git init
git add .
git status
```
The `git status` output should show your files in green (staged).  
The `models/` folder and `.png` files should NOT appear (they are gitignored).

---

### Step 4: Make your first commit
```
git commit -m "Initial commit: Fraud Detection Dashboard"
```

---

### Step 5: Create a repo on GitHub
1. Go to https://github.com/new
2. Repository name: `fraud-detection-dashboard` (or any name you like)
3. Set to **Public** (required for free Streamlit Cloud)
4. Do NOT tick "Add README" (we already have one)
5. Click **Create repository**

---

### Step 6: Connect and push
GitHub will show you commands. Use these (replace YOUR_USERNAME):
```
git remote add origin https://github.com/YOUR_USERNAME/fraud-detection-dashboard.git
git branch -M main
git push -u origin main
```

When prompted, enter your GitHub username and password.  
> **Note:** GitHub now requires a Personal Access Token instead of your password.  
> If the push fails, go to: GitHub → Settings → Developer settings → Personal access tokens → Generate new token (classic) → tick "repo" → copy the token → use it as your password.

✅ Refresh your GitHub page — all files should now be visible.

---

## PART 2 — Deploy to Streamlit Cloud

### Step 7: Sign in to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click **Sign in with GitHub**
3. Authorize Streamlit

---

### Step 8: Deploy the app
1. Click **New app**
2. Fill in:
   - **Repository:** `YOUR_USERNAME/fraud-detection-dashboard`
   - **Branch:** `main`
   - **Main file path:** `dashboard/streamlit_app.py`
3. Click **Deploy!**

Streamlit will install requirements and launch the app (takes 2-3 minutes).

---

### Step 9: Get your shareable link
Once deployed, you'll get a URL like:
```
https://YOUR_USERNAME-fraud-detection-dashboard-dashboard-streamlit-app-xxxxx.streamlit.app
```

Share this link with anyone — no installation needed, works in any browser.

---

## PART 3 — Update the README with your live link

Open `README.md` and replace this line:
```
> **[▶ Open Dashboard on Streamlit Cloud](#)** ← *(link added after deployment)*
```
With your actual URL:
```
> **[▶ Open Dashboard on Streamlit Cloud](https://your-actual-url.streamlit.app)**
```

Then push the update:
```
git add README.md
git commit -m "Add live Streamlit Cloud link"
git push
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `git` not recognized | Install from https://git-scm.com |
| Push asks for password → fails | Use Personal Access Token (see Step 6 note) |
| Streamlit Cloud shows "requirements not found" | Make sure `requirements.txt` is in the root folder |
| App crashes on Streamlit Cloud | Check the logs — usually a missing package; add it to `requirements.txt` |
| Data file not found | Make sure `data/synthetic_fraud_dataset1.csv` was committed (not gitignored) |
