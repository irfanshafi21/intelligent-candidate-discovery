<div align="center">

# 🤖 Intelligent Candidate Discovery System

### **India Runs Hackathon 2026 — Track 01: Data & AI Challenge**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://intelligent-candidate-discovery-ey8bkkwuov7b5g7gmafk7x.streamlit.app)
[![Demo Video](https://img.shields.io/badge/🎥_Demo_Video-YouTube-FF0000?style=for-the-badge)](https://youtu.be/3vAlT1Oe7FM)
[![Colab](https://img.shields.io/badge/📓_Open-Google_Colab-F9AB00?style=for-the-badge)](https://colab.research.google.com/drive/1p6pRJsb7AZ3lAqxFGtQ4rhTzefDwyCBL?usp=sharing)
[![GitHub](https://img.shields.io/badge/💻_GitHub-irfanshafi21-181717?style=for-the-badge)](https://github.com/irfanshafi21)

---

*AI-powered recruiter that shortlists the best candidates for **any job role** in seconds using NLP & Machine Learning*

</div>

---

## 🚨 The Problem

> Indian companies receive **400+ job applications** per role.
> HR teams waste **8–12 hours** manually screening resumes.
> The best candidates get **missed** because traditional keyword filters
> don't understand context, skill depth, or true candidate fit.

**The cost?** Wrong hires. Missed talent. Wasted time. Lost money.

---

## 💡 My Solution

An **AI-powered Candidate Discovery System** that:

| Feature | Description |
|---------|-------------|
| 🎯 Smart Matching | Understands job descriptions using NLP — not just keywords |
| ⚡ Instant Results | Ranks 50+ candidates in under 3 seconds |
| 🔄 Any Job Role | Works for ML Engineer, Frontend Dev, Data Analyst — any role |
| 📂 Flexible Input | Upload multiple CSV files or use built-in dataset |
| 📊 Visual Analytics | Bar charts + scatter plots for instant insights |
| ⬇️ Export Ready | Download full ranked shortlist as CSV |

---

## 🧠 How It Works

```
📋 Job Description
       ↓
🔤 TF-IDF Vectorizer  ←── converts text to mathematical vectors
       ↓
📐 Cosine Similarity  ←── measures how close candidate skills are to JD
       ↓
⚖️  Weighted Score    ←── Skill 70% + Experience 20% + Activity 10%
       ↓
🏆 Ranked Shortlist   ←── Top N candidates, download as CSV
```

### Scoring Formula
```
Final Score = (Skill Match × 0.70) + (Experience × 0.20) + (Activity × 0.10)
```

| Signal | Weight | Why |
|--------|--------|-----|
| 🎯 Skill Match | 70% | Skills are the most critical hiring signal |
| 📅 Experience | 20% | Seniority matters but isn't everything |
| ⚡ Activity Score | 10% | Engagement and activity as a bonus signal |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| ✅ Candidates ranked | 50+ |
| ⚡ Time to rank | Under 3 seconds |
| 🎯 Top match score | 90+ out of 100 |
| 🔄 Roles supported | Any job role |
| 📂 Files supported | Multiple CSV upload & merge |
| 💾 Output | Downloadable ranked CSV |

---

## ✨ Key Features

- ✅ **Any job role** — paste any job description, get instant rankings
- ✅ **Multiple CSV upload** — upload and merge multiple candidate files
- ✅ **Adjustable weights** — HR can tune skill vs experience vs activity
- ✅ **Auto keyword detection** — highlights key skills from JD
- ✅ **Candidate cards** — gold/silver/bronze ranked with score bars
- ✅ **Visual analytics** — bar chart + skill vs experience scatter plot
- ✅ **Download results** — export full ranked shortlist as CSV
- ✅ **Live web app** — no installation needed, runs in browser

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Core language |
| 🤖 scikit-learn | TF-IDF Vectorizer + Cosine Similarity |
| 🐼 pandas & numpy | Data processing and scoring |
| 📊 matplotlib | Charts and visualizations |
| 🌐 Streamlit | Live web app deployment |
| 📓 Google Colab | Notebook development and demo |

---

## 🚀 How to Run

### Option 1 — Live Demo *(Recommended — no setup needed)*
👉 **[Click here to try the app](https://intelligent-candidate-discovery-ey8bkkwuov7b5g7gmafk7x.streamlit.app)**

### Option 2 — Google Colab
👉 **[Open in Google Colab](https://colab.research.google.com/drive/1p6pRJsb7AZ3lAqxFGtQ4rhTzefDwyCBL?usp=sharing)**
1. Click the link above
2. Run all cells in order
3. `ranked_output.csv` will be generated automatically

### Option 3 — Run Locally
```bash
# Clone the repo
git clone https://github.com/irfanshafi21/intelligent-candidate-discovery

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📋 Candidate CSV Format

Upload your own candidates with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `name` | Candidate name | Rahul Sharma |
| `skills` | All skills space separated | Python machine learning TensorFlow |
| `experience_years` | Years of experience | 4 |
| `job_title` | Current or past role | ML Engineer |
| `activity_score` | Activity score 0–100 | 85 |
| `education` | Highest degree | M.Tech CS |

> 💡 You can upload **multiple CSV files** — the system merges them automatically!

---

## 📁 Repository Structure

```
📦 intelligent-candidate-discovery
 ┣ 📄 app.py                  → Streamlit web app (main file)
 ┣ 📓 colab_code.py           → Google Colab notebook
 ┣ 📊 candidates.csv          → Sample candidate dataset (50 profiles)
 ┣ 📊 candidates_file1.csv    → Sample dataset 1 (for multi-upload demo)
 ┣ 📊 candidates_file2.csv    → Sample dataset 2 (for multi-upload demo)
 ┣ 📈 ranked_output.csv       → Sample ranked output
 ┣ 🖼️ results_chart.png       → Visualization chart
 ┗ 📋 requirements.txt        → Python dependencies
```

---

## 🎯 Why This Wins

> 💼 **Real problem** — India's hiring crisis affects millions of companies
>
> ⚡ **Real solution** — Works live, ranks instantly, any role
>
> 🧠 **Real AI** — NLP + cosine similarity, not just keyword search
>
> 📦 **Production ready** — Live app, clean UI, downloadable output
>
> 🇮🇳 **Built for India** — Designed around Indian HR workflows and job portals

---

## 👤 Author

<div align="center">

**Irfan Shafi**

[![GitHub](https://img.shields.io/badge/GitHub-irfanshafi21-181717?style=flat-square&logo=github)](https://github.com/irfanshafi21)

*India Runs Hackathon 2026 — Track 01: Data & AI Challenge*

</div>
