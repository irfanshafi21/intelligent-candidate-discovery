<div align="center">

# 🤖 Intelligent Candidate Discovery System

### **India Runs Hackathon 2026 — Track 01: Data & AI Challenge**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://intelligent-candidate-discovery-ey8bkkwuov7b5g7gmafk7x.streamlit.app)
[![Demo Video](https://img.shields.io/badge/🎥_Demo_Video-YouTube-FF0000?style=for-the-badge)](https://youtu.be/RBYdjJrNFyY)
[![Colab](https://img.shields.io/badge/📓_Open-Google_Colab-F9AB00?style=for-the-badge)](https://colab.research.google.com/drive/1p6pRJsb7AZ3lAqxFGtQ4rhTzefDwyCBL?usp=sharing)
[![GitHub](https://img.shields.io/badge/💻_GitHub-irfanshafi21-181717?style=for-the-badge)](https://github.com/irfanshafi21)

---

*AI-powered recruiter that shortlists the best candidates for **any job role** in seconds using NLP & Machine Learning — built for the official Redrob JSONL dataset with full honeypot filtering*

</div>

---

## 🚨 The Problem

> Indian companies receive **400+ job applications** per role.
> HR teams waste **8–12 hours** manually screening resumes.
> The best candidates get **missed** because traditional keyword filters
> don't understand context, skill depth, or true candidate fit.
> Datasets contain **fake / irrelevant profiles (honeypots)** that pollute results.

**The cost?** Wrong hires. Missed talent. Wasted time. Lost money.

---

## 💡 My Solution

An **AI-powered Candidate Discovery System** that:

| Feature | Description |
|---------|-------------|
| 🎯 Smart Matching | Understands job descriptions using NLP — not just keywords |
| 🚫 Honeypot Filter | Automatically detects and removes fake/irrelevant profiles |
| ⚡ Instant Results | Ranks 1000+ candidates in seconds |
| 🔄 Any Job Role | Works for ML Engineer, Frontend Dev, Data Analyst — any role |
| 📂 Unified Upload | One drop zone — accepts JSONL, JSON, and CSV together |
| 📊 Visual Analytics | Bar charts + scatter plots for instant insights |
| ⬇️ Export Ready | Full CSV, Submission CSV (Top 100), and PDF report |

---

## 🧠 How It Works

### Mode 1 — Full JSONL Pipeline (Official Redrob Dataset)

```
📋 Job Description
       ↓
🚫 Honeypot Filter   ←── removes fake/non-ML profiles from dataset
       ↓
🔤 TF-IDF Vectorizer ←── converts text to mathematical vectors
       ↓
📐 Cosine Similarity ←── measures skill-to-JD fit
       ↓
🧬 Multi-Signal Score ←── 7 signals combined (see below)
       ↓
🏆 Ranked Shortlist  ←── Top N cards + charts + CSV + PDF
```

### Mode 2 — CSV Pipeline (Custom / Multi-file Upload)

```
📋 Job Description
       ↓
🔤 TF-IDF Vectorizer ←── vectorise JD + all candidate skill texts
       ↓
📐 Cosine Similarity ←── skill match score per candidate
       ↓
⚖️  Weighted Score   ←── Skill % + Experience % + Activity % (user-tunable)
       ↓
🏆 Ranked Shortlist  ←── Top N cards + charts + CSV + PDF
```

---

## 🔬 Scoring Model

### JSONL Mode — 7-Signal Composite Score

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| 🎯 Skill–JD Cosine Similarity | 25% | How well candidate skills match the job description |
| 🧬 ML Skill Depth | 25% | Count of ML skills × proficiency level × endorsements |
| 🕐 ML Experience (months) | 15% | Actual months worked in ML/AI roles from career history |
| 📅 Total Experience | 10% | Overall years of experience (normalised to 10 yrs = 100) |
| ⚡ Platform Activity | 15% | GitHub score, profile views, interview rate, recruiter saves, verified email |
| 🎓 Education | 5% | Degree level × institution tier × AI-field bonus |
| 📝 Skill Assessments | 5% | Average score across Redrob skill assessment tests |

```
Final Score = (Cosine × 0.25) + (ML Depth × 0.25) + (ML Exp × 0.15)
            + (Exp Score × 0.10) + (Activity × 0.15) + (Education × 0.05)
            + (Assessments × 0.05)
```

### CSV Mode — Tunable 3-Signal Score

```
Final Score = (Skill Match × skill_weight%) + (Experience × exp_weight%) + (Activity × act_weight%)
```

> Weights are fully adjustable via sliders in the UI — default: Skill 70% · Exp 20% · Activity 10%

---

## 🚫 Honeypot Detection

The official dataset contains deliberately planted **fake or irrelevant profiles** designed to test whether the system can distinguish signal from noise. The `is_honeypot()` function removes a profile if:

- The candidate's current title matches non-ML roles (marketing, HR, accountant, civil engineer, etc.) **AND** they have fewer than 3 ML skills **AND** their career history contains fewer than 2 ML keywords, **OR**
- The candidate claims 12+ ML skills but has **zero ML career history** (skill-stuffing honeypot)

Filtered count is shown in the dashboard metrics tile as **"🚫 Honeypots Removed"**.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| ✅ Candidates ranked | 1000+ (JSONL) / 50+ (CSV) |
| ⚡ Time to rank | Under 5 seconds |
| 🎯 Top match score | 90+ out of 100 |
| 🚫 Honeypots filtered | Automatic — zero manual review |
| 🔄 Roles supported | Any job role |
| 📂 Files supported | JSONL · JSON · CSV (single upload slot) |
| 💾 Outputs | Full CSV · Submission CSV (Top 100) · PDF Report |

---

## ✨ Key Features

- ✅ **JSONL full pipeline** — processes the official Redrob structured dataset end-to-end
- ✅ **Honeypot detection** — auto-removes fake/irrelevant profiles before scoring
- ✅ **Unified file uploader** — one drop zone accepts `.jsonl`, `.json`, and `.csv` together (up to 1 GB per file)
- ✅ **Any job role** — paste any job description, get instant rankings
- ✅ **Adjustable weights** — HR can tune skill vs experience vs activity via sliders
- ✅ **Auto keyword detection** — highlights key skills extracted from the JD
- ✅ **Candidate cards** — gold/silver/bronze ranked with score bars and "why ranked" chips
- ✅ **Visual analytics** — top-10 bar chart + skill vs experience scatter plot
- ✅ **3 download formats** — Full ranked CSV · Submission CSV (Top 100) · PDF report
- ✅ **Live web app** — no installation needed, runs in browser

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Core language |
| 🤖 scikit-learn | TF-IDF Vectorizer + Cosine Similarity |
| 🐼 pandas & numpy | Data processing and scoring |
| 📊 matplotlib | Charts and PDF report generation |
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

## 📂 Supported File Formats

### JSONL / JSON — Official Redrob Dataset Format

The system natively consumes the structured Redrob dataset. Each line must be a valid JSON object with the following structure:

```json
{
  "candidate_id": "C001",
  "profile": {
    "anonymized_name": "Candidate_1",
    "current_title": "ML Engineer",
    "years_of_experience": 5,
    "headline": "...",
    "summary": "..."
  },
  "skills": [
    { "name": "Python", "proficiency": "advanced", "endorsements": 12 }
  ],
  "career_history": [
    { "title": "ML Engineer", "description": "...", "duration_months": 24 }
  ],
  "education": [
    { "degree": "M.Tech", "field_of_study": "machine learning", "tier": "tier_1" }
  ],
  "redrob_signals": {
    "github_activity_score": 82,
    "profile_completeness_score": 90,
    "open_to_work_flag": true,
    "profile_views_received_30d": 34,
    "interview_completion_rate": 0.85,
    "recruiter_response_rate": 0.78,
    "verified_email": true,
    "saved_by_recruiters_30d": 6,
    "skill_assessment_scores": { "Python": 88, "ML": 91 }
  }
}
```

### CSV — Custom Candidate Format

Upload your own candidates with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `name` | Candidate name | Rahul Sharma |
| `skills` | All skills space-separated | Python machine learning TensorFlow |
| `experience_years` | Years of experience | 4 |
| `job_title` | Current or past role | ML Engineer |
| `activity_score` | Activity score 0–100 | 85 |
| `education` | Highest degree | M.Tech CS |

> 💡 You can upload **multiple CSV files and/or a JSONL file in the same drop zone** — JSONL takes priority when present; otherwise all CSVs are merged automatically.

---

## 📁 Repository Structure

```
📦 intelligent-candidate-discovery
 ┣ 📄 app.py                  → Streamlit web app (main file)
 ┣ 📓 colab_code.py           → Google Colab notebook
 ┣ 📊 candidates.csv          → Sample candidate dataset (50 profiles)
 ┣ 📊 candidates_file1.csv    → Sample dataset 1 (for multi-upload demo)
 ┣ 📊 candidates_file2.csv    → Sample dataset 2 (for multi-upload demo)
 ┣ 📊 candidates.jsonl        → Official Redrob JSONL dataset (place here)
 ┣ 📈 ranked_output.csv       → Sample ranked output
 ┣ 📄 submission.csv          → Submission-format output (Top 100, pre-generated)
 ┣ 🖼️ results_chart.png       → Visualization chart
 ┗ 📋 requirements.txt        → Python dependencies
```

---

## 🎯 Why This Wins

> 💼 **Real problem** — India's hiring crisis affects millions of companies
>
> ⚡ **Real solution** — Works live, ranks 1000+ candidates in seconds
>
> 🧠 **Real AI** — 7-signal composite scoring with TF-IDF + cosine similarity
>
> 🚫 **Honeypot aware** — automatically filters fake profiles the dataset plants
>
> 📦 **Production ready** — Live app, clean UI, CSV + PDF exports
>
> 🇮🇳 **Built for India** — Designed around Redrob's platform signals and Indian HR workflows

---

## 👤 Author

<div align="center">

**Irfan Shafi**

[![GitHub](https://img.shields.io/badge/GitHub-irfanshafi21-181717?style=flat-square&logo=github)](https://github.com/irfanshafi21)

*India Runs Hackathon 2026 — Track 01: Data & AI Challenge*

</div>
