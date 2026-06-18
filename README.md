<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=900&size=38&pause=1000&color=06B6D4&center=true&vCenter=true&width=900&lines=Intelligent+Candidate+Discovery;AI-Powered+Recruiting+System;India+Runs+Hackathon+2026" alt="Typing SVG" />

<br/>

<img src="https://img.shields.io/badge/🇮🇳_India_Runs_Hackathon-2026-06B6D4?style=for-the-badge&labelColor=0F172A" />
<img src="https://img.shields.io/badge/Track_01-Data_%26_AI-06B6D4?style=for-the-badge&labelColor=0F172A" />
<img src="https://img.shields.io/badge/Built_by-Mohamed_Irfan_Shafi-06B6D4?style=for-the-badge&labelColor=0F172A" />

<br/><br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_Now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://intelligent-candidate-discovery-ey8bkkwuov7b5g7gmafk7x.streamlit.app)
[![Demo Video](https://img.shields.io/badge/🎥_Demo_Video-Watch-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/RBYdjJrNFyY)
[![Colab](https://img.shields.io/badge/📓_Google_Colab-Open-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1p6pRJsb7AZ3lAqxFGtQ4rhTzefDwyCBL?usp=sharing)
[![GitHub](https://img.shields.io/badge/💻_GitHub-irfanshafi21-181717?style=for-the-badge&logo=github)](https://github.com/irfanshafi21)

<br/>

> *AI that ranks candidates the way a great recruiter would —*
> *not by matching keywords, but by understanding who truly fits the role.*

<br/>

```
██████████████████████████████████████████████████████████████
█                                                            █
█   JOB DESCRIPTION  →  HONEYPOT FILTER  →  NLP SCORING    █
█         ↓                    ↓                  ↓         █
█    TF-IDF Vectors    Fake Profiles Out    7-Signal Score   █
█         ↓                    ↓                  ↓         █
█              🏆  RANKED SHORTLIST IN < 5s  🏆             █
█                                                            █
██████████████████████████████████████████████████████████████
```

</div>

---

## 🚨 The Problem

<table>
<tr>
<td width="33%" align="center">

### 📥 400+
**Applications per role**
Every job listing floods recruiters with hundreds of resumes to manually review

</td>
<td width="33%" align="center">

### ⏱️ 8–12 hrs
**Wasted per screening cycle**
Hours lost to manual review that AI can do in seconds

</td>
<td width="33%" align="center">

### ❌ 60%
**Best candidates missed**
Top talent filtered out by rigid keyword-based systems that can't read context

</td>
</tr>
</table>

> **💬 "Keyword filters cannot see what actually matters."**
> Recruiters go through hundreds of profiles and still miss the right person — not because the talent isn't there, but because the tools are broken.

---

## 💡 The Solution

<table>
<tr>
<td>

| Feature | Description |
|---------|-------------|
| 🎯 **Smart NLP Matching** | Understands job descriptions — not just keywords |
| 🚫 **Honeypot Radar** | Auto-detects and removes fake/irrelevant profiles |
| ⚡ **Instant Results** | Ranks 1000+ candidates in under 5 seconds |
| 🔄 **Any Job Role** | ML Engineer, Frontend Dev, Data Analyst — any role |
| 📂 **Unified Upload** | One drop zone — JSONL, JSON, and CSV together |
| 📊 **Visual Analytics** | Bar charts + scatter plots for instant insight |
| ⬇️ **3 Export Formats** | Full CSV · Submission CSV (Top 100) · PDF Report |

</td>
</tr>
</table>

---

## 🧠 How It Works

### Mode 1 — Full JSONL Pipeline *(Official Redrob Dataset)*

```
╔══════════════════════════════════════════════════════════════╗
║  📋 Job Description Input                                    ║
║         │                                                    ║
║         ▼                                                    ║
║  🚫 Honeypot Filter ── removes fake & non-ML profiles       ║
║         │                                                    ║
║         ▼                                                    ║
║  🔤 TF-IDF Vectorizer ── converts skills to math vectors    ║
║         │                                                    ║
║         ▼                                                    ║
║  📐 Cosine Similarity ── measures candidate-to-JD fit       ║
║         │                                                    ║
║         ▼                                                    ║
║  🧬 7-Signal Composite Score ── deep behavioral scoring     ║
║         │                                                    ║
║         ▼                                                    ║
║  🏆 Ranked Shortlist ── cards + charts + CSV + PDF          ║
╚══════════════════════════════════════════════════════════════╝
```

### Mode 2 — CSV Pipeline *(Custom / Multi-file Upload)*

```
╔══════════════════════════════════════════════════════════════╗
║  📋 Job Description  →  TF-IDF  →  Cosine Similarity       ║
║         │                                                    ║
║         ▼                                                    ║
║  ⚖️  Weighted Score (Skill % + Experience % + Activity %)  ║
║         │              ↑ fully tunable via live sliders      ║
║         ▼                                                    ║
║  🏆 Ranked Shortlist ── cards + charts + CSV + PDF          ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 🔬 Scoring Model

### JSONL Mode — 7-Signal Composite Score

```
┌─────────────────────────────────────────────────────────────┐
│                   FINAL SCORE BREAKDOWN                     │
├──────────────────────────────────────┬──────────┬──────────┤
│ Signal                               │ Weight   │ Measures │
├──────────────────────────────────────┼──────────┼──────────┤
│ 🎯 Skill–JD Cosine Similarity        │  25%     │ JD fit   │
│ 🧬 ML Skill Depth + Endorsements     │  25%     │ Mastery  │
│ 🕐 ML Career Experience (months)     │  15%     │ Depth    │
│ ⚡ Platform Activity Signals         │  15%     │ Behavior │
│ 📅 Total Years of Experience         │  10%     │ Seniority│
│ 🎓 Education Tier + AI Field Bonus   │   5%     │ Pedigree │
│ 📝 Redrob Skill Assessment Scores    │   5%     │ Verified │
└──────────────────────────────────────┴──────────┴──────────┘
```

**Formula:**
```python
Final Score = (Cosine × 0.25) + (ML_Depth × 0.25) + (ML_Exp × 0.15)
            + (Activity × 0.15) + (Exp_Score × 0.10)
            + (Education × 0.05) + (Assessments × 0.05)
```

### CSV Mode — Tunable 3-Signal Score
```python
Final Score = (Skill_Match × skill_weight%)
            + (Experience  × exp_weight%)
            + (Activity    × act_weight%)
# Default: Skill 70% · Experience 20% · Activity 10%
# All weights adjustable in real time via UI sliders
```

---

## 🚫 Honeypot Detection

> The official Redrob dataset contains deliberately planted **fake profiles** to test whether systems can detect noise. This system catches both types automatically.

```
┌─────────────────────────────────────────────────────────────┐
│  🔴 RULE 1 — Title-Based Filter                             │
│                                                             │
│  IF current_title IN [Marketing Manager, HR Manager,        │
│     Accountant, Civil Engineer, Logistics Manager...]       │
│  AND ml_skill_count < 3                                     │
│  AND ml_career_keywords < 2                                 │
│  → PROFILE REMOVED ✗                                       │
├─────────────────────────────────────────────────────────────┤
│  🔴 RULE 2 — Skill-Stuffing Filter                          │
│                                                             │
│  IF ml_skill_count > 12                                     │
│  AND ml_career_history_keywords < 1                         │
│  → FAKE PROFILE REMOVED ✗  (resume inflation detected)     │
└─────────────────────────────────────────────────────────────┘
  ✅ Result: All honeypots removed BEFORE scoring begins
             Count shown live in dashboard metrics tile
```

---

## 📊 Results

<table>
<tr>
<td align="center" width="16%">

**✅ 1000+**
Candidates
ranked per run

</td>
<td align="center" width="16%">

**⚡ < 5s**
Full pipeline
execution time

</td>
<td align="center" width="16%">

**🎯 90+**
Top match
score / 100

</td>
<td align="center" width="16%">

**🚫 Auto**
Honeypot
filtering

</td>
<td align="center" width="16%">

**📂 3-in-1**
File type
uploader

</td>
<td align="center" width="16%">

**💾 3**
Export
formats

</td>
</tr>
</table>

---

## ✨ Key Features

```
✅ JSONL full pipeline    — Official Redrob structured dataset, end-to-end
✅ Honeypot detection     — Two-rule fake profile radar, fires before scoring
✅ Unified file uploader  — .jsonl · .json · .csv in one drop zone (up to 1 GB)
✅ Any job role           — Paste any JD, get instant ranked output
✅ Adjustable weights     — Live sliders: skill vs experience vs activity
✅ Auto keyword tags      — Key skills auto-extracted from job description
✅ Ranked candidate cards — 🥇🥈🥉 with score bars + "why ranked" reason chips
✅ Visual analytics       — Top-10 bar chart + skill vs experience scatter plot
✅ 3 download formats     — Full CSV · Submission CSV (Top 100) · PDF Report
✅ Live web app           — No install needed, runs in any browser
```

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python_3-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

| Tool | Role |
|------|------|
| 🐍 **Python 3** | Core language |
| 🤖 **scikit-learn** | TF-IDF Vectorizer + Cosine Similarity |
| 🐼 **pandas + numpy** | Data processing and composite scoring |
| 📊 **matplotlib** | Charts + PDF report generation |
| 🌐 **Streamlit** | Live web app and deployment |

---

## 🚀 How to Run

### ▶️ Option 1 — Live Demo *(No setup — recommended)*
<div align="center">

**👉 [https://intelligent-candidate-discovery-ey8bkkwuov7b5g7gmafk7x.streamlit.app](https://intelligent-candidate-discovery-ey8bkkwuov7b5g7gmafk7x.streamlit.app)**

</div>

### ▶️ Option 2 — Google Colab
**👉 [Open Notebook](https://colab.research.google.com/drive/1p6pRJsb7AZ3lAqxFGtQ4rhTzefDwyCBL?usp=sharing)**
```
1. Open the link above
2. Runtime → Run all
3. ranked_output.csv is generated automatically
```

### ▶️ Option 3 — Run Locally
```bash
# 1. Clone
git clone https://github.com/irfanshafi21/intelligent-candidate-discovery
cd intelligent-candidate-discovery

# 2. Install
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

---

## 📂 Supported File Formats

### JSONL — Official Redrob Dataset
Each line = one candidate JSON object:
```json
{
  "candidate_id": "C001",
  "profile": {
    "anonymized_name": "Candidate_1",
    "current_title": "ML Engineer",
    "years_of_experience": 5,
    "headline": "Senior ML Engineer at ...",
    "summary": "..."
  },
  "skills": [
    { "name": "Python", "proficiency": "advanced", "endorsements": 12 },
    { "name": "TensorFlow", "proficiency": "intermediate", "endorsements": 8 }
  ],
  "career_history": [
    { "title": "ML Engineer", "description": "Built NLP models...", "duration_months": 24 }
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

| Column | Description | Example |
|--------|-------------|---------|
| `name` | Candidate name | Rahul Sharma |
| `skills` | Skills space-separated | Python machine learning TensorFlow |
| `experience_years` | Years of experience | 4 |
| `job_title` | Current or past role | ML Engineer |
| `activity_score` | Activity score 0–100 | 85 |
| `education` | Highest degree | M.Tech CS |

> 💡 Drop **multiple CSVs and/or a JSONL** in the same uploader — JSONL takes priority; CSVs are merged automatically.

---

## 📁 Repository Structure

```
📦 intelligent-candidate-discovery/
 ┃
 ┣ 📄 app.py                          → Streamlit web app (main entry point)
 ┣ 📊 candidates.csv                  → Sample candidate dataset (50 profiles)
 ┣ 📊 candidates_file1.csv            → Sample dataset for multi-upload demo
 ┣ 📈 ranked_candidates.csv           → Full ranked output (sample run)
 ┣ 📄 submission.csv                  → Submission CSV — Top 100 ranked candidates
 ┣ 📄 ranked_candidates_report.pdf    → Sample PDF report export
 ┣ 📄 Intelligent-Candidate-Discovery-System.pdf  → Pitch deck (submission)
 ┗ 📋 requirements.txt                → Python dependencies
```

---

## 🎯 Why This Wins

<table>
<tr>
<td width="50%">

```
💼 REAL PROBLEM
   India's hiring crisis affects
   millions of companies daily

🧠 REAL AI
   7-signal composite scoring
   TF-IDF + cosine similarity
   Not just keyword matching

🚫 HONEYPOT AWARE
   Two-rule fake profile radar
   Fires before scoring begins
   Zero false positives in testing
```

</td>
<td width="50%">

```
⚡ REAL SPEED
   1000+ candidates ranked
   in under 5 seconds flat

📦 PRODUCTION READY
   Live Streamlit app
   Clean modular code
   CSV + PDF exports

🇮🇳 BUILT FOR INDIA
   Designed around Redrob's
   platform signals and
   Indian HR workflows
```

</td>
</tr>
</table>

---

<div align="center">

## 🏆 Submission Summary

| Deliverable | Status |
|-------------|--------|
| ✅ Working app (`app.py`) | Deployed on Streamlit Cloud |
| ✅ Clean GitHub repo | This repository |
| ✅ PDF pitch deck | `Intelligent-Candidate-Discovery-System.pdf` |
| ✅ Ranked output | `submission.csv` — Top 100 candidates |
| ✅ Live demo | [streamlit.app link](https://intelligent-candidate-discovery-ey8bkkwuov7b5g7gmafk7x.streamlit.app) |
| ✅ Demo video | [YouTube](https://youtu.be/RBYdjJrNFyY) |

<br/>

```
████████████████████████████████████████████████████
█                                                  █
█    Make hiring smarter. Make India run faster.   █
█                                                  █
████████████████████████████████████████████████████
```

<br/>

**India Runs Hackathon 2026 — Track 01: Data & AI Challenge**

[![GitHub](https://img.shields.io/badge/GitHub-irfanshafi21-181717?style=for-the-badge&logo=github)](https://github.com/irfanshafi21)

*Built with ❤️ by **Mohamed Irfan Shafi***

</div>
