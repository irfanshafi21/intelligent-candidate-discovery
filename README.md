# 🤖 Intelligent Candidate Discovery System
### India Runs Hackathon 2026 — Track 01: Data & AI Challenge

---

## 🎥 Demo Video
👉 [Click here to watch the demo](https://youtu.be/3vAlT1Oe7FM)

## 🚀 Live Demo
👉 [Click here to try the app](https://intelligent-candidate-discovery-ey8bkkwuov7b5g7gmafk7x.streamlit.app)

## 📓 Google Colab
👉 [Open in Google Colab](https://colab.research.google.com/drive/1p6pRJsb7AZ3lAqxFGtQ4rhTzefDwyCBL?usp=sharing)

## 💻 GitHub
👉 [View Repository](https://github.com/irfanshafi21)

---

## 📌 Problem Statement

Indian companies receive 400+ job applications per role.
HR teams spend 8–12 hours manually screening resumes and still
miss the best candidates. Traditional keyword filters fail to
understand context and true skill fit.

> **Goal:** Build an AI system that instantly shortlists the
> best candidates for any job role — saving time and improving
> hiring quality.

---

## 💡 Solution

An AI-powered recruiter that works for **any job role**:

- 📝 Paste or upload **any job description**
- 📂 Upload your own **candidate CSV** or use the default dataset
- 🤖 AI ranks all candidates by **skill match, experience & activity**
- 📊 View **charts and ranked table** instantly
- ⬇️ **Download results** as CSV

---

## 🧠 How It Works

```
Job Description → TF-IDF Vectorizer → Cosine Similarity → Weighted Score → Ranked Output
```

1. HR pastes or uploads any job description
2. TF-IDF vectorizer converts JD + candidate skills into vectors
3. Cosine similarity measures semantic skill fit between JD and each candidate
4. Three signals combined into one final score:
   - 🎯 Skill Match Score — **70%**
   - 📅 Experience Score — **20%**
   - ⚡ Activity Score — **10%**
5. All candidates ranked and exported as CSV instantly

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Candidates ranked | 50 |
| Time to rank | Under 3 seconds |
| Roles supported | Any job role |
| Top candidate score | 48.4 / 100 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| scikit-learn | TF-IDF vectorizer + Cosine Similarity |
| pandas & numpy | Data processing |
| matplotlib | Charts and visualization |
| Streamlit | Live web app deployment |
| Google Colab | Notebook development |

---

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app — main file |
| `colab_code.py` | Google Colab notebook code |
| `candidates.csv` | Sample candidate dataset (50 profiles) |
| `ranked_output.csv` | Sample ranked output |
| `results_chart.png` | Visualization chart |
| `requirements.txt` | Python dependencies |

---

## 🚀 How to Run

### Option 1 — Live Demo (Recommended)
Click the Streamlit link at the top of this page — no setup needed!

### Option 2 — Google Colab
👉 [Open in Google Colab](https://colab.research.google.com/drive/1p6pRJsb7AZ3lAqxFGtQ4rhTzefDwyCBL?usp=sharing)

1. Click the link above
2. Run all cells in order
3. `ranked_output.csv` will be generated automatically

### Option 3 — Run Locally
```bash
# Clone the repo
git clone https://github.com/irfanshafi21/your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📋 Candidate CSV Format

To use your own candidates, upload a CSV with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `name` | Candidate name | Rahul Sharma |
| `skills` | All skills space separated | Python machine learning TensorFlow |
| `experience_years` | Years of experience | 4 |
| `job_title` | Current or past role | ML Engineer |
| `activity_score` | Activity score 0-100 | 85 |
| `education` | Highest degree | M.Tech CS |

---

## ✨ Features

- ✅ Works for **any job role** — not just one fixed role
- ✅ Upload any **job description** as text or .txt file
- ✅ Upload your own **candidate dataset** as CSV
- ✅ **Adjustable scoring weights** via sliders
- ✅ **Auto keyword detection** from job description
- ✅ **Interactive charts** — bar chart + scatter plot
- ✅ **Download ranked results** as CSV instantly
- ✅ **CSV template** download for easy data entry

---

## 👤 Author

**Irfan Shafi**
- GitHub: [@irfanshafi21](https://github.com/irfanshafi21)
- Hackathon: India Runs Hackathon 2026
- Track: Track 01 — Data & AI Challenge
