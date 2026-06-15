# 🤖 Intelligent Candidate Discovery System
### India Runs Hackathon 2026 — Track 01: Data & AI Challenge

🔗 **Live Demo:** [your-streamlit-url-here]

## 📌 Problem Statement
Indian companies receive 400+ job applications per role.
Manual screening wastes 8–12 hours per hire and misses
the best candidates. Traditional keyword filters fail to
understand context and true skill fit.

## 💡 Solution
An AI-powered recruiter that works for **any job role**:
- Upload any job description (any role, any industry)
- Upload your own candidate CSV or use the default dataset
- Instantly get a ranked shortlist with scores
- Download results as CSV

## 🧠 How It Works
1. Paste or upload any job description
2. TF-IDF vectorizer converts JD + candidate skills to vectors
3. Cosine similarity measures semantic skill fit
4. Three signals combined into one final score:
   - Skill Match — 70%
   - Experience — 20%
   - Activity Score — 10%
5. Candidates ranked and exported as CSV

## 🛠️ Tech Stack
- Python 3
- Streamlit (live web app)
- Google Colab (notebook)
- scikit-learn (TF-IDF, Cosine Similarity)
- pandas, numpy, matplotlib

## 📊 Results
- 50 candidates ranked in under 3 seconds
- Works for any job role — not just one fixed role
- Adjustable scoring weights via sliders
- Download ranked output as CSV instantly

## 📁 Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit web app |
| `notebook.ipynb` | Google Colab notebook |
| `candidates.csv` | Sample candidate dataset |
| `ranked_output.csv` | Sample ranked output |
| `results_chart.png` | Visualization chart |
| `requirements.txt` | Python dependencies |

## 🚀 How to Run

**Option 1 — Live Demo**
Click the Streamlit link above

**Option 2 — Google Colab**
1. Open `notebook.ipynb` in Google Colab
2. Run all cells in order
3. `ranked_output.csv` will be generated

**Option 3 — Local**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👤 Author
- India Runs Hackathon 2026 participant
- Track 01 — Data & AI Challenge
