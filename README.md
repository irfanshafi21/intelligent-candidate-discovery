# 🤖 Intelligent Candidate Discovery System
### India Runs Hackathon 2026 — Track 01: Data & AI Challenge

## 📌 Problem Statement
Recruiters struggle to find the right candidates from thousands 
of profiles using traditional keyword filters. Hidden gems are 
missed because true potential goes beyond keywords.

## 💡 My Solution
An AI-powered ranking system that:
- Understands job descriptions deeply using NLP
- Scores candidates using TF-IDF semantic similarity
- Combines skill match + experience + activity signals
- Outputs a precise ranked shortlist instantly

## 🧠 How It Works
1. Input: Job Description text
2. TF-IDF vectorizer converts JD + candidate skills to vectors
3. Cosine similarity measures semantic fit (60% weight)
4. Experience score added (20% weight)
5. Activity score added (20% weight)
6. Final ranked CSV output generated

## 🛠️ Tech Stack
- Python 3
- Google Colab
- scikit-learn (TF-IDF, Cosine Similarity)
- pandas & numpy
- matplotlib & seaborn

## 📊 Results
- 50 candidates ranked successfully
- Top candidate achieved 85%+ match score
- Ranking chart generated and saved

## 📁 Files
| File | Description |
|------|-------------|
| `notebook.ipynb` | Complete Colab code |
| `candidates.csv` | Candidate dataset |
| `ranked_output.csv` | Final ranked output |
| `results_chart.png` | Visualization charts |

## 🚀 How to Run
1. Open notebook.ipynb in Google Colab
2. Run all cells in order
3. ranked_output.csv will be generated automatically

## 👤 Author
- India Runs Hackathon 2026 participant
- Track 01 — Data & AI Challenge
