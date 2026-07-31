# 🎯 Intelligent Candidate Discovery (ICD) Platform

AI-powered resume screening and candidate ranking system — upload resumes, paste a job description, and get an AI-ranked shortlist with match scores, ATS analysis, skill gaps, and interview questions.

Built with **Streamlit**, **Groq + Gemini LLMs**, and **Supabase**.

---

## 🧠 What It Does

1. Recruiter uploads multiple resumes (PDF / DOCX / ZIP) and pastes a job description.
2. Text is extracted from each resume (with OCR fallback for scanned/image resumes).
3. An LLM parses the resume into a structured profile (skills, experience, education).
4. The same/fallback LLM scores that profile against the job description (0–100, with a skills/experience/education breakdown, matched skills, gaps, and a recruiter-style summary).
5. An ATS score is generated (AI + local heuristics) to estimate how well the resume would survive an Applicant Tracking System.
6. Candidates are ranked on a dashboard; recruiter can shortlist, schedule interviews, generate AI interview questions, chat with an AI assistant about the pool, and export everything (CSV / Excel / PDF).
7. If Supabase is connected, all of this (jobs, candidates, interviews) is saved permanently; otherwise it works entirely in-session.

```
Upload Resumes + JD → Text Extraction (OCR if needed) → AI Parsing 
→ AI Scoring + ATS Check → Ranked Dashboard → Interview Qs / Reports / DB Save
```

---

## 🤖 AI Used

| Purpose | Provider / Model |
|---|---|
| Resume parsing, scoring, ATS analysis, interview questions, AI chat assistant | **Groq** — `llama-3.3-70b-versatile` (primary), with `llama-3.1-70b-versatile` and `llama-3.1-8b-instant` as automatic backups |
| Fallback provider (if Groq is unavailable/rate-limited/no key) | **Google Gemini** — `gemini-3.5-flash` (with `gemini-3-flash`, `gemini-2.5-flash`, `gemini-flash-latest` as further fallbacks) |

- The app calls Groq first (fast, generous free tier). If it fails or hits rate limits after retries, it automatically switches to Gemini — no user action needed.
- **Note:** this project uses **Generative AI (LLM API calls with prompt engineering)**, not classical/trained Machine Learning. There's no model training, datasets, or ML pipeline — resume understanding and scoring is done by prompting a general-purpose LLM. ATS scoring also has a **rule-based local heuristic** (keyword/section checks) that runs without any API call, as a lightweight/offline supplement to the AI score.

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| UI / App | Streamlit |
| AI | Groq (Llama 3.x) + Google Gemini (fallback) |
| Resume parsing | pypdf, python-docx, pytesseract (OCR) |
| Database | Supabase (Postgres) |
| Charts | Plotly |
| Reports | pandas, openpyxl (Excel), reportlab (PDF) |

---

## ✨ Features

- Bulk resume upload (PDF/DOCX/ZIP), processed in parallel
- AI resume parsing → structured candidate profiles
- AI match scoring (0–100) with skill gaps & recruiter summary
- ATS scoring (AI + local heuristic)
- Fake/non-resume detection
- Job posting management (create/edit/archive)
- Ranked, filterable candidate dashboard + analytics charts
- AI-generated, candidate-specific interview questions
- Interview scheduling
- AI chat assistant over your candidate pool
- CSV / Excel / PDF exports
- Persistent history via Supabase (optional — session-only fallback if not connected)
- Light/dark mode

---

## 📁 File Structure

```
icd_app/
├── app.py               # Streamlit UI — all pages, navigation, session state
├── ai_engine.py          # Groq/Gemini calls: parsing, scoring, ATS, interview Qs, chat
├── resume_parser.py      # Text extraction (PDF/DOCX/OCR) + local heuristics
├── db.py                 # Supabase client + CRUD (candidates, jobs, interviews)
├── reports.py             # CSV / Excel / PDF report generation
├── local_settings.py      # Local app settings/helpers
├── requirements.txt
├── supabase_schema.sql    # DB schema (run once in Supabase)
└── .streamlit/secrets.toml  # API keys & Supabase credentials (never commit)
```

**Pages:** Upload & Screen · Jobs · Dashboard · Candidates · Interview Prep · Interviews · Analytics · AI Assistant · Settings

---

## 🚀 Setup

**1. Get free API keys**
- Groq: [console.groq.com](https://console.groq.com) → API Keys → Create Key
- Gemini (optional fallback): [aistudio.google.com](https://aistudio.google.com) → Get API key

**2. (Optional) Supabase** — create a free project at [supabase.com](https://supabase.com), run `supabase_schema.sql` in the SQL editor, and grab your Project URL + anon key.

**3. Install & configure**
```bash
cd icd_app
pip install -r requirements.txt
```
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-groq-key"
GEMINI_API_KEY = "your-gemini-key"
SUPABASE_URL = "your-supabase-url"
SUPABASE_KEY = "your-supabase-anon-key"
```

**4. Run**
```bash
py -m streamlit run app.py   # Windows
streamlit run app.py         # macOS/Linux
```

**5. Deploy free** — push to GitHub → [share.streamlit.io](https://share.streamlit.io) → New app → add the same secrets in Settings → Secrets.

⚠️ Never commit `secrets.toml` with real keys — add it to `.gitignore`.

---

## 👤 Author

**Shafi (Irfan Shafi)** — [GitHub](https://github.com/irfanshafi21) · [Portfolio](https://irfanshafi21.github.io/portfolio) · irfanshafi210608@gmail.com
