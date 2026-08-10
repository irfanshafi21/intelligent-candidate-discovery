# Intelligent Candidate Discovery Platform

AI-powered resume screening and candidate ranking, built with Streamlit and Google Gemini.

## Features
- Upload multiple resumes (PDF/DOCX)
- Paste a job description
- AI parses each resume into a structured profile (skills, experience, education)
- AI scores each candidate against the JD (0-100, with a breakdown by skills/experience/education)
- Ranked candidate dashboard with matched skills, gaps, and recruiter-style summaries
- AI-generated, candidate-specific interview questions

## 1. Get free API keys

**Groq (primary — recommended, 30 requests/min free)**
1. Go to https://console.groq.com
2. Sign in with Google/GitHub/email
3. Go to **API Keys** → **Create API Key**
4. Copy the key — no credit card needed

**Gemini (automatic fallback — optional but recommended)**
1. Go to https://aistudio.google.com
2. Sign in with your Google account
3. Click **Get API key** → **Create API key**
4. Copy the key

You only need one to run the app, but having both means the app automatically
switches to Gemini if Groq is ever rate-limited or down — no code changes needed.

## 2. Local setup

```bash
cd icd_app
pip install -r requirements.txt
```

Create your secrets file:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Open `.streamlit/secrets.toml` and paste your key(s):
```toml
GROQ_API_KEY = "your-groq-key-here"
GEMINI_API_KEY = "your-gemini-key-here"
```

Run the app:
```bash
streamlit run app.py
```

## 3. Deploy for free on Streamlit Community Cloud
1. Push this folder to a GitHub repo (e.g. `irfanshafi21/icd-platform`)
2. Go to https://share.streamlit.io → **New app** → pick your repo/branch → main file `app.py`
3. In the app's **Settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your-groq-key-here"
   GEMINI_API_KEY = "your-gemini-key-here"
   ```
4. Deploy. Your app gets a public `*.streamlit.app` URL you can put on your portfolio/LinkedIn/resume.

**Important:** never commit `.streamlit/secrets.toml` (with your real key) to GitHub. Only the `.example` file should be committed. Add `.streamlit/secrets.toml` to your `.gitignore`.

## File structure
```
icd_app/
├── app.py              # Streamlit UI — 3 pages: Upload & Screen, Dashboard, Interview Prep
├── resume_parser.py    # PDF/DOCX text extraction
├── ai_engine.py         # Gemini API calls: parsing, scoring, interview questions
├── requirements.txt
└── .streamlit/
    └── secrets.toml.example
```

## Notes
- Scanned/image-only PDFs won't extract text (no OCR yet) — use text-based resumes for now.
- Gemini's free tier has daily rate limits; if you hit them, wait a bit or upgrade to a paid tier.
