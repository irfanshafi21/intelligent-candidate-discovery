# 🎯 Intelligent Candidate Discovery (ICD) Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge&logo=groq&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-Fallback_LLM-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![License](https://img.shields.io/badge/License-Portfolio_Use-00668A?style=for-the-badge)

</div>

**AI-powered resume screening, ATS scoring, and candidate ranking system.**

Upload resumes, paste a job description, and get an AI-ranked shortlist with match scores, ATS analysis, skill gaps, recruiter-style summaries, auto-generated interview questions, and exportable reports — all from a single Streamlit dashboard.

Built by **Shafi (Irfan Shafi)** — 2nd-year AI & ML student, Kongu Engineering College — as a portfolio project demonstrating applied LLM integration, structured data extraction, and full-stack app design.

<div align="center">

`Resume Parsing` • `ATS Scoring` • `Candidate Ranking` • `Interview Prep` • `AI Chat Assistant` • `CSV / Excel / PDF Exports`

</div>

---

## 📑 Table of Contents

<table>
<tr>
<td valign="top" width="33%">

### 🧭 Get Started
- [Overview](#-overview)
- [Problem It Solves](#-problem-it-solves)
- [Who It's For](#-who-its-for)
- [Core Features](#-core-features)
- [Tech Stack](#-tech-stack)
- [🎨 Design & Visual Identity](#-design--visual-identity)

### 🏗 How It Works
- [System Architecture](#-system-architecture)
- [End-to-End Workflow](#-end-to-end-workflow)
- [AI Layer — Models, Providers & Prompts](#-ai-layer--models-providers--prompts)
- [Is This Machine Learning?](#-is-this-machine-learning)

</td>
<td valign="top" width="33%">

### 🔍 Core Logic
- [Resume Text Extraction & OCR](#-resume-text-extraction--ocr)
- [ATS Scoring Logic](#-ats-scoring-logic)
- [Candidate Matching & Ranking](#-candidate-matching--ranking)
- [Database Layer (Supabase)](#-database-layer-supabase)
- [File-by-File Breakdown](#-file-by-file-breakdown)
- [Application Pages](#-application-pages)
- [Reports & Exports](#-reports--exports)

### 🔐 Trust & Reliability
- [Security Notes](#-security-notes)
- [Error Handling & Reliability](#-error-handling--reliability)
- [Known Limitations](#-known-limitations)

</td>
<td valign="top" width="33%">

### 🚀 Run It Yourself
- [Setup Guide](#-setup-guide)
- [Deployment Guide](#-deployment-guide)
- [Environment Variables](#-environment-variables)
- [Testing Checklist (Manual QA)](#-testing-checklist-manual-qa)

### 📚 Reference
- [Design Decisions & Trade-offs](#-design-decisions--trade-offs)
- [Frequently Asked Questions](#-frequently-asked-questions)
- [Future Scope](#-future-scope)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

</td>
</tr>
</table>

---

## 🧠 Overview

The Intelligent Candidate Discovery (ICD) Platform automates the first pass of resume screening — the repetitive, time-consuming part of recruiting where a human has to skim dozens or hundreds of resumes to find the ones worth a closer look.

A recruiter (or hiring manager, or student practicing for placements) provides:
- A **job description** (title + key requirements)
- A batch of **resumes** (individual PDFs/DOCX/images, or a ZIP folder of many)

The platform then:
1. Extracts clean text from every resume, regardless of format
2. Filters out anything that isn't actually a resume
3. Uses a large language model (LLM) to read each resume and turn it into a structured candidate profile
4. Uses the same LLM to **score** that profile against the job description
5. Runs a separate **ATS compatibility check** (partly AI, partly local/rule-based)
6. Ranks every candidate and presents them on an interactive dashboard
7. Lets the recruiter generate **tailored interview questions**, schedule interviews, chat with an AI assistant about the candidate pool, and export everything as CSV / Excel / PDF

Everything works with **zero cost** (free-tier APIs) and can run **entirely in-session** (no database required) or with **persistent history** if a free Supabase project is connected.

---

## ❓ Problem It Solves

Manual resume screening is slow, inconsistent, and biased toward superficial signals (keyword stuffing, resume design, familiar university names). For a single job opening, recruiters may receive 50–500+ applications. Reading each one carefully takes minutes; skimming risks missing good candidates or over-weighting bad ones.

This project solves that by:
- Giving every resume the **same** evaluation criteria, applied consistently
- Producing a **transparent breakdown** (skills / experience / education) instead of a single opaque score
- Surfacing **specific gaps**, not just a rejection
- Cutting the "first pass" review time from hours to minutes
- Giving the recruiter a **starting point**, not a final decision — humans remain in the loop for every hire

---

## 👥 Who It's For

- **Recruiters / HR teams** at small companies without an enterprise ATS
- **Hiring managers** doing their own first-pass screening
- **Hackathon / placement cell** style bulk shortlisting (e.g. college placement drives)
- **Students** (like the author) building a practical, demo-able AI product for a portfolio, interview, or hackathon submission

---

## ✨ Core Features

| Feature | Description |
|---|---|
| **Bulk resume upload** | Upload individual PDF/DOCX/JPG/PNG files, or a single ZIP containing many resumes |
| **Parallel processing** | Multiple resumes are processed concurrently (`ThreadPoolExecutor`) instead of one-by-one |
| **AI resume parsing** | Extracts name, email, phone, years of experience, education, skills, past roles, and a neutral summary |
| **AI candidate scoring** | 0–100 overall match score, broken down into Skills Match / Experience Fit / Education Fit, plus matched skills and gaps |
| **Configurable priority weighting** | Recruiter decides how much Skills vs Experience vs Education should matter for ranking |
| **ATS compatibility analysis** | Hybrid AI + local-heuristic score covering keyword coverage, grammar, structure, formatting, and readability |
| **Non-resume detection** | Two-pass check (fast local heuristic → AI confirmation) filters out cover letters, certificates, IDs, or unrelated files without silently dropping unusual real resumes |
| **Job management** | Create, edit, and archive job postings; reuse them across screening sessions |
| **Ranked candidate dashboard** | Sortable, filterable, searchable list with analytics charts (Plotly) |
| **AI interview question generator** | Produces role- and candidate-specific questions across 4 categories, each with a "what a strong answer looks like" rubric |
| **Interview scheduling** | Track upcoming interviews per candidate with status |
| **AI chat assistant** | Ask natural-language questions about the current candidate pool; answers are grounded in actual screening data, not invented |
| **Exports** | CSV, Excel (.xlsx), and PDF — individual candidate reports, shortlist reports, and interview reports |
| **Persistent history** | Optional Supabase backend keeps candidates/jobs/interviews across sessions and devices |
| **Light / dark mode** | Full theme toggle |
| **Graceful degradation** | Every external dependency (LLM provider, database) has a fallback path — the app never hard-crashes because one service is down |

---

## 🧱 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Frontend / App framework** | [Streamlit](https://streamlit.io) | Fastest way to ship a data-driven Python UI without writing separate frontend/backend code; ideal for a solo-built AI app |
| **Primary LLM provider** | [Groq](https://groq.com) (Llama 3.x models) | Extremely fast inference (LPU hardware), generous free tier, native JSON mode |
| **Fallback LLM provider** | [Google Gemini](https://aistudio.google.com) | Free tier, reliable, used automatically if Groq fails or isn't configured |
| **Database** | [Supabase](https://supabase.com) (hosted Postgres) | Free tier, instant REST API via `supabase-py`, no server to manage |
| **PDF text extraction** | `pypdf` | Lightweight, pure-Python PDF parsing |
| **DOCX text extraction** | `python-docx` | Reads paragraphs and table cells |
| **OCR (image resumes)** | `pytesseract` (Tesseract OCR engine) + `Pillow` | Converts scanned/photographed resumes into text |
| **Charts** | `Plotly` | Interactive analytics visualizations inside Streamlit |
| **Excel export** | `pandas` + `openpyxl` | DataFrame → `.xlsx` |
| **PDF report generation** | `reportlab` | Programmatic PDF report building (tables, styles, paragraphs) |

---

## 🎨 Design & Visual Identity

The dashboard uses a custom "Sky" design system — a calm, professional blue palette layered over Streamlit's default theme via injected CSS in `app.py`. Typography pairs **Plus Jakarta Sans** (headings — bold, geometric) with **Inter** (body text — highly legible at small sizes), both loaded from Google Fonts.

### Color Palette

| Swatch | Token | Hex | Used For |
|---|---|---|---|
| ![#00668A](https://placehold.co/60x24/00668A/00668A.png) | `--primary` | `#00668A` | Primary buttons, active tab, links, section labels |
| ![#004965](https://placehold.co/60x24/004965/004965.png) | `--primary-dark` | `#004965` | Headings, hover states, high-emphasis text |
| ![#39B8FD](https://placehold.co/60x24/39B8FD/39B8FD.png) | `--secondary-container` | `#39B8FD` | Sidebar accent line, highlight badges |
| ![#E0F2FE](https://placehold.co/60x24/E0F2FE/E0F2FE.png) | `--secondary` | `#E0F2FE` | Pill backgrounds, chips, soft highlight cards |
| ![#0B1C30](https://placehold.co/60x24/0B1C30/0B1C30.png) | `--text` | `#0B1C30` | Primary body text (light mode) |
| ![#3E484F](https://placehold.co/60x24/3E484F/3E484F.png) | `--text-secondary` | `#3E484F` | Secondary/muted text, captions |
| ![#BDC8D1](https://placehold.co/60x24/BDC8D1/BDC8D1.png) | `--border` | `#BDC8D1` | Card borders, dividers, input outlines |
| ![#1E293B](https://placehold.co/60x24/1E293B/1E293B.png) | Sidebar background | `#1E293B` | Dark navy sidebar, contrasts with light content area |
| ![#38BDF8](https://placehold.co/60x24/38BDF8/38BDF8.png) | Sidebar accent | `#38BDF8` | Sidebar branding text, active nav item, connection status |

Score-driven UI elements (match score chips, ATS verdicts) additionally use a semantic **green / amber / red** scale layered on top of the base palette, so a recruiter can gauge candidate quality at a glance without reading every number:

| Verdict | Meaning |
|---|---|
| 🟢 Green | Strong match (High ATS compatibility / high overall score) |
| 🟡 Amber | Partial match — worth a manual look |
| 🔴 Red | Weak match / low ATS compatibility |

### Typography

| Role | Font | Weight |
|---|---|---|
| Headings, candidate names, hero title | Plus Jakarta Sans | 700–800 (Bold / Extrabold) |
| Body text, tables, form inputs | Inter | 400–600 (Regular–Semibold) |
| Sidebar labels / eyebrow text | Inter | 700–800, uppercase, letter-spaced |

### UI Principles Applied

- **Left-border cards** — content cards use a 4px `--primary` left border instead of a full outline, giving structure without visual heaviness
- **Soft containers over hard boxes** — pill-shaped skill tags and score chips use `--secondary` backgrounds with `--primary-dark` text, rather than harsh borders
- **Dark sidebar / light canvas contrast** — the navy sidebar (`#1E293B`) anchors navigation and status, while the main content area stays light and high-contrast for reading dense candidate data
- **Consistent iconography** — every page and section uses a single leading emoji as a lightweight visual anchor (📤 Upload, 📊 Dashboard, 🗣️ Interview Prep, etc.) instead of custom icon assets
- **Full light/dark mode** — a dedicated dark-mode stylesheet override (see `app.py`, toggled from Settings) remaps text/background/border tokens without duplicating component logic

---

## 🏗 System Architecture

The app follows a simple **modular monolith** pattern — one Streamlit process, with responsibilities cleanly separated into single-purpose modules rather than one giant script:

```
                         ┌─────────────────────┐
                         │        app.py         │   Streamlit UI, session
                         │  (pages, navigation)   │   state, orchestration
                         └───────────┬───────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
   ┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
   │  resume_parser.py   │   │    ai_engine.py     │   │       db.py         │
   │  Text extraction     │   │  LLM calls (Groq/     │   │  Supabase CRUD       │
   │  + OCR + heuristics   │   │  Gemini), prompts     │   │  (candidates/jobs/    │
   └─────────────────────┘   └─────────────────────┘   │   interviews)         │
                                                          └─────────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │      reports.py         │  CSV / Excel / PDF
                         │   (export generation)    │  report builders
                         └─────────────────────────┘
```

**Design principles applied throughout:**
- **Separation of concerns** — UI (`app.py`), AI (`ai_engine.py`), parsing (`resume_parser.py`), persistence (`db.py`), and export (`reports.py`) never mix responsibilities.
- **Graceful degradation** — every external call (LLM, database) is wrapped so failure returns an empty/None result or falls back to another provider, instead of crashing the app.
- **Best-effort persistence** — Supabase writes never block the core screening flow; if a save fails, the app still shows the result to the user and simply keeps it session-local.
- **Provider abstraction** — `ai_engine.py` exposes a single unified `_call_json()` entry point; the rest of the app never needs to know whether Groq or Gemini actually answered.

---

## 🔄 End-to-End Workflow

```
 1. User opens the app
        │
        ▼
 2. Sidebar shows API key status + DB connection status
        │
        ▼
 3. User goes to "Upload & Screen"
        │
        ▼
 4. User enters job title + key requirements (or picks a saved Job)
    Sets priority weighting (Skills / Experience / Education) and "Top N"
        │
        ▼
 5. User uploads resumes — individual files or a ZIP
        │
        ▼
 6. For each file (processed in parallel):
        a. Extract raw text (PDF / DOCX / OCR for images)
        b. Fast local heuristic check: "does this look like a resume?"
           → borderline cases get a second AI opinion before being excluded
        c. Send resume text + job description to the LLM in ONE call:
           - Parses resume → structured profile
           - Scores profile against job description → score + breakdown
        d. Run ATS analysis (AI keyword/grammar check + local structure/
           formatting/readability heuristics)
        e. (Optional) Save the result to Supabase
        │
        ▼
 7. Dashboard shows all candidates ranked by weighted score
        │
        ▼
 8. Recruiter can:
        - Filter / search / sort candidates
        - View full profile + score breakdown + gaps
        - Generate AI interview questions for a candidate
        - Schedule an interview
        - Chat with the AI Assistant about the pool
        - Export CSV / Excel / PDF reports
        │
        ▼
 9. Everything screened this session appears in Analytics
    (merged with all-time history if Supabase is connected)
```

---

## 🤖 AI Layer — Models, Providers & Prompts

### Provider strategy

The app never hard-codes a single AI provider. `ai_engine.py` implements a **primary + fallback** strategy:

1. **Groq is tried first** — it's fast (LPU hardware, ~1–2s responses) and has a generous free tier.
2. **If Groq fails** (no key, rate-limited past retries, model retired, or any other error) **and a Gemini key is configured**, the app **automatically retries the same request on Gemini** — the user sees no interruption.
3. If neither key is configured, the app clearly tells the user no AI provider is available (via `check_api_key()` / `active_provider()`, shown live in the sidebar).

### Models used

| Purpose | Provider | Model order (first that works is used) |
|---|---|---|
| Resume parsing, scoring, ATS analysis, interview questions | **Groq** | `llama-3.3-70b-versatile` → `llama-3.1-70b-versatile` → `llama-3.1-8b-instant` → `mixtral-8x7b-32768` |
| AI chat assistant (speed-prioritized) | **Groq** | `llama-3.1-8b-instant` → `llama-3.3-70b-versatile` → `llama-3.1-70b-versatile` → `mixtral-8x7b-32768` |
| Fallback for all of the above | **Gemini** | `gemini-3.5-flash` → `gemini-3-flash` → `gemini-2.5-flash` → `gemini-flash-latest` |

Both providers are called with `temperature=0.2` (low randomness — the app needs consistent, evaluative output, not creative variation) and **native JSON response mode**, so the model is constrained to return valid JSON rather than free-form prose.

### How each AI call works

All AI calls funnel through one private function, `_call_json(prompt)`, which:
1. Tries Groq across its full model fallback list, retrying each model up to 3 times with backoff on rate-limit (429) errors
2. If a model is retired/not found (404), moves to the next model in the list automatically
3. If Groq is exhausted and Gemini is configured, retries the whole request on Gemini the same way
4. Cleans the raw response text (`_sanitize_json_text`) — strips markdown code fences, removes stray control characters, fixes trailing commas — before `json.loads()`, since LLMs occasionally wrap JSON in explanatory text or minor formatting noise
5. Raises a combined error only if **every** provider/model combination fails

### The five AI prompts used in this project

**1. `parse_and_score()` — resume parsing + scoring in a single call (main path)**
Combines parsing and scoring into one prompt so each candidate only needs **one** API round-trip instead of two, roughly halving latency per resume. Given the job description and resume text, the model returns:
```json
{
  "profile": {
    "name": "...", "email": "...", "phone": "...",
    "years_experience": "...", "education": "...",
    "skills": ["..."], "past_roles": ["..."], "summary": "..."
  },
  "score": {
    "overall_score": 0-100,
    "breakdown": { "skills_match": 0-100, "experience_fit": 0-100, "education_fit": 0-100 },
    "matched_skills": ["..."], "gaps": ["..."], "summary": "..."
  }
}
```
The prompt explicitly instructs the model to be *"objective and evidence-based; judge substance over keywords alone; don't penalize non-traditional resume formats"* — reducing the risk of penalizing candidates purely for resume formatting choices.

**2. `parse_resume_with_ai()` — standalone parsing**
Same profile schema as above, kept separate for cases where only parsing (no scoring) is needed.

**3. `score_candidate()` — standalone re-scoring**
Re-scores an already-parsed profile against a *different* job description without re-parsing the resume — useful for evaluating one candidate against multiple open roles.

**4. `analyze_ats_ai()` — ATS keyword & writing-quality analysis**
Focused specifically on the parts of ATS analysis that need language understanding (keyword coverage, grammar/clarity, missing terms, actionable suggestions, and a High/Medium/Low compatibility verdict) — structural/formatting metrics are computed separately without any AI call (see [ATS Scoring Logic](#-ats-scoring-logic)).

**5. `is_resume_ai_check()` — second-pass "is this actually a resume?" check**
Only triggered for files that failed the fast local heuristic — a safety net so an unusually formatted *real* resume doesn't get silently excluded on a single weak signal.

**6. `generate_interview_questions()` — interview guide generation**
Produces 3–4 questions in each of four categories — *Technical Validation, Experience Deep-Dive, Gap Probing, Culture & Motivation* — each paired with a `what_good_looks_like` field: concrete, specific points a strong answer should cover, grounded in the JD and that candidate's actual background (not generic interview advice).

**7. `ask_assistant()` — the AI Assistant chat**
A general-purpose assistant scoped to this app via a hard-coded `APP_KNOWLEDGE` system context describing every page and feature. When candidates have already been screened, answers are grounded **strictly** in their actual structured screening data (scores, matched skills, gaps) rather than the model inventing plausible-sounding details.

### Prompt engineering techniques used

- **Strict JSON schema instructions** in every prompt, with explicit type hints (`<integer 0-100>`) so downstream code can parse reliably
- **Role priming** ("You are an expert technical recruiter...") to bias the model toward the right tone and rigor
- **Explicit fairness instructions** ("don't penalize non-traditional resume formats", "judge substance over keywords alone") to reduce superficial bias
- **Grounding instructions** for the chat assistant, to prevent hallucinated candidate details
- **Context truncation** — resume text and job descriptions are truncated (e.g. `raw_text[:8000]`, `job_description[:4000]`) to control token usage and stay within model context limits
- **Native JSON mode** (`response_format={"type": "json_object"}` for Groq, `response_mime_type="application/json"` for Gemini) instead of relying purely on prompt instructions to produce valid JSON
- **Single combined-call optimization** (parse + score together) to reduce latency and API usage per resume

---

## 🔬 Is This Machine Learning?

**No — this project does not use trained/classical Machine Learning.** There is no dataset, no model training step, no `.pkl`/`.h5` saved model, no scikit-learn pipeline, and no accuracy/precision/recall metrics anywhere in the codebase.

What this project actually uses is **Generative AI via LLM APIs (Groq, Gemini)** — i.e., prompting a pre-trained, general-purpose large language model to perform parsing and evaluation tasks through carefully engineered prompts, rather than training a custom model on labeled resume data.

| | Classical Machine Learning | What this project does (Generative AI / LLM API) |
|---|---|---|
| Requires a dataset | Yes — labeled training data | No — the LLM was pre-trained by its provider |
| Training step | Yes (fit/train a model) | No — API calls to an already-trained model |
| Output | Learned numeric predictions from patterns in your own data | Reasoned, language-based output generated per-prompt |
| Evaluation metrics | Accuracy, Precision, Recall, F1, Cross-validation | Not applicable — no model to evaluate this way |
| What you control | Model architecture, hyperparameters, features | Prompt design, provider/model choice, temperature |

It's a common point of confusion because both fall under the broader "AI" umbrella, but **calling an LLM API is not the same as building or training an ML model.** The one place this project *does* use non-AI computation is the **local ATS heuristics** (see below) — pure rule-based Python (regex, keyword counting, ratio math), no AI involved at all.

---

## 📄 Resume Text Extraction & OCR

Handled entirely in `resume_parser.py`. Supported input types: `.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`, and `.zip` (a folder of any of the above).

| Format | Method | Function |
|---|---|---|
| PDF | `pypdf.PdfReader` — extracts text per page and joins | `_extract_pdf_text()` |
| DOCX | `python-docx` — reads paragraphs **and** table cells (resumes sometimes use tables for layout) | `_extract_docx_text()` |
| Image (PNG/JPG/JPEG) | OCR via `pytesseract` (Tesseract engine), image converted to RGB first | `_extract_image_text()` |
| ZIP | Unpacked in-memory; skips `__MACOSX`/`.DS_Store` junk and unsupported extensions | `extract_files_from_zip()` |

**OCR details:**
- On Windows, the app auto-detects Tesseract at its default install paths (`C:\Program Files\Tesseract-OCR\tesseract.exe`) so users don't need to manually edit their system PATH.
- If Tesseract isn't installed, the app raises a clear, actionable error pointing to the installer.
- If OCR extracts fewer than 20 characters, the file is rejected with a message suggesting a clearer/higher-resolution scan — rather than silently proceeding with near-empty text.
- A scanned PDF with **no embedded text layer** (i.e. actually just an image) is also caught — `pypdf` returning empty text triggers a clear error rather than a silent failure.

**Non-resume filtering (two-pass check):**
1. **Fast local heuristic** (`heuristic_resume_check`) — no API call, pure Python. Scans for resume section keywords (`experience`, `education`, `skills`, `projects`, etc.) via a hardcoded list, and looks for an email/phone pattern via regex. A "strong pass" requires ≥2 section keywords **and** contact info; a "medium pass" requires ≥3 section keywords even without contact info.
2. **AI confirmation** (`is_resume_ai_check`) — only called for files that fail step 1, as a second opinion, so an unusually formatted real resume isn't dropped on one weak signal alone.

---

## 📊 ATS Scoring Logic

ATS analysis is intentionally **hybrid** — split between what genuinely needs language understanding (AI) and what's a pure computation over the text (local, free, deterministic):

**AI-judged (via `analyze_ats_ai()` in `ai_engine.py`):**
- `keyword_score` (0–100) — how well the resume's terminology covers the job description's key terms
- `grammar_score` (0–100) — writing clarity, tense consistency, absence of typos
- `missing_keywords` — important JD terms/skills absent from the resume
- `suggestions` — concrete, specific improvement suggestions
- `ats_compatibility` — High / Medium / Low, with a one-sentence reason

**Locally computed (via `compute_local_ats_metrics()` in `resume_parser.py`, zero API calls):**
- `structure_score` — percentage of standard resume sections found (`experience`, `education`, `skills`, `summary`, `objective`, `projects`, `certification`) out of the full expected set
- `formatting_score` — based on bullet-point usage ratio, penalized by a "noise ratio" (fraction of characters outside normal alphanumeric/punctuation — a proxy for OCR garbage or parsing artifacts)
- `readability_score` — penalizes sentence length outside an 8–20 word ideal range, and total resume length outside a 250–900 word ideal range

This split exists because keyword relevance and grammar genuinely require language understanding, while section presence, bullet formatting, and sentence-length math are deterministic and don't benefit from (or need the cost/latency of) an LLM call.

---

## 🏆 Candidate Matching & Ranking

1. Each candidate gets an `overall_score` (0–100) and a `breakdown` of `skills_match`, `experience_fit`, and `education_fit` — all produced directly by the LLM's evaluation of the resume against the job description (not a keyword-overlap formula).
2. The recruiter sets a **priority weighting** on the Upload & Screen page for how much each of Skills / Experience / Education should matter — the dashboard ranks candidates using this weighting rather than a flat, one-size-fits-all average.
3. `matched_skills` and `gaps` are returned per candidate so the recruiter sees *why* a score landed where it did, not just the number.
4. A `Top N` setting controls how many candidates are highlighted as the shortlist.
5. Local keyword extraction (`extract_keywords()` in `app.py`) pulls meaningful technical/role terms out of the job description using a lightweight regex-and-stopword-filter approach — no API call — mainly used for quick UI display of key JD terms, preserving tokens like `ai/ml`, `c++`, `ci/cd` that naive word-splitting would otherwise mangle.

---

## 🗄 Database Layer (Supabase)

Persistence is entirely optional — `db.is_configured()` checks whether `SUPABASE_URL` and `SUPABASE_KEY` are set (via `st.secrets` or environment variables); if not, every `db.*` function returns an empty result and the app quietly runs session-only.

**Design principle:** persistence is *best-effort and non-blocking*. A failed Supabase write never surfaces as an error mid-screening — the candidate is still shown to the user; it just won't appear in cross-session Analytics.

### Tables used (inferred from `db.py`)

**`screening_history`** — one row per screened candidate
| Column | Description |
|---|---|
| `job_role`, `job_details`, `job_id` | The job this screening was run against (job_id links to `jobs` if a saved job was used) |
| `candidate_name`, `filename` | Identity of the source resume |
| `email`, `phone`, `years_experience`, `education` | Parsed profile fields |
| `skills`, `past_roles` (jsonb) | Parsed lists |
| `overall_score`, `skills_match`, `experience_fit`, `education_fit` | Scoring fields |
| `matched_skills`, `gaps` (jsonb) | Scoring detail lists |
| `recruiter_summary` | AI-generated verdict text |
| `screened_at` | Timestamp (used for ordering history) |

**`jobs`** — saved job postings
| Column | Description |
|---|---|
| `title`, `description`, `required_skills` (jsonb), `status` (`active`/`archived`) | Job posting fields |
| `created_at`, `updated_at` | Timestamps |

**`interviews`** — scheduled interviews
| Column | Description |
|---|---|
| `candidate_name`, `job_role`, `interview_type`, `status` | Interview metadata |
| `scheduled_at` | Timestamp used for ordering upcoming interviews |

### CRUD functions

| Function | Table | Operation |
|---|---|---|
| `save_screening_record()` | `screening_history` | INSERT |
| `fetch_screening_history()` | `screening_history` | SELECT, ordered by `screened_at` desc |
| `save_job()` / `update_job()` / `delete_job()` / `fetch_jobs()` | `jobs` | Full CRUD |
| `save_interview()` / `update_interview()` / `fetch_interviews()` | `interviews` | Create, update, list (ordered by `scheduled_at` asc) |

The Supabase client itself is created once per session via `@st.cache_resource` (not re-instantiated on every call), and the app tracks `get_last_error()` so a connection/query failure can be surfaced in the Settings page instead of failing invisibly.

> **Note:** this repository does not include a bundled `.sql` schema file — the table/column shapes above are reconstructed directly from what `db.py` reads and writes. If setting up your own Supabase project, create these three tables (`screening_history`, `jobs`, `interviews`) with the columns listed above before connecting.

---

## 📂 File-by-File Breakdown

### `app.py` (~2,150 lines) — Streamlit UI & orchestration
The entry point. Defines `st.set_page_config`, all 9 pages, sidebar navigation, session-state initialization, dark-mode CSS override, the sidebar's live API-key/DB-connection status, local keyword extraction, and the merged local+remote jobs store (`get_jobs()` — combines Supabase jobs with any session-local ones that fell back due to a save failure, so a job never silently disappears). Imports and calls into every other module; contains no AI-calling logic itself.

### `ai_engine.py` (~630 lines) — LLM integration layer
Everything AI-related: provider key management, the Groq/Gemini calling logic with retries and fallback, JSON response sanitization, and all seven prompt-building functions described above ([AI Layer](#-ai-layer--models-providers--prompts)).

### `resume_parser.py` (~230 lines) — Text extraction & local heuristics
PDF/DOCX/image text extraction, ZIP unpacking, the two-tier resume-detection heuristic, and the fully local ATS sub-score computation (`compute_local_ats_metrics`).

### `db.py` (~230 lines) — Supabase persistence
Client creation/caching, and CRUD for `screening_history`, `jobs`, and `interviews`, all wrapped to degrade gracefully when Supabase isn't configured or a call fails.

### `reports.py` (~180 lines) — Export generation
Converts candidate/interview data into `pandas` DataFrames, then into CSV bytes, Excel bytes (`openpyxl`), or styled PDF reports (`reportlab`) — individual candidate reports, shortlist reports, and interview reports. Operates only on data already computed elsewhere; never fabricates figures.

### `local_settings.py` (~40 lines) — Local app settings/helpers
Small helper module for local app configuration used across pages.

### `requirements.txt` — Python dependencies
```
streamlit>=1.35.0
groq>=0.11.0
google-generativeai>=0.7.0
pypdf>=4.2.0
python-docx>=1.1.0
pillow>=10.0.0
pytesseract>=0.3.10
plotly>=5.20.0
supabase>=2.7.0
pandas>=2.0.0
openpyxl>=3.1.0
reportlab>=4.0.0
```

### `.streamlit/secrets.toml`
Holds `GROQ_API_KEY`, `GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`. **Never commit this file with real values** — see [Security Notes](#-security-notes).

---

## 🧭 Application Pages

| Page | Purpose |
|---|---|
| **📤 Upload & Screen** | Enter job title/requirements, set priority weighting and Top N, upload resumes (files or ZIP), run screening |
| **📋 Jobs** | Create, edit, and archive saved job postings for reuse |
| **📊 Dashboard** | Ranked shortlist with scores, breakdowns, matched skills, gaps, summaries |
| **👥 Candidates** | Full candidate list with search/filter |
| **🗣️ Interview Prep** | AI-generated interview questions per candidate, grouped by category |
| **📅 Interviews** | Schedule and track interviews |
| **📈 Analytics** | Charts and trends across screened candidates (session + Supabase history) |
| **🤖 AI Assistant** | Chat interface grounded in the current candidate pool and app knowledge |
| **⚙️ Settings** | API key overrides (session-level), connection status, diagnostics |

---

## 📦 Reports & Exports

| Export | Format | Contents |
|---|---|---|
| Candidate table | CSV / Excel | All screened candidates with scores, breakdowns, matched skills, gaps, contact info |
| Individual candidate report | PDF | Full profile + score breakdown + AI summary for one candidate |
| Shortlist report | PDF | Formatted, presentation-ready shortlist for sharing with stakeholders |
| Interview report | PDF / CSV / Excel | Scheduled interviews with candidate, role, type, time, status |

All export functions live in `reports.py` and operate purely on data the app already computed — no additional API calls are made during export.

---

## 🔐 Security Notes

- **API keys and DB credentials** are read from `st.secrets` (`.streamlit/secrets.toml`) or environment variables — never hard-coded in source.
- `.streamlit/secrets.toml` must be excluded from version control (`.gitignore`); only a `.example` template (without real values) should ever be committed.
- A **session-level key override** exists on the Settings page, letting a user test a different key for the current session only — without touching the underlying secrets file.
- Supabase access uses the **anon/public API key** intended for client-side use; row-level security (RLS) should be configured on the Supabase project according to your own access requirements (not enforced by this codebase, which assumes a trusted single-tenant use case).
- Resume text sent to LLM providers is truncated (not sent in full for very long documents) to control token usage — this is a cost/performance measure, not a security control.
- No user authentication layer exists in this project — it's designed for a single recruiter/session use case, not multi-tenant SaaS.

---

## 🛡 Error Handling & Reliability

- **LLM calls**: automatic retry with backoff on rate limits, automatic fallback across models within a provider, automatic fallback across providers (Groq → Gemini), and JSON-response sanitization before parsing.
- **Database calls**: every `db.*` function catches exceptions and returns `False`/`None`/`[]` instead of raising — Analytics and other DB-dependent views fall back to session-only data.
- **File extraction**: empty/unreadable PDFs, DOCX, and low-quality OCR images raise clear, user-facing error messages rather than silently producing garbage input to the AI.
- **Non-resume detection**: two-pass (heuristic + AI) so a real resume in an unusual format isn't dropped on a single false signal.
- **Job storage**: `get_jobs()` merges Supabase-backed jobs with any that fell back to session-local storage due to a save failure, so a job a user just created never appears to vanish.

---

## 🚀 Setup Guide

### 1. Get free API keys

**Groq (primary — recommended)**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign in → **API Keys** → **Create API Key**
3. Copy the key (no credit card required)

**Gemini (automatic fallback — optional but recommended)**
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Google account → **Get API key** → **Create API key**
3. Copy the key

You only need one key to run the app; with both configured, Gemini kicks in automatically if Groq is ever unavailable.

### 2. (Optional) Set up Supabase for persistent history

1. Create a free project at [supabase.com](https://supabase.com)
2. In the SQL editor, create the `screening_history`, `jobs`, and `interviews` tables using the columns described in [Database Layer](#-database-layer-supabase)
3. Copy your **Project URL** and **anon/public API key** from Project Settings → API

Without this step, the app still works fully — data just won't persist across sessions.

### 3. Install Tesseract OCR (for scanned/image resumes)

- **Windows:** install from [UB-Mannheim's Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

If skipped, everything else still works — only image-based resume uploads will fail with a clear error.

### 4. Install dependencies

```bash
cd icd_app
pip install -r requirements.txt
```

### 5. Configure secrets

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-groq-key-here"
GEMINI_API_KEY = "your-gemini-key-here"

SUPABASE_URL = "your-supabase-project-url"
SUPABASE_KEY = "your-supabase-anon-key"
```

### 6. Run the app

```bash
# Windows
py -m streamlit run app.py

# macOS / Linux
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

---

## ☁️ Deployment Guide

**Streamlit Community Cloud (free):**
1. Push this project to a GitHub repository (e.g. `irfanshafi21/icd-platform`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repo, branch, and set the main file to `app.py`
4. In the app's **Settings → Secrets**, paste the same key/value pairs shown above
5. Deploy — the app gets a public `*.streamlit.app` URL suitable for a portfolio, resume, or LinkedIn link

**⚠️ Important:** never commit `.streamlit/secrets.toml` with real values. Add it to `.gitignore` and only commit a `.example` version with placeholder text.

---

## 🔑 Environment Variables

| Variable | Required? | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Recommended (one of Groq/Gemini required) | Primary LLM provider |
| `GEMINI_API_KEY` | Optional | Fallback LLM provider |
| `SUPABASE_URL` | Optional | Enables persistent storage |
| `SUPABASE_KEY` | Optional | Enables persistent storage |

All four can be set either in `.streamlit/secrets.toml` (local/Streamlit Cloud) or as plain OS environment variables — `_get_secret()` in both `ai_engine.py` and `db.py` checks `st.secrets` first, then falls back to `os.environ`.

---

## ⚠️ Known Limitations

- OCR accuracy depends heavily on scan/photo quality — low-resolution or skewed images may extract poorly or fail the 20-character minimum check.
- Free-tier LLM rate limits can add latency during heavy bulk-upload sessions, even with automatic retry/fallback.
- No multi-user authentication — intended for single-recruiter or small-team use, not a multi-tenant SaaS product.
- Scoring quality is bounded by the underlying LLM's reasoning — it is not a certified or legally validated hiring tool and should support, not replace, human judgment.
- Without Supabase configured, all data is lost when the browser session ends.
- No dedicated `.sql` schema file is bundled in this repo; tables must be created manually based on the fields `db.py` expects.

---

## 🔮 Future Scope

- Bundle an official `supabase_schema.sql` migration file with Row-Level Security policies
- Add authentication/multi-tenant support for team use
- Add resume-to-resume duplicate detection
- Support additional LLM providers (e.g. OpenAI, Anthropic) in the same fallback chain
- Add semantic/embedding-based search over the candidate pool (currently keyword/heuristic-based)
- Add configurable scoring rubrics per job/industry
- Batch export of all shortlisted candidates in a single combined PDF

---

## 📁 Project Structure

```
icd_app/
├── app.py                    # Streamlit UI — all 9 pages, navigation, session state
├── ai_engine.py               # Groq/Gemini calls: parsing, scoring, ATS, interview Qs, chat
├── resume_parser.py           # Text extraction (PDF/DOCX/OCR) + local heuristics
├── db.py                      # Supabase client + CRUD (candidates, jobs, interviews)
├── reports.py                  # CSV / Excel / PDF report generation
├── local_settings.py           # Local app settings/helpers
├── requirements.txt            # Python dependencies
└── .streamlit/
    └── secrets.toml            # API keys & Supabase credentials (never commit)
```

---

## 👤 Author

**Shafi (Irfan Shafi)**
2nd-year AI & Machine Learning student, Kongu Engineering College (Perundurai, Erode) — Batch 2023–2027

- GitHub: [irfanshafi21](https://github.com/irfanshafi21)
- Portfolio: [irfanshafi21.github.io/portfolio](https://irfanshafi21.github.io/portfolio)
- Email: irfanshafi210608@gmail.com

Open to internships and ML/AI collaborations.

---

## 🧪 Testing Checklist (Manual QA)

Since this is a solo-built portfolio project without an automated CI pipeline, testing is manual. This checklist is what's actually run through before every deploy:

- [ ] Upload a single well-formatted PDF resume → confirm parsed fields (name/email/phone/skills) are correct
- [ ] Upload a DOCX resume that uses tables for layout → confirm table-cell text is still captured
- [ ] Upload a scanned/photographed image resume → confirm OCR extracts usable text
- [ ] Upload a ZIP containing a mix of PDF/DOCX/images → confirm all are unpacked and processed
- [ ] Upload a non-resume file (e.g. a certificate or cover letter) → confirm it's filtered out, with a reason shown
- [ ] Upload an unusually formatted *real* resume that fails the fast heuristic → confirm the AI second-pass check correctly keeps it
- [ ] Temporarily remove `GROQ_API_KEY` → confirm the app automatically falls back to Gemini without user-visible errors
- [ ] Temporarily remove both keys → confirm the sidebar clearly shows "no AI provider configured" instead of crashing
- [ ] Disconnect Supabase (or leave `SUPABASE_URL`/`SUPABASE_KEY` unset) → confirm the app still runs fully session-local
- [ ] Create a Job, screen candidates against it, then edit the Job → confirm existing screening records aren't corrupted
- [ ] Export CSV, Excel, and PDF for the same candidate set → confirm figures match what's shown on the Dashboard
- [ ] Toggle light/dark mode → confirm all custom CSS overrides still render legibly
- [ ] Generate interview questions for a low-scoring candidate → confirm "Gap Probing" questions reference their actual gaps, not generic filler
- [ ] Ask the AI Assistant a question about a candidate that hasn't been screened yet → confirm it says so instead of inventing an answer

---

## 🧩 Design Decisions & Trade-offs

A few choices in this project were deliberate trade-offs rather than "the only way to do it":

**Why Streamlit instead of a separate frontend/backend?**
For a solo project meant to be demo-able quickly, Streamlit's single-process model removes the need for a REST API layer, a separate JS frontend, and CORS/auth handshaking between them. The trade-off is less UI flexibility than a custom frontend would offer, and Streamlit's rerun-on-interaction model requires careful session-state management.

**Why combine parsing and scoring into one LLM call?**
Two separate calls would be architecturally cleaner, but doubles latency and token cost per resume. Given this app's primary use case is bulk screening, the combined `parse_and_score()` call was judged the better trade-off — a small increase in prompt complexity in exchange for roughly halved per-candidate latency.

**Why a two-pass (heuristic + AI) resume detector instead of AI-only?**
An AI-only check would work, but costs an API call per file even for the obvious majority of cases. Running the free local heuristic first and only escalating ambiguous cases to the AI keeps API usage down without sacrificing accuracy on edge cases.

**Why is ATS scoring split between AI and local heuristics, instead of fully AI?**
Structural checks (section presence, bullet ratios, sentence-length math) are deterministic — an LLM call would add latency and cost for something regular Python already computes exactly. Reserving AI for the parts that actually need language understanding keeps the app faster and cheaper without accuracy loss.

**Why Supabase instead of a self-hosted database?**
The project targets a free, zero-maintenance deployment path suitable for a student portfolio project. Supabase's free tier and instant REST API meant no server to provision, at the cost of vendor lock-in and the anon-key trade-off documented in [Security Notes](#-security-notes).

---

## ❓ Frequently Asked Questions

**Does this replace a human recruiter's decision?**
No. Every score, gap, and summary is meant as a *starting point* for a human to review faster — not an automated hire/reject decision.

**Can I use this without any API keys?**
No — at least one of `GROQ_API_KEY` or `GEMINI_API_KEY` is required, since every core feature depends on an LLM call. The app will load without one, but screening will fail with a clear message.

**Can I use this without Supabase?**
Yes — Supabase is entirely optional. Without it, the app works fully within a single browser session; data just doesn't persist once the tab is closed.

**Is my resume data sent anywhere permanently?**
Resume text is sent to whichever LLM provider is active for parsing/scoring, and — if configured — a structured summary is stored in your own Supabase project. No data is sent to any third party beyond the AI provider and your own database.

**Why did my resume get filtered out as "not a resume"?**
Check the reason shown in the UI — most commonly very short documents, resumes that are almost entirely images with little extractable text, or files that genuinely aren't resumes.

**What happens if I hit a free-tier rate limit mid-screening?**
`ai_engine.py` retries automatically with backoff, and falls back from Groq to Gemini if configured. If both are exhausted, the affected candidate(s) show a clear error rather than a corrupted result.

---

## 🤝 Contributing

This started as a solo portfolio project, but suggestions and pull requests are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Keep changes scoped — one feature/fix per PR is easier to review
4. Follow the existing module boundaries (UI logic stays in `app.py`; AI calls stay in `ai_engine.py`; parsing stays in `resume_parser.py`; persistence stays in `db.py`; exports stay in `reports.py`)
5. Test manually against the [Testing Checklist](#-testing-checklist-manual-qa) above before opening a PR
6. Open a pull request describing what changed and why

Bug reports and feature requests are welcome via GitHub Issues — please include steps to reproduce, the file type/format involved, and whether Groq or Gemini was active at the time.

---

## 📜 License

This project is shared for portfolio and educational purposes. If you'd like to reuse, extend, or build on it, please provide attribution back to the original author. Reach out via the contact details in the [Author](#-author) section for anything beyond that (commercial use, redistribution, etc.).

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io) — for making a full data app achievable without separate frontend/backend code
- [Groq](https://groq.com) — for extremely fast, generous free-tier LLM inference
- [Google AI Studio / Gemini](https://aistudio.google.com) — for a reliable fallback provider
- [Supabase](https://supabase.com) — for a zero-maintenance free Postgres backend
- The open-source maintainers of `pypdf`, `python-docx`, `pytesseract`, `Plotly`, `pandas`, `openpyxl`, and `reportlab`
