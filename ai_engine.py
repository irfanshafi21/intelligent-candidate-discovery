"""
AI engine — wraps LLM API calls for:
  1. Structured resume parsing
  2. Candidate scoring against a job description
  3. Interview question generation

Provider strategy: Groq and Cerebras are both fast, free (~30 req/min each),
so calls are randomly split between whichever of the two are configured —
roughly doubling effective throughput and combining their daily quotas.
Gemini is the final fallback if both fail or aren't configured.

Keys required (at least one):
  GROQ_API_KEY     — free at https://console.groq.com
  CEREBRAS_API_KEY — free at https://cloud.cerebras.ai
  GEMINI_API_KEY   — free at https://aistudio.google.com
Set them in st.secrets (.streamlit/secrets.toml) or as environment variables.
"""

import os
import json
import time
import re
import random
import streamlit as st

# ----------------------------- API KEY HELPERS -----------------------------

def _get_secret(name: str) -> str | None:
    # A session-level override set on the Settings page takes priority —
    # lets a person test/swap a key for this session without touching files.
    try:
        override = st.session_state.get(f"override_{name}")
        if override:
            return override
    except Exception:
        pass
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def _get_groq_key() -> str | None:
    return _get_secret("GROQ_API_KEY")


def _get_cerebras_key() -> str | None:
    return _get_secret("CEREBRAS_API_KEY")


def _get_gemini_key() -> str | None:
    return _get_secret("GEMINI_API_KEY")


def check_api_key() -> bool:
    """True if at least one provider is configured."""
    return bool(_get_groq_key() or _get_cerebras_key() or _get_gemini_key())


def active_provider() -> str:
    """Which provider(s) will be used, for display in the UI."""
    fast = [n for n, k in (("Groq", _get_groq_key()), ("Cerebras", _get_cerebras_key())) if k]
    if fast:
        label = " + ".join(fast) + (" (split)" if len(fast) > 1 else " (primary)")
        return label + (" + Gemini fallback" if _get_gemini_key() else "")
    elif _get_gemini_key():
        return "Gemini (no Groq/Cerebras key set)"
    return "None configured"


# ----------------------------- JSON CLEANUP -----------------------------

def _sanitize_json_text(text: str) -> str:
    """Clean up common issues in model-returned JSON before parsing."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    # Strip raw control characters (unescaped newlines/tabs inside strings break json.loads)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _extract_retry_seconds(msg: str, attempt: int) -> int:
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg) or re.search(r"try again in ([\d.]+)s", msg, re.I)
    if match:
        return int(float(match.group(1))) + 2
    return (attempt + 1) * 15


# ----------------------------- GROQ PROVIDER -----------------------------

GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

# Chat replies prioritize speed over the extra reasoning depth scoring needs —
# llama-3.1-8b-instant on Groq's LPU hardware typically replies in ~1-2s.
# Falls back to the bigger models only if the instant one is unavailable.
GROQ_CHAT_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]


def _call_groq(prompt: str, max_attempts: int = 3) -> dict:
    from groq import Groq

    api_key = _get_groq_key()
    if not api_key:
        raise RuntimeError("Groq API key not configured.")
    client = Groq(api_key=api_key)

    last_error = None
    for model_name in GROQ_FALLBACK_MODELS:
        for attempt in range(max_attempts):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise JSON-generating assistant. Always return only valid JSON, no prose, no markdown fences."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                raw_text = completion.choices[0].message.content
                cleaned = _sanitize_json_text(raw_text)
                return json.loads(cleaned)
            except json.JSONDecodeError as parse_err:
                last_error = parse_err
                if attempt < max_attempts - 1:
                    continue
                break  # try next model
            except Exception as e:
                msg = str(e)
                last_error = e
                if "429" in msg or "rate limit" in msg.lower():
                    if attempt < max_attempts - 1:
                        time.sleep(_extract_retry_seconds(msg, attempt))
                        continue
                    break  # exhausted retries on this model, try next model
                if "404" in msg or "decommissioned" in msg.lower() or "not found" in msg.lower():
                    break  # model retired, try next model in the list
                raise  # unrelated error (e.g. bad key) — surface immediately
    raise last_error or RuntimeError("Groq: all models failed")


# ----------------------------- CEREBRAS PROVIDER -----------------------------
# Cerebras runs open-weight models on its wafer-scale chips — similar speed
# and RPM to Groq (30 req/min free), but a much larger daily token budget.
# Splitting load between the two roughly doubles effective throughput.

CEREBRAS_FALLBACK_MODELS = ["llama-3.3-70b", "llama3.1-8b", "qwen-3-32b", "gpt-oss-120b"]
CEREBRAS_CHAT_MODELS = ["llama3.1-8b", "llama-3.3-70b", "qwen-3-32b"]


def _call_cerebras(prompt: str, max_attempts: int = 3) -> dict:
    from cerebras.cloud.sdk import Cerebras

    api_key = _get_cerebras_key()
    if not api_key:
        raise RuntimeError("Cerebras API key not configured.")
    client = Cerebras(api_key=api_key)

    last_error = None
    for model_name in CEREBRAS_FALLBACK_MODELS:
        for attempt in range(max_attempts):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise JSON-generating assistant. Always return only valid JSON, no prose, no markdown fences."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                raw_text = completion.choices[0].message.content
                cleaned = _sanitize_json_text(raw_text)
                return json.loads(cleaned)
            except json.JSONDecodeError as parse_err:
                last_error = parse_err
                if attempt < max_attempts - 1:
                    continue
                break  # try next model
            except Exception as e:
                msg = str(e)
                last_error = e
                if "429" in msg or "rate limit" in msg.lower():
                    if attempt < max_attempts - 1:
                        time.sleep(_extract_retry_seconds(msg, attempt))
                        continue
                    break
                if "404" in msg or "not found" in msg.lower() or "does not exist" in msg.lower():
                    break  # model unavailable, try next model in the list
                raise
    raise last_error or RuntimeError("Cerebras: all models failed")


def _call_cerebras_text(prompt: str, max_attempts: int = 3) -> str:
    from cerebras.cloud.sdk import Cerebras

    api_key = _get_cerebras_key()
    client = Cerebras(api_key=api_key)
    last_error = None
    for model_name in CEREBRAS_CHAT_MODELS:
        for attempt in range(max_attempts):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise, grounded recruiting assistant. Never fabricate information not present in the data you're given. Keep replies concise — a few sentences or short bullet points, not long essays."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e)
                last_error = e
                if "429" in msg or "rate limit" in msg.lower():
                    if attempt < max_attempts - 1:
                        time.sleep(_extract_retry_seconds(msg, attempt))
                        continue
                    break
                if "404" in msg or "not found" in msg.lower() or "does not exist" in msg.lower():
                    break
                raise
    raise last_error or RuntimeError("Cerebras: all models failed")


# ----------------------------- GEMINI PROVIDER (fallback) -----------------------------

GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3-flash", "gemini-2.5-flash", "gemini-flash-latest"]


def _call_gemini(prompt: str, max_attempts: int = 3) -> dict:
    import google.generativeai as genai

    api_key = _get_gemini_key()
    if not api_key:
        raise RuntimeError("Gemini API key not configured.")
    genai.configure(api_key=api_key)

    last_error = None
    for model_name in GEMINI_FALLBACK_MODELS:
        model = genai.GenerativeModel(model_name)
        for attempt in range(max_attempts):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                cleaned = _sanitize_json_text(response.text)
                return json.loads(cleaned)
            except json.JSONDecodeError as parse_err:
                last_error = parse_err
                if attempt < max_attempts - 1:
                    continue
                break
            except Exception as e:
                msg = str(e)
                last_error = e
                if "429" in msg:
                    if attempt < max_attempts - 1:
                        time.sleep(_extract_retry_seconds(msg, attempt))
                        continue
                    break
                if "404" in msg or "no longer available" in msg.lower() or "not found" in msg.lower():
                    break
                raise
    raise last_error or RuntimeError("Gemini: all models failed")


# ----------------------------- UNIFIED ENTRY POINT -----------------------------

def _call_json(prompt: str) -> dict:
    """
    Randomly pick between Groq and Cerebras for each call (whichever are
    configured) — both are fast and free at ~30 req/min, so alternating
    roughly doubles effective throughput and combines their daily quotas.
    Falls back to Gemini only if the fast provider(s) both fail.
    """
    fast_providers = []
    if _get_groq_key():
        fast_providers.append(("Groq", _call_groq))
    if _get_cerebras_key():
        fast_providers.append(("Cerebras", _call_cerebras))
    gemini_key = _get_gemini_key()

    if fast_providers:
        random.shuffle(fast_providers)
        errors = {}
        for name, fn in fast_providers:
            try:
                return fn(prompt)
            except Exception as e:
                errors[name] = e
        if gemini_key:
            try:
                return _call_gemini(prompt)
            except Exception as gemini_error:
                errors["Gemini"] = gemini_error
        raise RuntimeError(f"All configured providers failed: {errors}")

    if gemini_key:
        return _call_gemini(prompt)

    raise RuntimeError("No API key configured. Add GROQ_API_KEY, CEREBRAS_API_KEY, and/or GEMINI_API_KEY.")


# ----------------------------- PROMPTS -----------------------------

def parse_resume_with_ai(raw_text: str) -> dict:
    """
    Extract structured candidate profile fields from raw resume text.
    (Kept standalone for cases where only parsing is needed.)
    """
    prompt = f"""You are a resume parsing engine. Extract structured information from the resume text below.

Return ONLY valid JSON with this exact schema:
{{
  "name": "candidate full name",
  "email": "email or empty string",
  "phone": "phone or empty string",
  "years_experience": "e.g. '3 years' or 'Fresher/Entry-level'",
  "education": "highest degree + institution, one line",
  "skills": ["skill1", "skill2", ...],
  "past_roles": ["role at company", ...],
  "summary": "2-sentence neutral summary of the candidate's background"
}}

Resume text:
---
{raw_text[:8000]}
---
"""
    return _call_json(prompt)


def parse_and_score(raw_text: str, job_description: str) -> tuple[dict, dict]:
    """
    Parse the resume AND score it against the job description in a single
    API call — same reasoning depth as calling parse + score separately,
    just done in one pass instead of two, which roughly halves latency
    per candidate.
    """
    prompt = f"""You are an expert technical recruiter and resume parser. Do TWO things in one pass:

STEP 1 — Extract structured information from the resume.
STEP 2 — Evaluate how well this candidate fits the job description below. Be objective and
evidence-based; judge substance over keywords alone; don't penalize non-traditional resume formats.

JOB DESCRIPTION:
---
{job_description[:4000]}
---

RESUME TEXT:
---
{raw_text[:8000]}
---

Return ONLY valid JSON with this exact schema:
{{
  "profile": {{
    "name": "candidate full name",
    "email": "email or empty string",
    "phone": "phone or empty string",
    "years_experience": "e.g. '3 years' or 'Fresher/Entry-level'",
    "education": "highest degree + institution, one line",
    "skills": ["skill1", "skill2", ...],
    "past_roles": ["role at company", ...],
    "summary": "2-sentence neutral summary of the candidate's background"
  }},
  "score": {{
    "overall_score": <integer 0-100>,
    "breakdown": {{
      "skills_match": <integer 0-100>,
      "experience_fit": <integer 0-100>,
      "education_fit": <integer 0-100>
    }},
    "matched_skills": ["skill that matches JD requirements", ...],
    "gaps": ["missing or weak area relative to JD", ...],
    "summary": "2-3 sentence recruiter-style verdict on this candidate's fit"
  }}
}}
"""
    result = _call_json(prompt)
    return result.get("profile", {}), result.get("score", {})


def score_candidate(profile: dict, raw_text: str, job_description: str) -> dict:
    """
    Score a parsed candidate profile against a job description.
    (Kept standalone for re-scoring an already-parsed candidate against a
    different job description, without re-parsing the resume.)
    """
    prompt = f"""You are an expert technical recruiter. Evaluate how well this candidate fits the job description.
Be objective, evidence-based, and avoid penalizing non-traditional resume formats. Judge substance over keywords alone.

JOB DESCRIPTION:
---
{job_description[:4000]}
---

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

CANDIDATE RESUME (raw, for extra context):
---
{raw_text[:6000]}
---

Return ONLY valid JSON with this exact schema:
{{
  "overall_score": <integer 0-100>,
  "breakdown": {{
    "skills_match": <integer 0-100>,
    "experience_fit": <integer 0-100>,
    "education_fit": <integer 0-100>
  }},
  "matched_skills": ["skill that matches JD requirements", ...],
  "gaps": ["missing or weak area relative to JD", ...],
  "summary": "2-3 sentence recruiter-style verdict on this candidate's fit"
}}
"""
    return _call_json(prompt)


def analyze_ats_ai(raw_text: str, job_description: str) -> dict:
    """
    AI-judged portion of ATS analysis: keyword coverage, grammar quality,
    missing keywords, concrete improvement suggestions, and an overall
    ATS-compatibility verdict. Formatting/Structure/Readability are computed
    separately via free local heuristics (see resume_parser.compute_local_ats_metrics)
    since those are algorithmic, not judgment calls — this call only covers
    the parts that genuinely need language understanding.
    """
    prompt = f"""You are an ATS (Applicant Tracking System) resume analyzer. Evaluate this resume's
text against the job description for keyword coverage and writing quality — the kind of
analysis a tool like Jobscan or an ATS parser would surface to a candidate.

JOB DESCRIPTION:
---
{job_description[:3000]}
---

RESUME TEXT:
---
{raw_text[:6000]}
---

Return ONLY valid JSON with this exact schema:
{{
  "keyword_score": <integer 0-100, how well the resume's terminology covers the JD's key terms>,
  "grammar_score": <integer 0-100, writing quality: clarity, tense consistency, no typos>,
  "missing_keywords": ["important JD term/skill absent from the resume", ...],
  "suggestions": ["concrete, specific improvement suggestion", ...],
  "ats_compatibility": "High" or "Medium" or "Low",
  "compatibility_reason": "one sentence explaining the compatibility verdict"
}}
"""
    return _call_json(prompt)


def is_resume_ai_check(text: str) -> dict:
    """
    Second-pass, AI-based check for whether a document is actually a resume/CV.
    Only called for files that failed the fast local heuristic check — this
    is the "check twice before excluding" safety net, so an unusually
    formatted real resume doesn't get silently dropped.
    """
    prompt = f"""Determine whether the text below is a resume/CV (a document describing a
person's work experience, education, and skills for a job application) or something else
entirely (e.g. a cover letter, certificate, random document, ID card, receipt, unrelated image).

Return ONLY valid JSON with this schema:
{{
  "is_resume": true or false,
  "reason": "one short sentence explaining why"
}}

TEXT (may be partial or OCR'd, so tolerate minor noise):
---
{text[:3000]}
---
"""
    return _call_json(prompt)


def generate_interview_questions(profile: dict, score_data: dict, job_description: str) -> dict:
    """
    Generate role- and candidate-specific interview questions, grouped by
    section, each paired with concrete guidance on what a strong answer
    should cover — so the recruiter has a concrete rubric to evaluate
    against rather than just a bare question.
    """
    prompt = f"""You are preparing an interview guide for a recruiter. Based on the candidate's profile,
their identified gaps, and the job description, generate targeted interview questions.

For EACH question, also provide "what_good_looks_like": concrete, specific points a strong
answer should cover, grounded in the job description and this candidate's actual background
(not generic advice). This is what the recruiter will use to judge the candidate's real answer,
so be specific and evidence-based rather than vague.

JOB DESCRIPTION:
---
{job_description[:3000]}
---

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

IDENTIFIED GAPS:
{json.dumps(score_data.get('gaps', []))}

Return ONLY valid JSON with this exact schema (3-4 items per section):
{{
  "Technical Validation": [
    {{"question": "...", "what_good_looks_like": "specific points a strong answer covers"}}
  ],
  "Experience Deep-Dive": [
    {{"question": "...", "what_good_looks_like": "..."}}
  ],
  "Gap Probing": [
    {{"question": "question targeting an identified gap", "what_good_looks_like": "..."}}
  ],
  "Culture & Motivation": [
    {{"question": "...", "what_good_looks_like": "..."}}
  ]
}}
"""
    return _call_json(prompt)


APP_KNOWLEDGE = """
This is the Intelligent Candidate Discovery Platform — an AI-powered resume screening and
candidate ranking app. Its features:

- Resume Screening: enter a job title + key requirements, set priority weighting (how much
  Skills/Experience/Education matter for ranking), set "Top N" (how many candidates to shortlist),
  then upload resumes as individual files (PDF/DOCX/JPG/PNG, image resumes are read via OCR) or as
  a ZIP folder. Non-resume files (invoices, certificates, random documents) are auto-detected and
  excluded via a two-pass check (a fast local keyword/contact-info check, then an AI confirmation
  before final exclusion) — nothing gets dropped on a single weak signal.
- Home: the command center and landing page. Shows live recruitment stats (resumes uploaded/analyzed,
  shortlisted/selected/rejected counts, interviews pending/completed, candidates awaiting an interview
  decision, active job openings, completed recruitments, average ATS score, hiring success rate), a
  recent activity timeline, and recruitment charts (hiring funnel, score distribution, top skills,
  education mix).
- Resume Screening: also shows a "Top Candidates" section (ranked by priority-weighted score) with
  Select/Reject actions, once resumes have been screened.
- Shortlisted: lists candidates with an ATS score ≥ 70 — and, once they've been interviewed, their
  interview score must also be ≥ 70 to remain on this list.
- Interview: combines interview prep and tracking in one page. The "Prep & Schedule" tab generates
  candidate- and role-specific interview questions in four categories (Technical Validation, Experience
  Deep-Dive, Gap Probing, Culture & Motivation), each with "what a strong answer covers" guidance, and
  lets the recruiter schedule an interview. The candidate picker shows each candidate's ATS score next
  to their name. The "Track Interviews" tab shows scheduled/completed/cancelled interviews; once an
  interview is completed, the recruiter enters an interview score out of 100, which saves immediately
  and also feeds the Shortlisted page's pass/fail check.
- Reports: export candidate, shortlist, and interview data as PDF, Excel, or CSV.
- AI Insights (this chat): answers questions about screened candidates (grounded strictly in
  their actual screening data), explains how to use the app, and can suggest ideas — e.g. how to
  word a job description, what priority weighting suits a given role, what to watch for when
  screening a particular type of role.
"""


def ask_assistant(question: str, candidates: list, job_role: str, job_details: str, chat_history: list) -> str:
    """
    General-purpose assistant for this app. Works in two modes depending on
    whether candidates have been screened yet:

    - With screened candidates: answers are grounded strictly in the actual
      structured data already computed (scores, matched skills, gaps,
      summaries) — never invents facts about a candidate.
    - Without any screened candidates yet (or for questions unrelated to any
      specific candidate): still fully functional — explains how the app
      works, and gives practical recruiting/job-description/weighting ideas.
      It's honest about which mode it's answering in, rather than pretending
      to know things about candidates that don't exist yet.
    """
    candidate_summaries = []
    for c in candidates:
        if c["score"].get("error"):
            continue
        p, s = c["profile"], c["score"]
        candidate_summaries.append({
            "name": c["name"],
            "years_experience": p.get("years_experience"),
            "education": p.get("education"),
            "skills": p.get("skills", []),
            "past_roles": p.get("past_roles", []),
            "overall_score": s.get("overall_score"),
            "breakdown": s.get("breakdown", {}),
            "matched_skills": s.get("matched_skills", []),
            "gaps": s.get("gaps", []),
            "recruiter_summary": s.get("summary"),
        })

    history_text = ""
    for turn in chat_history[-6:]:  # last few turns for context, keep prompt small
        history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    if candidate_summaries:
        data_block = f"""CANDIDATE DATA (the only source of truth for candidate-specific questions — {len(candidate_summaries)} candidate(s) screened this session):
{json.dumps(candidate_summaries, separators=(',', ':'))[:4500]}

JOB ROLE BEING SCREENED FOR: {job_role or '(not set)'}
JOB REQUIREMENTS: {job_details or '(not set)'}"""
    else:
        data_block = (
            "No candidates have been screened yet this session (the CANDIDATE DATA is empty). "
            "If the recruiter asks about a specific candidate, say clearly that nothing has been "
            "screened yet and suggest they use the Resume Screening page first. You can still fully "
            "answer questions about how the app works, or give general recruiting/job-description/"
            "interview ideas."
        )

    prompt = f"""You are the built-in AI Insights assistant for the Intelligent Candidate Discovery Platform.
You help the recruiter in two ways: (1) answering questions about candidates they've screened,
strictly grounded in real data, and (2) answering questions about the app itself, plus offering
practical, concrete ideas (job description wording, priority-weighting suggestions, interview
strategy, what to watch for in a given role) — this second mode does not require any screened data.

APP KNOWLEDGE (use this to answer "how does this work" / "what can this app do" questions):
{APP_KNOWLEDGE}

Rules:
- For candidate-specific questions: answer ONLY using the CANDIDATE DATA below. Never invent
  facts, scores, or skills that aren't present in it. If something isn't covered by the data
  (a skill never mentioned, salary expectations, availability, a candidate not in the list),
  say clearly that this wasn't captured during screening — do not guess or fabricate.
- For app-usage questions: answer from the APP KNOWLEDGE above.
- For open-ended requests ("give me ideas for...", "how should I word...", "what should I ask
  a candidate for X role"): give practical, specific, well-reasoned suggestions — this is
  general recruiting expertise, not something that needs to be "looked up," so answer confidently.
- Keep answers concise, concrete, and recruiter-friendly. Use bullet points for comparisons or lists.

{data_block}

RECENT CONVERSATION:
{history_text}

RECRUITER'S QUESTION:
{question}

Answer now."""

    # This is a free-text answer, not JSON — use a lighter-weight direct call.
    fast_providers = []
    if _get_groq_key():
        fast_providers.append(("Groq", _call_groq_text))
    if _get_cerebras_key():
        fast_providers.append(("Cerebras", _call_cerebras_text))
    gemini_key = _get_gemini_key()

    if fast_providers:
        random.shuffle(fast_providers)
        last_error = None
        for name, fn in fast_providers:
            try:
                return fn(prompt)
            except Exception as e:
                last_error = e
        if gemini_key:
            return _call_gemini_text(prompt)
        raise last_error
    elif gemini_key:
        return _call_gemini_text(prompt)
    raise RuntimeError("No API key configured.")


def _call_groq_text(prompt: str, max_attempts: int = 3) -> str:
    from groq import Groq
    api_key = _get_groq_key()
    client = Groq(api_key=api_key)
    last_error = None
    for model_name in GROQ_CHAT_MODELS:
        for attempt in range(max_attempts):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise, grounded recruiting assistant. Never fabricate information not present in the data you're given. Keep replies concise — a few sentences or short bullet points, not long essays."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500,  # keeps replies snappy; concise answers don't need more
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e)
                last_error = e
                if "429" in msg or "rate limit" in msg.lower():
                    if attempt < max_attempts - 1:
                        time.sleep(_extract_retry_seconds(msg, attempt))
                        continue
                    break
                if "404" in msg or "decommissioned" in msg.lower() or "not found" in msg.lower():
                    break
                raise
    raise last_error or RuntimeError("Groq: all models failed")


def _call_gemini_text(prompt: str, max_attempts: int = 3) -> str:
    import google.generativeai as genai
    api_key = _get_gemini_key()
    genai.configure(api_key=api_key)
    last_error = None
    for model_name in GEMINI_FALLBACK_MODELS:
        model = genai.GenerativeModel(model_name)
        for attempt in range(max_attempts):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.1),
                )
                return response.text.strip()
            except Exception as e:
                msg = str(e)
                last_error = e
                if "429" in msg:
                    if attempt < max_attempts - 1:
                        time.sleep(_extract_retry_seconds(msg, attempt))
                        continue
                    break
                if "404" in msg or "no longer available" in msg.lower() or "not found" in msg.lower():
                    break
                raise
    raise last_error or RuntimeError("Gemini: all models failed")
