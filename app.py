"""
Intelligent Candidate Discovery Platform
AI-Powered Resume Screening and Candidate Ranking

Built with Streamlit + Google Gemini API
"""

import streamlit as st
import logging

# The screening loop runs each candidate on a worker thread via
# ThreadPoolExecutor. Streamlit logs a "missing ScriptRunContext" warning for
# every such thread — it's expected and harmless (only the main thread needs
# that context), but it clutters the terminal, so it's silenced here.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
import json
import time
import random
import base64
import os
from datetime import datetime
from contextlib import contextmanager

from concurrent.futures import ThreadPoolExecutor, as_completed
from resume_parser import (
    extract_text_from_file,
    extract_text_from_bytes,
    extract_files_from_zip,
    heuristic_resume_check,
    compute_local_ats_metrics,
    assess_extraction_confidence,
)
from ai_engine import (
    parse_and_score,
    generate_interview_questions,
    check_api_key,
    active_provider,
    is_resume_ai_check,
    ask_assistant,
    analyze_ats_ai,
)
import db
import auth
import reports
import email_utils

# Google Fonts family names for browser-side signature style previews — keys
# must match reports.SIGNATURE_STYLES so the preview matches what the PDF renders.
SIGNATURE_FONT_FAMILIES = {
    "Elegant Script": "Great Vibes",
    "Flowing Cursive": "Alex Brush",
    "Casual Handwriting": "Sacramento",
    "Rounded Script": "Dancing Script",
    "Bold Brush": "Pacifico",
}
import local_settings
import components

STOPWORDS = {
    "a","an","the","and","or","for","with","in","on","of","to","is","are","be",
    "we","you","our","this","that","as","at","by","from","have","has","will",
    "who","looking","seeking","required","requirements","requirement","preferred",
    "plus","bonus","strong","good","excellent","years","year","experience",
    "role","job","candidate","candidates","should","must","need","needs",
    "including","etc","using","use","work","working","team","ability","skills",
    "skill","knowledge","familiarity","proficient","proficiency",
}


def extract_keywords(text: str, max_keywords: int = 14) -> list[str]:
    """
    Lightweight local keyword extraction from job text — no API call needed.
    Pulls out meaningful technical/role terms, preserving things like
    'ai/ml', 'c++', 'ci/cd' that a naive word-split would otherwise mangle.
    """
    import re as _re
    if not text or not text.strip():
        return []
    tokens = _re.findall(r"[a-zA-Z][a-zA-Z0-9+/#.\-]{1,30}", text.lower())
    seen = []
    for tok in tokens:
        tok = tok.strip(".-")
        if not tok or tok in STOPWORDS or len(tok) < 2:
            continue
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= max_keywords:
            break
    return seen


# ----------------------------- JOBS storage (Supabase-backed, session-local fallback) -----------------------------

def get_jobs(include_archived: bool = True) -> list[dict]:
    """Merge remote (Supabase) jobs with any session-local ones that fell
    back due to a save failure — this way a job never silently disappears
    just because is_configured() is True but the actual insert failed."""
    remote = db.fetch_jobs(include_archived=include_archived) if db.is_configured() else []
    local = st.session_state.local_jobs
    if not include_archived:
        local = [j for j in local if j.get("status") == "active"]
    all_jobs = remote + local
    return sorted(all_jobs, key=lambda j: j.get("created_at", ""), reverse=True)


def create_job(job: dict) -> tuple[dict, bool]:
    """Returns (row, saved_to_supabase). If Supabase is configured but the
    save actually fails, the real error is available via db.get_last_error()
    so it can be shown to the user instead of failing silently."""
    if db.is_configured():
        saved = db.save_job(job)
        if saved:
            return saved, True
    new_id = max([j["id"] for j in st.session_state.local_jobs], default=0) + 1
    row = {**job, "id": new_id, "created_at": datetime.now().isoformat(), "status": job.get("status", "active")}
    st.session_state.local_jobs.append(row)
    return row, False


def update_job_record(job_id, updates: dict) -> None:
    if db.is_configured():
        db.update_job(job_id, updates)
    for j in st.session_state.local_jobs:
        if j["id"] == job_id:
            j.update(updates)


def remove_job(job_id) -> None:
    if db.is_configured():
        db.delete_job(job_id)
    st.session_state.local_jobs = [j for j in st.session_state.local_jobs if j["id"] != job_id]


# ----------------------------- CANDIDATES storage (Supabase-backed, session-local fallback) -----------------------------
# Same pattern as jobs/interviews above: Supabase is the source of truth so
# candidates, decisions, bookmarks, notes, and interview scores all survive a
# closed browser / restarted app. local_candidates is only a fallback for the
# rare case a save fails while Supabase is otherwise configured.

def _row_to_candidate(row: dict) -> dict:
    """Normalize a raw Supabase screening_history row into the candidate dict
    shape the rest of the app already expects."""
    return {
        "id": row.get("id"),
        "filename": row.get("filename") or row.get("candidate_name") or "unknown",
        "name": row.get("candidate_name") or row.get("filename") or "Unknown",
        "raw_text": row.get("raw_text", ""),
        "profile": row.get("profile_json") or {},
        "score": row.get("score_json") or {"overall_score": row.get("overall_score", 0)},
        "screened_at": row.get("screened_at"),
        "job_id": row.get("job_id"),
        "job_role": row.get("job_role") or "",
        "status": row.get("decision_status", "Waiting"),
        "interview_score": row.get("interview_score"),
        "bookmarked": bool(row.get("bookmarked", False)),
        "notes": row.get("notes", ""),
        "questions": row.get("questions_json") or {},
    }


def get_candidates() -> list[dict]:
    """All persisted, non-cleared candidates, merged from Supabase and any
    session-local fallback rows that failed to save remotely."""
    remote = [_row_to_candidate(r) for r in db.fetch_screening_history()] if db.is_configured() else []
    local = st.session_state.local_candidates
    all_c = remote + local
    return sorted(all_c, key=lambda c: c.get("screened_at") or "", reverse=True)


def add_candidate_record(candidate: dict, job_role: str, job_details: str, job_id=None) -> dict:
    """Persist a freshly-screened candidate. Returns the normalized candidate
    dict (with a stable 'id') regardless of whether Supabase or the local
    fallback ended up storing it, so it can be referenced later."""
    if db.is_configured():
        saved = db.save_screening_record(candidate, job_role, job_details, job_id=job_id)
        if saved:
            return _row_to_candidate(saved)
    existing_ids = [int(str(c["id"]).replace("local-", "")) for c in st.session_state.local_candidates]
    new_id = f"local-{max(existing_ids, default=0) + 1}"
    row = {**candidate, "id": new_id, "job_id": job_id, "job_role": job_role, "bookmarked": False, "notes": candidate.get("notes", "")}
    st.session_state.local_candidates.append(row)
    return row


def update_candidate_record(candidate_id, updates: dict) -> None:
    """Write a candidate change (decision status, bookmark, notes, interview
    score, or a profile edit like fixing contact details) straight through
    to Supabase, keeping the local fallback list in sync too for candidates
    that only ever existed there."""
    db_updates = {}
    if "status" in updates:
        db_updates["decision_status"] = updates["status"]
    if "questions" in updates:
        db_updates["questions_json"] = json.dumps(updates["questions"])
    if "profile" in updates:
        db_updates["profile_json"] = json.dumps(updates["profile"])
        # Keep the dedicated email/phone columns in sync too, in case
        # anything queries them directly instead of profile_json.
        db_updates["email"] = updates["profile"].get("email")
        db_updates["phone"] = updates["profile"].get("phone")
    for k in ("bookmarked", "notes", "interview_score"):
        if k in updates:
            db_updates[k] = updates[k]
    if db.is_configured() and not str(candidate_id).startswith("local-"):
        db.update_screening_record(candidate_id, db_updates)
    for c in st.session_state.local_candidates:
        if c["id"] == candidate_id:
            c.update(updates)


def clear_all_candidates_record(job_id=None) -> None:
    """Soft-delete every active candidate (optionally scoped to one job) in
    Supabase, and drop the local fallback list — it's session-only anyway."""
    if db.is_configured():
        db.clear_screening_records(job_id=job_id)
    if job_id is None:
        st.session_state.local_candidates = []
    else:
        st.session_state.local_candidates = [c for c in st.session_state.local_candidates if c.get("job_id") != job_id]


# ----------------------------- INTERVIEW PREP (shared between Candidates profile & Interview page) -----------------------------

def render_interview_prep(candidate: dict, key_prefix: str) -> None:
    """
    Generates and displays AI interview questions for one candidate, persisted
    to Supabase so prep work survives closing the app. Each question doubles
    as a live checklist during the interview itself: mark it "Asked" and jot
    what the candidate actually said, right under the rubric for what a
    strong answer should cover — turning the sheet from a study guide into
    something usable while the interview is happening.
    """
    has_questions = bool(candidate.get("questions"))
    gen_label = "🔄 Regenerate Interview Questions" if has_questions else "✨ Generate Interview Questions"
    if st.button(gen_label, type="primary", key=f"{key_prefix}_gen_q"):
        with st.spinner("Generating tailored questions..."):
            try:
                raw_questions = generate_interview_questions(
                    candidate["profile"], candidate["score"],
                    f"Job Role: {st.session_state.job_role}\n\nKey Requirements:\n{st.session_state.job_details}"
                )
                # Seed each question with the live-checklist fields.
                for section, qs in raw_questions.items():
                    for item in qs:
                        if isinstance(item, dict):
                            item.setdefault("asked", False)
                            item.setdefault("candidate_answer", "")
                candidate["questions"] = raw_questions
                update_candidate_record(candidate["id"], {"questions": raw_questions})
                st.rerun()
            except Exception as e:
                st.error(f"Couldn't generate questions: {e}")

    if candidate.get("questions"):
        st.caption("Each question includes what a strong, evidence-based answer should cover. "
                   "Check off questions as you ask them and jot what the candidate actually said — "
                   "it's saved automatically for later review.")
        for section, qs in candidate["questions"].items():
            st.markdown(f"**{section}**")
            for i, item in enumerate(qs):
                if not isinstance(item, dict):
                    item = {"question": str(item), "what_good_looks_like": ""}
                q_text = item.get("question", "")
                guidance = item.get("what_good_looks_like", "")
                asked = bool(item.get("asked", False))
                answer = item.get("candidate_answer", "")
                qkey = f"{key_prefix}_{section}_{i}".replace(" ", "_")

                st.markdown(f"**Q:** {q_text}")
                if guidance:
                    st.markdown(f"""
                    <div style="background:#E3F5FD; border-left:3px solid #38BDF8; border-radius:8px; padding:8px 14px; margin:4px 0 10px; font-size:0.85rem; color:#12314A;">
                        <b>What a strong answer covers:</b> {guidance}
                    </div>
                    """, unsafe_allow_html=True)

                acol, ncol = st.columns([1, 4])
                with acol:
                    new_asked = st.checkbox("Asked", value=asked, key=f"asked_{qkey}")
                if new_asked != asked:
                    item["asked"] = new_asked
                    update_candidate_record(candidate["id"], {"questions": candidate["questions"]})
                    st.rerun()

                with ncol:
                    new_answer = st.text_input("What the candidate said", value=answer, key=f"answer_{qkey}", label_visibility="collapsed", placeholder="What the candidate said (optional)")
                if st.button("💾 Save note", key=f"save_ans_{qkey}"):
                    item["candidate_answer"] = new_answer
                    update_candidate_record(candidate["id"], {"questions": candidate["questions"]})
                    st.success("Saved.", icon="✅")
                st.markdown("---")


# ----------------------------- INTERVIEWS storage (Supabase-backed, session-local fallback) -----------------------------

def get_interviews() -> list[dict]:
    remote = db.fetch_interviews() if db.is_configured() else []
    all_interviews = remote + st.session_state.local_interviews
    return sorted(all_interviews, key=lambda i: i.get("scheduled_at", ""))


def create_interview(interview: dict) -> tuple[dict, bool]:
    if db.is_configured():
        saved = db.save_interview(interview)
        if saved:
            return saved, True
    new_id = max([i["id"] for i in st.session_state.local_interviews], default=0) + 1
    row = {**interview, "id": new_id}
    st.session_state.local_interviews.append(row)
    return row, False


def notify(message: str, icon: str = "✅"):
    """Success notification that respects the 'show success toasts' preference
    from Settings. Applied to the main lifecycle events (screening complete,
    job saved, interview scheduled) — not every minor confirmation in the app."""
    if st.session_state.get("local_prefs", {}).get("show_success_toasts", True):
        st.success(message, icon=icon)


def page_header(icon: str, title: str, subtitle: str = ""):
    """Consistent, styled header used at the top of every page.
    HTML is built as a single unindented line — Streamlit's markdown still
    runs content through a Markdown parser before injecting raw HTML, and
    lines indented 4+ spaces get treated as literal code blocks, which
    corrupted this exact header when subtitle was empty."""
    sub_html = f'<div class="page-header-sub">{subtitle}</div>' if subtitle else ""
    html = (
        '<div class="page-header">'
        '<div class="page-header-decoration"></div>'
        f'<div class="page-header-icon">{icon}</div>'
        '<div class="page-header-text">'
        f'<div class="page-header-title">{title}</div>'
        f'{sub_html}'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def update_interview_record(interview_id, updates: dict) -> bool:
    """Returns True if the change is actually persisted somewhere (Supabase
    or local fallback), False if a Supabase write failed — callers should
    check this and show db.get_last_error() rather than assuming success,
    since a failed write with no visible error is how a 'locked' score can
    silently never actually save."""
    ok = True
    if db.is_configured():
        ok = db.update_interview(interview_id, updates)
    for i in st.session_state.local_interviews:
        if i["id"] == interview_id:
            i.update(updates)
    return ok

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="Intelligent Candidate Discovery Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- SPLASH SCREEN (logo video, once per browser session) -----------------------------
@st.cache_data
def _load_splash_video_b64():
    video_path = os.path.join(os.path.dirname(__file__), "assets", "logo_video.mp4")
    try:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return None

if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    _splash_b64 = _load_splash_video_b64()
    if _splash_b64:
        st.markdown(f"""
        <style>
        #icd-splash-overlay {{
            position: fixed; inset: 0; z-index: 999999;
            background: #FFFFFF; display: flex; align-items: center; justify-content: center;
            animation: icdSplashFade 0.6s ease 4.9s forwards;
        }}
        #icd-splash-overlay video {{
            width: 100%; height: 100%; object-fit: cover;
        }}
        @keyframes icdSplashFade {{
            to {{ opacity: 0; visibility: hidden; pointer-events: none; }}
        }}
        </style>
        <div id="icd-splash-overlay">
            <video autoplay muted playsinline>
                <source src="data:video/mp4;base64,{_splash_b64}" type="video/mp4">
            </video>
        </div>
        """, unsafe_allow_html=True)
    st.session_state.splash_shown = True

# ----------------------------- STYLING -----------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Pages with st.chat_input (e.g. AI Insights) can otherwise get vertically
       centered by Streamlit's flex layout when content is short, pushing
       everything down and creating a gap above the header. Force top-anchored
       instead — chat_input still docks to the bottom on its own regardless. */
    [data-testid="stMainBlockContainer"], [data-testid="stMain"], .main {
        justify-content: flex-start !important;
    }

    :root {
        /* Core palette (accurate ICD Platform Design System tokens) */
        --primary: #00506B;
        --primary-dark: #00394D;
        --secondary: #E0F2FE;
        --secondary-container: #39B8FD;
        --accent: #38BDF8;
        --bg: #FAFBFC;
        --card: #FFFFFF;
        --border: #BDC8D1;
        --text: #0B1C30;
        --text-secondary: #3E484F;
        --success: #22C55E;
        --warning: #F59E0B;
        --danger: #BA1A1A;
        --danger-container: #FFDAD6;
        --on-danger-container: #93000A;
        --info: #3B82F6;
        --hover: #EFF4FF;

        /* Radius scale */
        --radius-sm: 4px;
        --radius: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --radius-full: 9999px;

        /* Elevation (Level 1 = card, Level 2 = dropdown/popover) */
        --shadow-1: 0 1px 3px 0 rgba(15,23,42,0.05), 0 1px 2px -1px rgba(15,23,42,0.05);
        --shadow-2: 0 10px 15px -3px rgba(15,23,42,0.08);

        /* Back-compat aliases used throughout the component CSS below */
        --sky: var(--primary);
        --sky-dark: var(--primary-dark);
        --sky-soft: var(--secondary);
        --sky-border: var(--border);
        --ink: var(--text);
        --bg-page: var(--bg);
        --green: var(--success);
        --amber: var(--warning);
        --red: var(--danger);
    }
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; color: var(--text); }
    h1, h2, h3, h4, .hero-title, .candidate-name, .panel-subhead, .control-panel-header,
    .page-header-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main > div { padding-top: 0.6rem; }
    .stApp { background: var(--bg); }
    section[data-testid="stMain"] .block-container { padding-top: 1rem; }

    /* ---- Hero banner ---- */
    .hero-box {
        background: linear-gradient(135deg, #E3F5FD 0%, #ffffff 60%);
        border: 1px solid var(--sky-border);
        border-radius: 16px;
        padding: 30px 34px;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }
    .hero-box::after {
        content: "";
        position: absolute; top: -40%; right: -10%;
        width: 260px; height: 260px; border-radius: 50%;
        background: radial-gradient(circle, rgba(41,182,246,0.18) 0%, rgba(41,182,246,0) 70%);
    }
    .hero-title {
        color: var(--sky-dark);
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        margin-bottom: 4px;
    }
    .hero-sub { color: #4a6b80; font-size: 0.97rem; max-width: 700px; }

    /* ---- Sidebar (dark navy per reference design) ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #16213A 100%);
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    section[data-testid="stSidebar"] .stRadio label { font-weight: 600; }
    section[data-testid="stSidebar"] hr, section[data-testid="stSidebar"] div[style*="background:#D6EEFB"] {
        background: rgba(255,255,255,0.12) !important;
    }

    /* ---- Candidate cards ---- */
    .candidate-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: var(--radius-lg);
        padding: 22px 28px;
        margin-bottom: 16px;
        box-shadow: var(--shadow-1);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-2);
    }
    .candidate-avatar {
        flex-shrink: 0;
        width: 52px; height: 52px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem; font-weight: 800; color: #ffffff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .candidate-info { flex-grow: 1; }
    .candidate-name { font-size: 1.28rem; font-weight: 800; color: var(--ink); line-height: 1.2; }
    .candidate-rank-label {
        font-size: 0.72rem; font-weight: 700; color: var(--sky-dark);
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;
    }
    .rank-badge {
        display: inline-block;
        background: var(--sky-dark);
        color: #ffffff;
        font-weight: 800;
        font-size: 0.82rem;
        border-radius: 20px;
        padding: 3px 14px;
        margin-right: 10px;
        letter-spacing: 0.3px;
    }
    .score-pill {
        display: inline-block;
        font-weight: 800;
        border-radius: 10px;
        padding: 8px 18px;
        font-size: 1.15rem;
        flex-shrink: 0;
    }
    .skill-chip {
        display: inline-block;
        background: var(--secondary);
        color: var(--primary-dark);
        border: none;
        border-radius: var(--radius-full);
        padding: 4px 12px;
        margin: 2px 4px 2px 0;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .gap-chip {
        display: inline-block;
        background: var(--error-container, #ffdad6);
        color: #93000a;
        border: none;
        border-radius: var(--radius-full);
        padding: 4px 12px;
        margin: 2px 4px 2px 0;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .priority-note {
        background: var(--secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 10px 14px;
        font-size: 0.85rem;
        color: var(--primary-dark);
        margin-top: 6px;
    }

    /* ---- Control panel ---- */
    .control-panel-header {
        display: flex; align-items: center; gap: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem; font-weight: 700; color: var(--text);
        margin-bottom: 4px;
    }
    .panel-subhead {
        font-weight: 600; font-size: 12px; color: var(--primary-dark);
        text-transform: uppercase; letter-spacing: 0.02em;
        margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
    }
    .keyword-chip {
        display: inline-block;
        background: var(--secondary);
        color: var(--primary-dark);
        border: none;
        border-radius: var(--radius-full);
        padding: 4px 12px;
        margin: 3px 5px 3px 0;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .weight-bar {
        display: flex; height: 10px; border-radius: 6px; overflow: hidden;
        margin: 10px 0 6px;
        border: 1px solid #e4e4e4;
    }
    .weight-bar-seg { height: 100%; }

    /* ---- Buttons (gradient primary per reference design) ---- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        border-radius: var(--radius);
        padding: 0.6rem 1.4rem;
        box-shadow: 0 4px 14px rgba(56,189,248,0.35);
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, #004965 100%);
        box-shadow: 0 6px 18px rgba(14,165,233,0.45);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"]:active { transform: translateY(0); }

    /* Secondary (main-area, non-nav) buttons: light sky bg, sky-dark text — the "Ghost" look
       (transparent + gray text) is reserved for sidebar nav, styled separately below. */
    .stButton > button[kind="secondary"] {
        background: var(--secondary);
        color: var(--primary-dark);
        font-weight: 600;
        border: none;
        border-radius: var(--radius);
        box-shadow: none;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #CDEBFC;
    }

    /* ---- Sidebar nav buttons: muted light text on dark bg, active = gradient pill ---- */
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        color: #94A3B8 !important;
        border: none;
        border-radius: var(--radius);
        font-weight: 600;
        text-align: left;
        justify-content: flex-start;
        text-transform: none;
        letter-spacing: 0;
        padding: 0.55rem 1rem;
        box-shadow: none;
        margin-bottom: 2px;
        transition: background 0.15s ease, color 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.08);
        color: #F1F5F9 !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: #ffffff !important;
        border: none;
        box-shadow: 0 4px 12px rgba(56,189,248,0.35);
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, #004965 100%);
    }

    /* ---- Metrics ---- */
    div[data-testid="stMetricValue"] { color: var(--primary-dark); font-weight: 700; font-family: 'Plus Jakarta Sans', sans-serif; }
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 12px 16px 6px;
        box-shadow: var(--shadow-1);
    }

    /* ---- Page headers (unified across every page) ---- */
    @keyframes headerFadeIn {
        from { opacity: 0; transform: translateY(-8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .page-header {
        position: relative;
        display: flex;
        align-items: center;
        gap: 18px;
        overflow: hidden;
        margin: 8px 0 26px;
        padding: 20px 26px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-1);
        animation: headerFadeIn 0.45s ease;
    }
    .page-header-decoration {
        position: absolute; top: -50%; right: -6%;
        width: 220px; height: 220px; border-radius: 50%;
        background: radial-gradient(circle, rgba(56,189,248,0.10) 0%, rgba(56,189,248,0) 70%);
        pointer-events: none;
    }
    .page-header-icon {
        position: relative; z-index: 1;
        flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        width: 52px; height: 52px; border-radius: var(--radius-md);
        background: var(--secondary);
        font-size: 1.5rem;
    }
    .page-header-text {
        position: relative; z-index: 1;
        display: flex; flex-direction: column; gap: 4px;
    }
    .page-header-title {
        font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 1.75rem;
        color: var(--text); letter-spacing: -0.02em; line-height: 1.25;
    }
    .page-header-sub {
        font-size: 0.88rem; color: var(--text-secondary); line-height: 1.4;
    }

    /* ---- Generic containers/expanders ---- */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-1);
    }
    div[data-testid="stExpander"] summary { font-weight: 600; color: var(--text); }

    /* Bordered st.container(border=True) blocks — used for cards throughout */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: var(--radius-lg) !important;
        border-color: var(--border) !important;
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: var(--shadow-2);
    }

    /* ---- Clean, minimal tabs (no loud solid-fill background) ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; border-bottom: 1px solid var(--border); background: transparent; padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600; color: var(--text-secondary); border-radius: 0;
        padding: 8px 16px; transition: color 0.15s ease;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--primary-dark); }
    .stTabs [aria-selected="true"] {
        color: var(--primary-dark) !important;
        font-weight: 800;
        box-shadow: inset 0 -2px 0 var(--primary-dark);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* ---- Inputs: consistent rounding + focus ring ---- */
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div,
    .stDateInput input, .stTimeInput input {
        border-radius: 10px !important;
        border-color: var(--border) !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.18) !important;
    }
    .stSlider [role="slider"] { background: var(--primary-dark) !important; }
    .stCheckbox label, .stRadio label { font-weight: 500; }

    /* ---- Secondary (non-primary) buttons in main area ---- */
    .stButton > button[kind="secondary"] {
        border-radius: 10px; border-color: var(--border); color: var(--text);
        transition: all 0.15s ease;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--primary); color: var(--primary-dark); background: var(--secondary);
    }

    /* ---- Dividers ---- */
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

components.inject_design_system_css()

# ----------------------------- AUTH GATE (company login/signup) -----------------------------
# Only enforced when Supabase is actually configured — an app with no backend
# configured yet shouldn't lock itself out. Once logged in + a company
# profile exists, auth_user/auth_company stay in session_state for the rest
# of the app (and future stages) to use.
def _render_auth_gate():
    if not db.is_configured():
        return  # no backend configured — skip auth entirely, behave as before

    if auth.is_logged_in():
        company = auth.get_company_for_current_user()
        if company and st.session_state.get("auth_screen") != "signup_success":
            return  # fully set up and not mid-way through showing the new-org success screen — let the app render
        if not company and st.session_state.get("auth_screen") not in ("signup1", "signup_success"):
            st.session_state.auth_screen = "signup1"
        _render_auth_flow()
        st.stop()
    else:
        _render_auth_flow()
        st.stop()


_AUTH_CSS = "\n".join(line.strip() for line in """
<style>
.auth-hero { border-radius: 24px; padding: 3rem 2.5rem; min-height: 560px;
background: linear-gradient(160deg, #0B2A52 0%, #001736 70%, #001025 100%);
display: flex; flex-direction: column; justify-content: center; align-items: center;
text-align: center; color: white; }
.auth-hero-card { background: white; border-radius: 16px; padding: 1.75rem 2rem;
box-shadow: 0 10px 25px rgba(0,0,0,0.15); margin-bottom: 2rem; }
.auth-hero h1 { font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0 0 1rem 0; letter-spacing: -0.01em; }
.auth-hero p { font-size: 1rem; opacity: 0.85; max-width: 380px; margin: 0 auto; line-height: 1.5; }
.auth-badge { display: inline-block; background: #E5EEFF; color: #0051D5; font-weight: 700;
font-size: 0.7rem; letter-spacing: 0.04em; padding: 6px 12px; border-radius: 8px; margin-bottom: 1rem; }
div[data-testid="stForm"] button[kind="primary"], .stButton > button[kind="primary"] {
background: #001736 !important; border-color: #001736 !important; border-radius: 8px !important;
}
div[data-testid="stForm"] button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
background: #002B5C !important; border-color: #002B5C !important;
}
</style>
""".strip("\n").split("\n"))


def _auth_hero(title: str, subtitle: str):
    # IMPORTANT: every line here must have ZERO leading whitespace — Streamlit's
    # markdown parser treats 4+ leading spaces as a preformatted code block
    # (standard Markdown behavior) and will render it as literal text instead
    # of HTML, even with unsafe_allow_html=True. That flag only controls
    # whether HTML tags get escaped; it doesn't disable Markdown's own
    # indentation-based code-block detection, which runs first.
    logo_b64 = ""
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_header.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("ascii")
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:32px;" />' if logo_b64 else \
        '<span style="font-size:1.8rem;">🧭</span>'

    lines = [
        _AUTH_CSS,
        '<div class="auth-hero">',
        '<div class="auth-hero-card">',
        f'<div style="display:flex; align-items:center; gap:10px; justify-content:center;">{logo_html}</div>',
        '</div>',
        f'<h1>{title}</h1>',
        f'<p>{subtitle}</p>',
        '</div>',
    ]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def _render_auth_flow():
    st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)
    screen = st.session_state.get("auth_screen", "choose_company")
    left, right = st.columns([0.45, 0.55], gap="large")
    with right:
        _auth_hero(
            "Intelligent Candidate Discovery",
            "Empowering high-volume recruiters with cognitive precision and AI-driven insights to find the perfect talent faster.",
        )
    with left:
        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
        if screen == "choose_company":
            _render_choose_company()
        elif screen == "access_code":
            _render_access_code()
        elif screen == "signup1":
            _render_signup_step1()
        elif screen == "signup_success":
            _render_signup_success()


def _render_choose_company():
    st.markdown("### Company Selection")
    st.caption("Find and select your organization to continue.")

    search = st.text_input("Search your organization", placeholder="Type at least 3 characters...",
                            label_visibility="collapsed", key="choose_company_search")
    if len(search.strip()) >= 3:
        results = auth.list_companies(search)
        if not results:
            st.caption("No matching organizations found.")
        for comp in results:
            c1, c2 = st.columns([1, 5])
            with c1:
                if comp.get("logo_base64"):
                    import base64 as _b64
                    st.image(_b64.b64decode(comp["logo_base64"]), width=36)
                else:
                    st.markdown("🏢")
            with c2:
                if st.button(comp["name"], key=f"pick_company_{comp['id']}", width="stretch"):
                    st.session_state.chosen_company = comp
                    st.session_state.auth_screen = "access_code"
                    st.rerun()
    else:
        st.caption("Please enter 3 or more characters.")

    st.markdown("")
    if st.button("Don't see your organization? Create one", key="goto_signup_from_choose"):
        st.session_state.auth_screen = "signup1"
        st.rerun()


def _render_access_code():
    """The entire login step — enter the organization's code, get straight
    into the app. No email, no password, no separate 'Welcome Back' screen."""
    chosen = st.session_state.get("chosen_company")
    if not chosen:
        st.session_state.auth_screen = "choose_company"
        st.rerun()
        return

    c1, c2 = st.columns([1, 5])
    with c1:
        if chosen.get("logo_base64"):
            import base64 as _b64
            st.image(_b64.b64decode(chosen["logo_base64"]), width=40)
    with c2:
        st.markdown(f'<span class="auth-badge">{chosen["name"].upper()}</span>', unsafe_allow_html=True)

    st.markdown("### Enter Access Code")
    st.caption("Ask your organization's admin for this code — that's all you need to get in.")

    with st.form("access_code_form"):
        code = st.text_input("Access code", key="access_code_input", placeholder="e.g. 4721", max_chars=4)
        submitted = st.form_submit_button("Enter →", type="primary", width="stretch")
    if submitted:
        if not code.strip():
            st.error("Enter the access code.")
        else:
            ok, msg = auth.enter_company_with_code(chosen["id"], code)
            if ok:
                st.session_state.pop("chosen_company", None)
                st.rerun()
            else:
                st.error(msg)

    if st.button("← Choose a different organization", key="access_code_back"):
        st.session_state.auth_screen = "choose_company"
        st.session_state.pop("chosen_company", None)
        st.rerun()


def _render_signup_step1():
    already_logged_in = auth.is_logged_in()  # recovery case: account exists, org row never finished saving

    if already_logged_in:
        st.info("Finishing your organization setup below.")
        if st.button("🚪 Start over", key="signup1_logout"):
            auth.sign_out()
            st.rerun()
    else:
        if st.button("← Back", key="signup1_back"):
            st.session_state.auth_screen = "choose_company"
            st.rerun()

    st.markdown("### Create Organization")
    st.caption("Register a new organization on the ICD Platform — no account or password needed, "
                "you'll get a 4-digit access code at the end.")

    org = st.session_state.get("signup_org", {})
    logo = st.file_uploader("Organization Logo (optional)", type=["png", "jpg", "jpeg"], key="signup_logo")
    name = st.text_input("Organization Name *", value=org.get("name", ""), key="signup_org_name")
    website = st.text_input("Official Website", value=org.get("website", ""), key="signup_org_website",
                             placeholder="https://yourcompany.com")
    c1, c2 = st.columns(2)
    industry_options = ["", "Technology", "Finance", "Healthcare", "Education", "Retail", "Manufacturing", "Other"]
    industry = c1.selectbox("Industry", industry_options,
                             index=industry_options.index(org["industry"]) if org.get("industry") in industry_options else 0,
                             key="signup_org_industry")
    size_options = ["", "1 - 50", "51 - 200", "201 - 1000", "1000+"]
    company_size = c2.selectbox("Company Size", size_options,
                                 index=size_options.index(org["company_size"]) if org.get("company_size") in size_options else 0,
                                 key="signup_org_size")

    b1, b2 = st.columns(2)
    if not already_logged_in and b1.button("Cancel", key="signup1_cancel", width="stretch"):
        st.session_state.auth_screen = "choose_company"
        st.session_state.pop("signup_org", None)
        st.rerun()
    if b2.button("Create Organization", key="signup1_next", type="primary", width="stretch"):
        if not name.strip():
            st.error("Organization Name is required.")
        else:
            ok, msg = auth.create_company(
                name.strip(), logo.getvalue() if logo else org.get("logo_bytes"),
                website=website.strip(), industry=industry, company_size=company_size,
            )
            if ok:
                st.session_state.pop("signup_org", None)
                st.session_state.auth_screen = "signup_success"
                st.rerun()
            else:
                st.error(msg)


def _render_signup_success():
    company = st.session_state.get("auth_company") or {}
    code = company.get("access_code", "")

    st.markdown("### 🎉 Organization Created")
    st.caption(f"**{company.get('name', 'Your organization')}** is ready to go.")

    st.markdown("---")
    st.markdown("**Your organization's access code**")
    st.caption("This is the only thing anyone needs to get in — share it with your team. "
                "Save it somewhere safe; you can also find it later in the sidebar.")
    st.markdown(f"""
    <div style="text-align:center; background:#F1F5F9; border:2px dashed #001736; border-radius:12px; padding:1.5rem; margin:1rem 0;">
        <span style="font-size:2.5rem; font-weight:800; letter-spacing:0.3rem; color:#001736; font-family:monospace;">{code}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if st.button("Continue to ICD Platform →", type="primary", width="stretch"):
        st.session_state.auth_screen = "choose_company"
        st.rerun()


_render_auth_gate()

if auth.is_logged_in():
    with st.sidebar:
        _company = st.session_state.get("auth_company") or {}

        # Force readable, high-contrast text everywhere in the sidebar. Three
        # zones, each needing different text color because their IMMEDIATE
        # background differs:
        #  1. Expander HEADERS (summary bars) — light background -> dark text
        #  2. Expander BODY, directly on the dark sidebar -> light text
        #  3. Light-background sub-widgets INSIDE that body (file uploader
        #     dropzone, select boxes, text inputs) -> dark text again, since
        #     their own background is white even though their container's
        #     background is dark. This is what kept getting missed before.
        st.markdown("""
        <style>
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] p,
        section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] label {
            color: #F1F5F9 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
            color: #0B1C30 !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *,
        section[data-testid="stSidebar"] [data-baseweb="select"],
        section[data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #0B1C30 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # App logo in a clean white card (the PNG has a white background, so it
        # needs its own container rather than sitting directly on the dark
        # sidebar) — embedded as base64 for crisp, un-blurred rendering at a
        # size that matches its real aspect ratio instead of being stretched.
        _app_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_header.png")
        if os.path.exists(_app_logo_path):
            with open(_app_logo_path, "rb") as _f:
                _logo_b64 = base64.b64encode(_f.read()).decode("ascii")
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:14px 18px; margin-bottom:12px; text-align:center;">
                <img src="data:image/png;base64,{_logo_b64}" style="width:100%; max-width:200px; height:auto; display:block; margin:0 auto;" />
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        cols = st.columns([1, 3])
        if _company.get("logo_base64"):
            import base64 as _b64
            cols[0].image(_b64.b64decode(_company["logo_base64"]), width=56)
        cols[1].markdown(f"**{_company.get('name', '')}**")

        if _company.get("access_code"):
            with st.expander("🔑 Access code"):
                st.markdown(f"""
                <div style="text-align:center; background:#F1F5F9; border:1px solid #CBD5E1; border-radius:8px;
                            padding:10px; margin:4px 0;">
                    <span style="font-size:1.4rem; font-weight:800; letter-spacing:0.2rem; color:#0B1C30 !important; font-family:monospace;">{_company["access_code"]}</span>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Share this with teammates so they can find and unlock your organization at login.")

        with st.expander("⚙️ Company Settings"):
            with st.form("company_settings_form"):
                new_name = st.text_input("Organization Name", value=_company.get("name", ""))
                new_logo = st.file_uploader("Organization Logo", type=["png", "jpg", "jpeg"])
                new_website = st.text_input("Official Website", value=_company.get("website", ""))
                _industry_options = ["", "Technology", "Finance", "Healthcare", "Education", "Retail", "Manufacturing", "Other"]
                new_industry = st.selectbox("Industry", _industry_options,
                                             index=_industry_options.index(_company["industry"]) if _company.get("industry") in _industry_options else 0)
                _size_options = ["", "1 - 50", "51 - 200", "201 - 1000", "1000+"]
                new_size = st.selectbox("Company Size", _size_options,
                                         index=_size_options.index(_company["company_size"]) if _company.get("company_size") in _size_options else 0)
                settings_submitted = st.form_submit_button("💾 Save Changes", type="primary")
            if settings_submitted:
                if not new_name.strip():
                    st.error("Organization Name is required.")
                else:
                    _updates = {"name": new_name.strip(), "website": new_website.strip(),
                                "industry": new_industry, "company_size": new_size}
                    if new_logo is not None:
                        import base64 as _b64
                        _updates["logo_base64"] = _b64.b64encode(new_logo.getvalue()).decode("ascii")
                    ok, msg = auth.update_company(_company.get("id"), _updates)
                    if ok:
                        st.success("Saved.")
                        st.rerun()
                    else:
                        st.error(msg)

        if st.button("🚪 Log out", key="global_logout_btn"):
            auth.sign_out()
            st.rerun()


# ----------------------------- SESSION STATE -----------------------------
if "local_candidates" not in st.session_state:
    st.session_state.local_candidates = []  # session-only fallback when Supabase isn't configured or a save fails
if "candidates" not in st.session_state:
    st.session_state.candidates = []  # list of dicts: {id, name, raw_text, profile, score, ...} — resynced from Supabase below
if "job_role" not in st.session_state:
    st.session_state.job_role = ""
if "job_details" not in st.session_state:
    st.session_state.job_details = ""
if "weights" not in st.session_state:
    st.session_state.weights = {"skills": 40, "experience": 40, "education": 20}
if "top_n" not in st.session_state:
    st.session_state.top_n = 10
if "excluded_files" not in st.session_state:
    st.session_state.excluded_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "local_prefs" not in st.session_state:
    st.session_state.local_prefs = local_settings.load_settings()
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (st.session_state.local_prefs.get("default_theme") == "dark")
if "global_search" not in st.session_state:
    st.session_state.global_search = ""
if "compare_list" not in st.session_state:
    st.session_state.compare_list = []
if "candidates_view" not in st.session_state:
    st.session_state.candidates_view = "grid"  # "grid" | "profile" | "compare"
if "selected_candidate_key" not in st.session_state:
    st.session_state.selected_candidate_key = None
if "local_jobs" not in st.session_state:
    st.session_state.local_jobs = []  # session-only fallback when Supabase isn't configured
if "local_interviews" not in st.session_state:
    st.session_state.local_interviews = []  # session-only fallback
if "selected_job_id" not in st.session_state:
    st.session_state.selected_job_id = None
if "jobs_view" not in st.session_state:
    st.session_state.jobs_view = "list"  # "list" | "form"
if "editing_job_id" not in st.session_state:
    st.session_state.editing_job_id = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# Every rerun starts by loading whatever is actually persisted (Supabase, or
# the session-local fallback) — this is what makes candidates, decisions,
# bookmarks, notes, and interview scores survive closing/reopening the app.
# The Resume Screening page below may temporarily replace this with just the
# current run's results for display, but a fresh save happens per-candidate,
# so the very next rerun re-syncs the full persisted set again.
st.session_state.candidates = get_candidates()

# ----------------------------- DARK MODE OVERRIDE (applied after base palette) -----------------------------
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    :root {
        --bg: #0F172A; --card: #1E293B; --text: #F1F5F9; --text-secondary: #94A3B8;
        --border: #334155; --hover: #1E293B; --secondary: #0C4A6E;
    }
    .stApp { background: var(--bg); }
    section[data-testid="stSidebar"] { background: #1E293B !important; border-right: 1px solid #334155 !important; }
    section[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
    .candidate-card, div[data-testid="stMetric"], .st-key-fab_panel { background: #1E293B !important; }
    .stTextInput input, .stTextArea textarea { background: #1E293B !important; color: #F1F5F9 !important; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        transition: transform 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
        border-radius: 10px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        transform: translateX(3px);
        background: rgba(56,189,248,0.12) !important;
        border-color: rgba(56,189,248,0.4) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(90deg, #00668A, #38BDF8) !important;
        border: none !important;
        box-shadow: 0 2px 10px rgba(56,189,248,0.35);
        position: relative;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]::before {
        content: ""; position: absolute; left: -8px; top: 50%; transform: translateY(-50%);
        width: 4px; height: 60%; background: #FFFFFF; border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:6px 0 2px;">
        <div style="font-size:1.5rem; font-weight:800; color:#38BDF8; letter-spacing:-0.3px;">🎯 ICD Platform</div>
        <div style="font-size:0.8rem; color:#94A3B8; margin-top:2px;">AI-Powered Resume Screening & Candidate Ranking</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="height:1px; background:linear-gradient(90deg,#38BDF8,transparent); margin:14px 0 16px;"></div>', unsafe_allow_html=True)

    key_status = check_api_key()
    if key_status:
        st.markdown(f"""
        <div style="background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); border-radius:10px; padding:10px 14px; margin-bottom:6px;">
            <span style="color:#38BDF8; font-weight:700; font-size:0.85rem;">✅ API ready</span><br>
            <span style="color:#CBD5E1; font-size:0.78rem;">{active_provider()}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No API key found", icon="⚠️")
        with st.expander("How to add your API key"):
            st.markdown(
                "**Recommended: Groq (30 req/min free)**\n"
                "1. Get a free key at **console.groq.com** → API Keys → Create API Key\n\n"
                "**Recommended: Cerebras (30 req/min free, huge daily quota)**\n"
                "1. Get a free key at **cloud.cerebras.ai** → API Keys\n\n"
                "Adding both roughly doubles throughput — calls are split between them automatically.\n\n"
                "**Optional: Gemini (final fallback)**\n"
                "1. Get a free key at **aistudio.google.com** → Get API key\n\n"
                "Then create `.streamlit/secrets.toml`:\n"
                "```toml\nGROQ_API_KEY = \"your-groq-key\"\nCEREBRAS_API_KEY = \"your-cerebras-key\"\nGEMINI_API_KEY = \"your-gemini-key\"\n```\n"
                "You only need one — restart the app after saving."
            )

    if st.session_state.candidates:
        n_ok = len([c for c in st.session_state.candidates if not c["score"].get("error")])
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); border-radius:10px; padding:10px 14px; margin-top:8px;">
            <span style="color:#E2E8F0; font-size:0.8rem;">📋 <b>{n_ok}</b> candidate(s) stored</span>
        </div>
        """, unsafe_allow_html=True)

    if db.is_configured():
        st.markdown("""
        <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); border-radius:10px; padding:8px 14px; margin-top:8px;">
            <span style="color:#38BDF8; font-size:0.78rem;">🗄️ Persistent history: <b>connected</b></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10); border-radius:10px; padding:8px 14px; margin-top:8px;">
            <span style="color:#94A3B8; font-size:0.78rem;">🗄️ Persistent history: not connected (Home charts use this session only)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px; background:rgba(255,255,255,0.12); margin:18px 0 14px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem; font-weight:800; color:#38BDF8; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:6px;">Navigation</div>', unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Home"

    nav_items = ["🏠 Home", "📤 Resume Screening", "👥 Candidates", "📋 Jobs", "🗣️ Interview", "⭐ Shortlisted", "📄 Reports", "🤖 AI Insights"]
    for item in nav_items:
        is_active = st.session_state.current_page == item
        if st.button(item, key=f"nav_{item}", width="stretch", type="primary" if is_active else "secondary"):
            st.session_state.current_page = item
            st.rerun()
    page = st.session_state.current_page

    st.markdown('<div style="height:1px; background:rgba(255,255,255,0.12); margin:18px 0 14px;"></div>', unsafe_allow_html=True)
    if st.session_state.candidates:
        if "confirm_clear_candidates" not in st.session_state:
            st.session_state.confirm_clear_candidates = False

        if not st.session_state.confirm_clear_candidates:
            if st.button("🗑️ Clear all candidates", width="stretch"):
                st.session_state.confirm_clear_candidates = True
                st.rerun()
        else:
            st.warning("Remove all screened candidates? They'll disappear from Home, Candidates, Interview, and "
                       "Shortlisted, but the records are archived (not permanently deleted) in Supabase.")
            cc1, cc2 = st.columns(2)
            if cc1.button("✅ Yes, clear", width="stretch"):
                clear_all_candidates_record()
                st.session_state.candidates = []
                st.session_state.processed = False
                st.session_state.chat_history = []
                st.session_state.confirm_clear_candidates = False
                st.rerun()
            if cc2.button("Cancel", width="stretch"):
                st.session_state.confirm_clear_candidates = False
                st.rerun()

# ----------------------------- TOP NAVBAR -----------------------------
@st.cache_data
def _load_logo_b64():
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo_header.png")
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return None

_logo_b64 = _load_logo_b64()


_ICD_LOADER_CSS = """
<style>
  .icd-loader-root{
    --blue-light:#2E7CFF;
    --blue-mid:#005CFF;
    --blue-dark:#001E5A;
    width:100%;
    display:flex;
    align-items:center;
    justify-content:center;
    padding:14px 0 10px;
  }
  .icd-loader-root *{box-sizing:border-box;}
  .icd-stage{ position:relative; width:260px; max-width:80vw; height:230px;
    display:flex; align-items:center; justify-content:center; }
  .icd-core-wrap{ position:absolute; top:38%; left:50%; transform:translate(-50%,-50%);
    width:200px; height:200px; display:flex; align-items:center; justify-content:center; }
  .icd-ring{ position:absolute; width:176px; height:176px; border-radius:50%; padding:2px;
    background:conic-gradient(from 0deg, rgba(0,92,255,0) 0deg, rgba(46,124,255,0.15) 60deg,
      var(--blue-mid) 150deg, #7fb2ff 200deg, var(--blue-mid) 260deg, rgba(0,92,255,0.15) 320deg, rgba(0,92,255,0) 360deg);
    -webkit-mask:radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 4px));
    mask:radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 4px));
    animation: icdSpin 4s linear infinite, icdRingFade 7s ease-in-out infinite;
    filter:drop-shadow(0 0 10px rgba(0,92,255,0.55)); opacity:0; }
  .icd-ring.inner{ width:148px; height:148px;
    animation: icdSpinReverse 6s linear infinite, icdRingFade 7s ease-in-out infinite;
    animation-delay:0s, 0.15s; opacity:0; filter:drop-shadow(0 0 8px rgba(0,92,255,0.4));
    background:conic-gradient(from 90deg, rgba(0,30,90,0) 0deg, rgba(0,30,90,0.25) 90deg,
      var(--blue-dark) 180deg, rgba(0,30,90,0.25) 270deg, rgba(0,30,90,0) 360deg); }
  @keyframes icdSpin{ from{transform:rotate(0deg);} to{transform:rotate(360deg);} }
  @keyframes icdSpinReverse{ from{transform:rotate(360deg);} to{transform:rotate(0deg);} }
  @keyframes icdRingFade{ 0%{opacity:0;} 10%{opacity:1;} 90%{opacity:1;} 100%{opacity:0;} }
  .icd-orbit{ position:absolute; width:188px; height:188px; animation: icdSpin 10s linear infinite; }
  .icd-orbit .icd-dot{ position:absolute; width:7px; height:7px; border-radius:50%;
    background:radial-gradient(circle, #ffffff 0%, var(--blue-mid) 60%, transparent 100%);
    box-shadow:0 0 8px 2px rgba(0,92,255,0.8); }
  .icd-orbit .icd-dot:nth-child(1){ top:0; left:50%; transform:translate(-50%,-50%);}
  .icd-orbit .icd-dot:nth-child(2){ top:50%; left:100%; transform:translate(-50%,-50%);}
  .icd-orbit .icd-dot:nth-child(3){ top:100%; left:50%; transform:translate(-50%,-50%);}
  .icd-orbit .icd-dot:nth-child(4){ top:50%; left:0%; transform:translate(-50%,-50%);}
  .icd-orbit.reverse{ width:204px; height:204px; animation: icdSpinReverse 14s linear infinite; }
  .icd-orbit.reverse .icd-dot{ width:5px; height:5px; box-shadow:0 0 6px 1px rgba(0,92,255,0.6); }
  .icd-orbit.reverse .icd-dot:nth-child(1){ top:8%; left:75%; transform:translate(-50%,-50%);}
  .icd-orbit.reverse .icd-dot:nth-child(2){ top:75%; left:92%; transform:translate(-50%,-50%);}
  .icd-orbit.reverse .icd-dot:nth-child(3){ top:92%; left:25%; transform:translate(-50%,-50%);}
  .icd-orbit.reverse .icd-dot:nth-child(4){ top:25%; left:8%; transform:translate(-50%,-50%);}
  .icd-logo-glow{ position:relative; width:118px; height:auto; display:flex; align-items:center;
    justify-content:center; opacity:0; transform:scale(0.85);
    animation: icdLogoAppear 7s cubic-bezier(.22,1,.36,1) infinite, icdBreathe 7s ease-in-out infinite; }
  @keyframes icdLogoAppear{ 0%{opacity:0; transform:scale(0.8);} 18%{opacity:1; transform:scale(1);}
    85%{opacity:1; transform:scale(1);} 100%{opacity:0; transform:scale(0.92);} }
  @keyframes icdBreathe{
    0%,18%{ filter:drop-shadow(0 0 6px rgba(0,92,255,0.25)) drop-shadow(0 0 0px rgba(0,92,255,0)); }
    50%{ filter:drop-shadow(0 0 20px rgba(0,92,255,0.65)) drop-shadow(0 0 34px rgba(0,92,255,0.25)); }
    85%,100%{ filter:drop-shadow(0 0 6px rgba(0,92,255,0.25)) drop-shadow(0 0 0px rgba(0,92,255,0)); } }
  .icd-logo-glow img{ width:100%; height:auto; display:block; position:relative; z-index:2; }
  .icd-glass-plate{ position:absolute; width:140px; height:90px; border-radius:18px;
    background:rgba(255,255,255,0.35); backdrop-filter:blur(5px);
    box-shadow:0 0 30px rgba(0,92,255,0.15) inset, 0 10px 30px rgba(0,60,180,0.12);
    border:1px solid rgba(255,255,255,0.6); z-index:1; }
  .icd-scan-line{ position:absolute; left:8%; width:84%; height:3px; top:0; border-radius:3px;
    background:linear-gradient(90deg, rgba(0,92,255,0) 0%, #7fb2ff 20%, #ffffff 50%, #7fb2ff 80%, rgba(0,92,255,0) 100%);
    box-shadow:0 0 14px 3px rgba(0,92,255,0.7); z-index:3; opacity:0; animation: icdScanMove 7s ease-in-out infinite; }
  @keyframes icdScanMove{ 0%,16%{ top:6%; opacity:0;} 22%{ opacity:1;} 50%{ top:94%; opacity:1;}
    56%{ opacity:0;} 100%{ top:94%; opacity:0;} }
  .icd-status{ position:absolute; bottom:2px; left:50%; transform:translateX(-50%); text-align:center;
    opacity:0; width:100%; animation: icdTextFade 7s ease-in-out infinite; }
  @keyframes icdTextFade{ 0%,15%{opacity:0; transform:translate(-50%,10px);} 25%{opacity:1; transform:translate(-50%,0);}
    88%{opacity:1; transform:translate(-50%,0);} 100%{opacity:0; transform:translate(-50%,-6px);} }
  .icd-status .icd-label{ font-family:'Segoe UI',Arial,sans-serif; font-size:13px; letter-spacing:1.5px;
    font-weight:600; color:var(--blue-dark); text-transform:uppercase; }
  .icd-status .icd-highlight{ background:linear-gradient(90deg,var(--blue-mid),var(--blue-light));
    -webkit-background-clip:text; background-clip:text; color:transparent; }
  .icd-dots{ display:inline-flex; gap:4px; margin-left:5px; vertical-align:middle; }
  .icd-dots span{ width:5px; height:5px; border-radius:50%; background:var(--blue-mid); display:inline-block;
    animation: icdDotPulse 1.2s ease-in-out infinite; box-shadow:0 0 6px rgba(0,92,255,0.6); }
  .icd-dots span:nth-child(2){animation-delay:0.2s;}
  .icd-dots span:nth-child(3){animation-delay:0.4s;}
  @keyframes icdDotPulse{ 0%,80%,100%{ transform:scale(0.6); opacity:0.4;} 40%{ transform:scale(1.2); opacity:1;} }
</style>
"""


def _icd_loader_html(message: str) -> str:
    logo_tag = (
        f'<img src="data:image/png;base64,{_logo_b64}" alt="ICD Platform">'
        if _logo_b64 else ""
    )
    return f"""
{_ICD_LOADER_CSS}
<div class="icd-loader-root">
  <div class="icd-stage">
    <div class="icd-core-wrap">
      <div class="icd-orbit reverse">
        <div class="icd-dot"></div><div class="icd-dot"></div><div class="icd-dot"></div><div class="icd-dot"></div>
      </div>
      <div class="icd-orbit">
        <div class="icd-dot"></div><div class="icd-dot"></div><div class="icd-dot"></div><div class="icd-dot"></div>
      </div>
      <div class="icd-ring"></div>
      <div class="icd-ring inner"></div>
      <div class="icd-logo-glow">
        <div class="icd-glass-plate"></div>
        <div class="icd-scan-line"></div>
        {logo_tag}
      </div>
    </div>
    <div class="icd-status">
      <div class="icd-label"><span class="icd-highlight">AI Assistant</span> {message}<span class="icd-dots"><span></span><span></span><span></span></span></div>
    </div>
  </div>
</div>
"""


@contextmanager
def logo_loading(message: str = "Loading..."):
    """Premium animated loading screen (rotating AI ring, breathing logo glow,
    scan line) shown while the AI Assistant is generating a reply. The
    answer text is only written to the page after this block exits, so the
    animation fully owns the screen until a real response is ready."""
    ph = st.empty()
    ph.markdown(_icd_loader_html(message), unsafe_allow_html=True)
    try:
        yield
    finally:
        ph.empty()

st.markdown("""
<style>
.st-key-topnavbar {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 30px 36px; margin: 6px 0 28px; min-height: 128px;
    box-shadow: var(--shadow-1);
    position: sticky; top: 0; z-index: 999;
    display: flex; align-items: center;
}
.navbar-logo-row { display:flex; align-items:center; gap:18px; }
.navbar-logo-row img { height:70px; width:auto; flex-shrink:0; }
.navbar-logo { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.9rem; color: var(--text); line-height:1.15; }
.navbar-logo span { color: var(--primary); }
.navbar-tagline { font-size:0.72rem; color: var(--primary); font-weight:700; letter-spacing:0.6px; text-transform:uppercase; margin-top:2px; }
.navbar-breadcrumb { font-size:0.78rem; color: var(--text-secondary); margin-top:3px; }
.navbar-breadcrumb b { color: var(--primary); font-weight:700; }
.st-key-topnavbar div[data-testid="stButton"] { display:flex; align-items:center; height:100%; justify-content:flex-end; }
.st-key-topnavbar div[data-testid="stButton"] button { min-height:44px; }
</style>
""", unsafe_allow_html=True)

if "tagline_quote" not in st.session_state:
    st.session_state.tagline_quote = random.choice([
        "\u201cGreat teams aren't found by luck \u2014 they're discovered on purpose.\u201d",
        "\u201cThe right resume is easy to miss. The right process isn't.\u201d",
        "\u201cHiring well is the highest-leverage thing a team ever does.\u201d",
        "\u201cEvery great hire starts with someone taking the time to really look.\u201d",
    ])

with st.container(key="topnavbar"):
    nav_logo, nav_action = st.columns([5.2, 1.3], vertical_alignment="center")

    with nav_logo:
        current_label = st.session_state.get("current_page", "🏠 Home")
        breadcrumb = "🏠 Home" if current_label == "🏠 Home" else f"🏠 Home / <b>{current_label}</b>"
        logo_img_html = f'<img src="data:image/png;base64,{_logo_b64}" alt="ICD Platform logo">' if _logo_b64 else "🎯"
        st.markdown(f"""
        <div class="navbar-logo-row">
            {logo_img_html}
            <div>
                <div class="navbar-logo"><span>ICD</span> Platform</div>
                <div class="navbar-tagline">Intelligent Candidate Discovery Platform</div>
                <div class="navbar-breadcrumb">{breadcrumb}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with nav_action:
        if st.button("➕ Quick Action", key="nav_quick_action", width="stretch"):
            st.session_state.current_page = "📤 Resume Screening"
            st.rerun()

# ============================================================
# PAGE 1 — UPLOAD & SCREEN
# ============================================================
if page == "📤 Resume Screening":
    page_header("📤", "Resume Screening", "Set your job requirements and priorities, then upload resumes to screen.")
    st.markdown('<div class="control-panel-header">⚙️ Recruiter Control Panel</div>', unsafe_allow_html=True)
    st.write("")

    p1, p2, p3, p4 = st.columns([1.3, 1, 1.1, 0.7], gap="medium")

    with p1:
        st.markdown('<div class="panel-subhead">📋 Job Description</div>', unsafe_allow_html=True)

        saved_jobs = get_jobs(include_archived=False)
        if saved_jobs:
            job_options = ["— Type manually —"] + [j["title"] for j in saved_jobs]
            picked = st.selectbox("Use a saved job", job_options, label_visibility="collapsed", key="job_picker")
            if picked != "— Type manually —":
                picked_job = next(j for j in saved_jobs if j["title"] == picked)
                if st.session_state.selected_job_id != picked_job["id"]:
                    st.session_state.selected_job_id = picked_job["id"]
                    st.session_state.job_role = picked_job["title"]
                    details_parts = [
                        picked_job.get("description", ""),
                        ("Responsibilities: " + picked_job["responsibilities"]) if picked_job.get("responsibilities") else "",
                        ("Required skills: " + ", ".join(picked_job.get("required_skills", []))) if picked_job.get("required_skills") else "",
                    ]
                    st.session_state.job_details = "\n\n".join(p for p in details_parts if p)
                    st.rerun()

        job_role = st.text_input("Job title", value=st.session_state.job_role, placeholder="e.g. Senior AI/ML Engineer", label_visibility="collapsed")
        st.session_state.job_role = job_role
        job_details = st.text_area(
            "Requirements", value=st.session_state.job_details, height=170,
            placeholder="e.g. Required: machine learning, deep learning, TensorFlow, PyTorch, "
                        "NLP, LLMs, transformers, model deployment, MLOps.",
            label_visibility="collapsed",
        )
        st.session_state.job_details = job_details

    jd_full_text = f"{job_role} {job_details}"
    keywords = extract_keywords(jd_full_text)

    with p2:
        st.markdown('<div class="panel-subhead">🔍 Keywords</div>', unsafe_allow_html=True)
        if keywords:
            components.chip_list(keywords, variant="keyword")
        else:
            st.caption("Auto-extracted from the job description as you type.")

    with p3:
        st.markdown('<div class="panel-subhead">⚖️ Scoring Weights</div>', unsafe_allow_html=True)
        weight_skills = st.slider("🎯 Skill", 0, 100, st.session_state.weights["skills"], key="w_skills")
        weight_experience = st.slider("📅 Experience", 0, 100, st.session_state.weights["experience"], key="w_experience")
        weight_education = st.slider("🎓 Education", 0, 100, st.session_state.weights["education"], key="w_education")

        total_w = weight_skills + weight_experience + weight_education
        if total_w == 0:
            weight_skills, weight_experience, weight_education = 34, 33, 33
            total_w = 100
        st.session_state.weights = {"skills": weight_skills, "experience": weight_experience, "education": weight_education}

        norm_skills = round(weight_skills / total_w * 100)
        norm_exp = round(weight_experience / total_w * 100)
        norm_edu = 100 - norm_skills - norm_exp
        st.markdown(f"""
        <div class="weight-bar">
            <div class="weight-bar-seg" style="width:{norm_skills}%; background:#12314A;"></div>
            <div class="weight-bar-seg" style="width:{norm_exp}%; background:#38BDF8;"></div>
            <div class="weight-bar-seg" style="width:{norm_edu}%; background:#d8d3bc;"></div>
        </div>
        <div style="font-size:0.78rem; color:#555; font-weight:600;">
            Skill {norm_skills}% &nbsp; Experience {norm_exp}% &nbsp; Education {norm_edu}%
        </div>
        """, unsafe_allow_html=True)

    with p4:
        st.markdown('<div class="panel-subhead">👥 Top N</div>', unsafe_allow_html=True)
        top_n = st.slider("Candidates to show", 1, 50, st.session_state.top_n, label_visibility="collapsed")
        st.session_state.top_n = top_n
        st.markdown(f'<div style="text-align:center; font-size:1.6rem; font-weight:800; color:#00668A;">{top_n}</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="panel-subhead">📤 Upload Resumes</div>', unsafe_allow_html=True)
    tab_files, tab_zip = st.tabs(["Individual files", "📁 Upload a folder (as .zip)"])

    with tab_files:
        files = st.file_uploader(
            "PDF, DOCX, or a photo/scan of a resume (JPG/PNG) — upload as many as you like",
            type=["pdf", "docx", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="individual_files",
        )

    with tab_zip:
        st.caption("Zip your resumes folder and upload it here — mixed PDF/DOCX/image files are fine. "
                    "Non-resume files (certificates, cover letters, random documents) are detected and skipped automatically.")
        zip_file = st.file_uploader(
            "Upload a .zip of resumes",
            type=["zip"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="zip_upload",
        )

    # Unify both sources into a single (name, bytes) list.
    raw_items = []  # list of (name, bytes)
    if files:
        for f in files:
            raw_items.append((f.name, f.read()))
            f.seek(0)
    if zip_file:
        try:
            zip_items = extract_files_from_zip(zip_file)
            raw_items.extend(zip_items)
            st.caption(f"📁 {len(zip_items)} supported file(s) found inside the ZIP.")
        except Exception as e:
            st.error(f"Could not read ZIP: {e}")

    if raw_items:
        st.caption(f"{len(raw_items)} file(s) ready to process")
        n_images = sum(1 for name, _ in raw_items if name.lower().endswith((".jpg", ".jpeg", ".png")))
        if n_images:
            st.caption(f"📷 {n_images} image resume(s) will be read via OCR — clearer scans work best.")

    st.divider()

    jd = f"Job Role: {job_role}\n\nKey Requirements:\n{job_details}".strip()
    run = st.button(
        "🚀 Screen Candidates",
        type="primary",
        width="stretch",
        disabled=not (raw_items and job_role.strip() and check_api_key()),
    )

    if not check_api_key():
        st.info("Add an API key in the sidebar to enable screening.")
    elif not job_role.strip():
        st.info("Enter a job title above to get started.")
    elif not raw_items:
        st.info("Upload at least one resume (or a ZIP folder) to get started.")

    if run:
        st.session_state.candidates = []
        st.session_state.excluded_files = []
        _screen_logo_ph = st.empty()
        if _logo_b64:
            _screen_logo_ph.markdown(f"""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:18px 0 6px;">
                <img src="data:image/png;base64,{_logo_b64}" style="width:48px; height:auto; animation: icdPulse 1.3s ease-in-out infinite;">
            </div>
            <style>
            @keyframes icdPulse {{
                0%, 100% {{ opacity: 0.32; transform: scale(0.86); }}
                50% {{ opacity: 1; transform: scale(1.08); }}
            }}
            </style>
            """, unsafe_allow_html=True)
        progress = st.progress(0, text="Extracting text from files...")
        total = len(raw_items)

        # Step 1: extract text from every file up front (fast, local, no API calls)
        file_texts = {}
        for name, data in raw_items:
            try:
                file_texts[name] = extract_text_from_bytes(name, data)
            except Exception as e:
                file_texts[name] = None
                st.session_state.candidates.append({
                    "filename": name, "name": name, "raw_text": "",
                    "profile": {}, "score": {"overall_score": 0, "error": str(e)},
                    "screened_at": datetime.now().isoformat(), "status": "Waiting",
                })

        # Step 1.5: two-pass "is this actually a resume?" filter.
        # Pass 1 (free, instant): local heuristic — section keywords + contact info.
        # Pass 2 (only for files that failed pass 1): a real AI classification call,
        # so a resume with an unusual layout never gets silently dropped on a
        # single weak heuristic signal.
        confirmed_texts = {}
        for name, text in file_texts.items():
            if not text:
                continue
            check = heuristic_resume_check(text)
            if check["looks_like_resume"]:
                confirmed_texts[name] = text
                continue
            # Second opinion before excluding.
            try:
                ai_check = is_resume_ai_check(text)
            except Exception:
                # If the AI check itself fails (e.g. rate limit), err on the
                # side of keeping the file rather than losing a real resume.
                confirmed_texts[name] = text
                continue
            if ai_check.get("is_resume"):
                confirmed_texts[name] = text
            else:
                st.session_state.excluded_files.append({
                    "filename": name,
                    "reason": ai_check.get("reason", "Does not appear to be a resume."),
                })

        if st.session_state.excluded_files:
            with st.expander(f"⚠️ {len(st.session_state.excluded_files)} file(s) excluded — not recognized as resumes", expanded=True):
                for ex in st.session_state.excluded_files:
                    st.write(f"**{ex['filename']}** — {ex['reason']}")
                st.caption("Each of these was checked twice (a fast local pass, then an AI confirmation) before being excluded.")

        eta_ph = st.empty()


        # Snapshot the priority weights at run time and normalize to 100.
        w = st.session_state.weights
        w_total = max(1, w["skills"] + w["experience"] + w["education"])
        w_norm = {
            "skills_match": w["skills"] / w_total,
            "experience_fit": w["experience"] / w_total,
            "education_fit": w["education"] / w_total,
        }

        def process_one(name, raw_text):
            profile, score_result = parse_and_score(raw_text, jd)
            breakdown = score_result.get("breakdown", {})
            weighted = round(
                breakdown.get("skills_match", 0) * w_norm["skills_match"]
                + breakdown.get("experience_fit", 0) * w_norm["experience_fit"]
                + breakdown.get("education_fit", 0) * w_norm["education_fit"]
            )
            score_result["ai_overall_score"] = score_result.get("overall_score", weighted)
            score_result["overall_score"] = weighted  # ranking uses the weighted score
            # Nested inside profile (not a separate top-level field) so it
            # travels automatically through profile_json on save/reload — no
            # schema change needed.
            profile["extraction_flags"] = assess_extraction_confidence(profile, raw_text)
            return {
                "filename": name,
                "name": profile.get("name") or name,
                "raw_text": raw_text,
                "profile": profile,
                "score": score_result,
                "screened_at": datetime.now().isoformat(),
                "status": "Waiting",
            }

        # Step 2: parse + score all confirmed resumes concurrently — each
        # candidate is independent, so there's no need to wait for one before
        # starting the next. Capped at 5 workers to stay within free-tier RPM limits.
        valid_items = list(confirmed_texts.items())
        total = len(raw_items)  # includes excluded/failed files in the progress denominator
        completed = total - len(valid_items)  # extraction failures + excluded non-resumes already "done"

        def _format_eta(seconds: float) -> str:
            seconds = max(0, round(seconds))
            if seconds < 60:
                return f"{seconds}s"
            m, s = divmod(seconds, 60)
            return f"{m}m {s}s"

        screen_start = time.time()
        if valid_items:
            eta_ph.markdown("⏱️ Estimated time remaining: **calculating...**")

        with ThreadPoolExecutor(max_workers=min(12, max(1, len(valid_items)))) as executor:
            futures = {
                executor.submit(process_one, name, text): name
                for name, text in valid_items
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    # Persist immediately — this is what makes the candidate
                    # survive a closed browser / restarted app. Falls back to
                    # session-only storage if Supabase isn't configured or the
                    # save fails, but either way we get a stable "id" back so
                    # bookmarking/decisions/interview scores can reference it.
                    saved = add_candidate_record(result, job_role, job_details, job_id=st.session_state.selected_job_id)
                    st.session_state.candidates.append(saved)
                except Exception as e:
                    st.session_state.candidates.append({
                        "filename": name, "name": name, "raw_text": "",
                        "profile": {}, "score": {"overall_score": 0, "error": str(e)},
                        "screened_at": datetime.now().isoformat(), "status": "Waiting",
                    })
                completed += 1
                scored_count = completed - (total - len(valid_items))  # AI-scored items only, excludes pre-counted exclusions
                elapsed = time.time() - screen_start
                remaining_items = len(valid_items) - scored_count
                if scored_count > 0 and remaining_items > 0:
                    avg_per_item = elapsed / scored_count
                    eta_ph.markdown(f"⏱️ Estimated time remaining: **{_format_eta(avg_per_item * remaining_items)}**")
                elif remaining_items <= 0:
                    eta_ph.markdown("⏱️ Estimated time remaining: **0s** ✅")
                progress.progress(completed / max(total, 1), text=f"Screened {completed}/{total}...")

        progress.empty()
        _screen_logo_ph.empty()
        st.session_state.processed = True
        if valid_items:
            saved_note = " Saved to persistent history." if db.is_configured() else ""
            notify(f"Screened {len(valid_items)} candidate(s).{saved_note} See the Top Candidates section below.")
        else:
            st.warning("No files were recognized as resumes — nothing to screen. Check the excluded list above.")

    # Quick preview list
    if st.session_state.candidates:
        st.divider()
        st.markdown("#### Processed this session")
        for c in st.session_state.candidates:
            err = c["score"].get("error")
            if err:
                st.warning(f"**{c['filename']}** — failed to process: {err}")
            else:
                st.write(f"✅ **{c['name']}** — overall score: {c['score'].get('overall_score', '—')}/100")

    # ---- Top Candidates (ranked shortlist, moved here from Home) ----
    _valid_for_ranking = [c for c in st.session_state.candidates if not c["score"].get("error")]
    if _valid_for_ranking:
        st.divider()
        page_header("🏆", f"Top Candidates — Top {st.session_state.top_n}")
        st.caption("Ranked using your priority weighting — not a flat average. Mark candidates Selected or Rejected to track hiring decisions.")

        _ranked = sorted(_valid_for_ranking, key=lambda c: c["score"].get("overall_score", 0), reverse=True)

        filter_col, sort_col = st.columns([2, 1])
        with filter_col:
            min_score = st.slider("Minimum score filter", 0, 100, 0, key="screening_min_score")
        with sort_col:
            st.write("")

        search_term = st.session_state.global_search.strip().lower()
        if search_term:
            def _matches_search(c):
                if search_term in c["name"].lower():
                    return True
                skills = c["profile"].get("skills", []) or []
                return any(search_term in str(s).lower() for s in skills)
            ranked_filtered = [c for c in _ranked if _matches_search(c)]
            st.caption(f"🔍 Filtering by \"{st.session_state.global_search}\" — {len(ranked_filtered)} match(es)")
        else:
            ranked_filtered = _ranked

        shortlist = ranked_filtered[: st.session_state.top_n]
        for idx, c in enumerate(shortlist, start=1):
            score = c["score"].get("overall_score", 0)
            ai_score = c["score"].get("ai_overall_score", score)
            if score < min_score:
                continue

            color = "#1e8e3e" if score >= 75 else ("#b8860b" if score >= 50 else "#c0392b")
            bg = "#e6f4ea" if score >= 75 else ("#fdf3e2" if score >= 50 else "#fdecea")

            if idx == 1:
                avatar_bg, avatar_text = "linear-gradient(135deg,#FFD54F,#F9A825)", "🏆"
            elif idx == 2:
                avatar_bg, avatar_text = "linear-gradient(135deg,#CFD8DC,#90A4AE)", "🥈"
            elif idx == 3:
                avatar_bg, avatar_text = "linear-gradient(135deg,#D7B08C,#A9714F)", "🥉"
            else:
                avatar_bg, avatar_text = "linear-gradient(135deg,#38BDF8,#00668A)", str(idx)

            status_label = c.get("status", "Waiting")
            status_html = components.status_chip_html(status_label)

            with st.container():
                st.markdown(f"""
                <div class="candidate-card">
                    <div class="candidate-avatar" style="background:{avatar_bg};">{avatar_text}</div>
                    <div class="candidate-info">
                        <div class="candidate-rank-label">Rank #{idx}</div>
                        <div class="candidate-name">{c['name']}</div>
                    </div>
                    {status_html}
                    <span class="score-pill" style="background:{bg}; color:{color};">{score}/100</span>
                    {'<span class="score-pill" style="background:#FEF3C7; color:#92400E;">⚠️ Review</span>' if c["profile"].get("extraction_flags") else ''}
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View details — {c['name']}"):
                    profile = c["profile"]
                    score_data = c["score"]

                    if profile.get("extraction_flags"):
                        st.warning("**Extraction flags — worth a manual double-check:**\n\n" +
                                   "\n".join(f"- {flag}" for flag in profile["extraction_flags"]))

                    dec1, dec2, dec3 = st.columns([1, 1, 2])
                    with dec1:
                        if components.styled_button("✅ Mark Selected", key=f"screen_select_{idx}_{c.get('filename','')}", variant="success", width="stretch"):
                            update_candidate_record(c["id"], {"status": "Selected"})
                            st.rerun()
                    with dec2:
                        if components.styled_button("❌ Mark Rejected", key=f"screen_reject_{idx}_{c.get('filename','')}", variant="danger", width="stretch"):
                            update_candidate_record(c["id"], {"status": "Rejected"})
                            st.rerun()

                    cc1, cc2 = st.columns([1, 1])
                    with cc1:
                        st.markdown("**Summary**")
                        st.write(score_data.get("summary", "—"))
                        if abs(score - ai_score) >= 3:
                            st.caption(f"Priority-weighted score: {score}/100 · AI's unweighted baseline: {ai_score}/100")

                        st.markdown("**Matched Skills**")
                        components.chip_list(score_data.get("matched_skills", []), variant="skill", empty_text="None identified")

                        st.markdown("**Gaps**")
                        components.chip_list(score_data.get("gaps", []), variant="gap", empty_text="No significant gaps identified")

                    with cc2:
                        st.markdown("**Score Breakdown**")
                        breakdown = score_data.get("breakdown", {})
                        for k, v in breakdown.items():
                            st.write(f"{k.replace('_', ' ').title()}")
                            st.progress(min(max(v, 0), 100) / 100)

                        st.markdown("**Experience**")
                        st.write(profile.get("years_experience", "—"))

                        st.markdown("**Education**")
                        st.write(profile.get("education", "—"))

                    st.markdown("**Contact**")
                    st.write(f"{profile.get('email', '—')} | {profile.get('phone', '—')}")

# ============================================================
# PAGE — JOBS (Phase 5)
# ============================================================
elif page == "📋 Jobs":
    page_header("📋", "Job Management", "Create and manage job postings, then use them to screen candidates.")
    if not db.is_configured():
        st.caption("🗄️ Not connected to Supabase — jobs are saved for this session only. "
                    "Connect Supabase (sidebar) to keep job postings across sessions.")

    if st.session_state.jobs_view == "list":
        top_l, top_r = st.columns([4, 1.3])
        with top_l:
            st.caption(f"{len(get_jobs())} job posting(s)")
        with top_r:
            if st.button("➕ Create New Job", type="primary", width="stretch"):
                st.session_state.editing_job_id = None
                st.session_state.jobs_view = "form"
                st.rerun()

        show_archived = st.checkbox("Show archived jobs", value=False)
        jobs = get_jobs(include_archived=show_archived)
        if not show_archived:
            jobs = [j for j in jobs if j.get("status") == "active"]

        if not jobs:
            st.info("No job postings yet. Click **Create New Job** to add one.")
        else:
            for job in jobs:
                is_archived = job.get("status") == "archived"
                with st.container(border=True):
                    jc1, jc2 = st.columns([3, 1.6])
                    with jc1:
                        st.markdown(f"**{job['title']}**{'  🗄️ Archived' if is_archived else ''}")
                        meta = " · ".join(filter(None, [job.get("department"), job.get("location"), job.get("employment_type")]))
                        st.caption(meta or "—")
                        if job.get("required_skills"):
                            components.chip_list(job["required_skills"][:8], variant="keyword")
                    with jc2:
                        b1, b2, b3, b4 = st.columns(4)
                        if b1.button("✏️", key=f"edit_job_{job['id']}", help="Edit"):
                            st.session_state.editing_job_id = job["id"]
                            st.session_state.jobs_view = "form"
                            st.rerun()
                        if b2.button("📋", key=f"dup_job_{job['id']}", help="Duplicate"):
                            dup = {k: v for k, v in job.items() if k not in ("id", "created_at", "updated_at")}
                            dup["title"] = dup["title"] + " (Copy)"
                            _, saved_remote = create_job(dup)
                            if db.is_configured() and not saved_remote:
                                st.warning(f"Saved locally only — Supabase error: {db.get_last_error()}")
                            st.rerun()
                        if b3.button("↩️" if is_archived else "🗄️", key=f"arch_job_{job['id']}", help="Unarchive" if is_archived else "Archive"):
                            update_job_record(job["id"], {"status": "active" if is_archived else "archived"})
                            st.rerun()
                        if b4.button("🗑️", key=f"del_job_{job['id']}", help="Delete permanently"):
                            remove_job(job["id"])
                            st.rerun()

                    if st.button("Use this job for screening →", key=f"use_job_{job['id']}"):
                        st.session_state.selected_job_id = job["id"]
                        st.session_state.job_role = job["title"]
                        details_parts = [
                            job.get("description", ""),
                            ("Responsibilities: " + job["responsibilities"]) if job.get("responsibilities") else "",
                            ("Required skills: " + ", ".join(job.get("required_skills", []))) if job.get("required_skills") else "",
                        ]
                        st.session_state.job_details = "\n\n".join(p for p in details_parts if p)
                        st.session_state.current_page = "📤 Resume Screening"
                        st.rerun()

    else:  # jobs_view == "form"
        editing = st.session_state.editing_job_id is not None
        existing = next((j for j in get_jobs() if j["id"] == st.session_state.editing_job_id), {}) if editing else {}

        st.markdown(f"##### {'Edit Job' if editing else 'Create New Job'}")
        if st.button("← Back to job list"):
            st.session_state.jobs_view = "list"
            st.rerun()

        with st.form("job_form"):
            f1, f2 = st.columns(2)
            title = f1.text_input("Job Title *", value=existing.get("title", ""))
            department = f2.text_input("Department", value=existing.get("department", ""))
            f3, f4 = st.columns(2)
            location = f3.text_input("Location", value=existing.get("location", ""))
            salary_range = f4.text_input("Salary Range", value=existing.get("salary_range", ""))
            f5, f6 = st.columns(2)
            experience_level = f5.text_input("Experience Level", value=existing.get("experience_level", ""), placeholder="e.g. 2-4 years")
            employment_type = f6.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Internship"],
                                             index=["Full-time", "Part-time", "Contract", "Internship"].index(existing["employment_type"]) if existing.get("employment_type") in ["Full-time", "Part-time", "Contract", "Internship"] else 0)
            description = st.text_area("Job Description", value=existing.get("description", ""), height=100)
            responsibilities = st.text_area("Responsibilities", value=existing.get("responsibilities", ""), height=80)
            benefits = st.text_area("Benefits", value=existing.get("benefits", ""), height=80)
            skills_text = st.text_input("Required Skills (comma-separated)",
                                          value=", ".join(existing.get("required_skills", [])) if existing.get("required_skills") else "")
            deadline = st.date_input("Application Deadline", value=None)

            submitted = st.form_submit_button("💾 Save Job", type="primary")
            if submitted:
                if not title.strip():
                    st.error("Job title is required.")
                else:
                    job_data = {
                        "title": title.strip(), "department": department, "location": location,
                        "salary_range": salary_range, "experience_level": experience_level,
                        "employment_type": employment_type, "description": description,
                        "responsibilities": responsibilities, "benefits": benefits,
                        "required_skills": [s.strip() for s in skills_text.split(",") if s.strip()],
                        "deadline": deadline.isoformat() if deadline else None,
                    }
                    if editing:
                        update_job_record(st.session_state.editing_job_id, job_data)
                        st.success("Job updated.")
                    else:
                        _, saved_remote = create_job(job_data)
                        if db.is_configured() and not saved_remote:
                            st.warning(f"Job created, but saved locally only (this session) — Supabase error: {db.get_last_error()}")
                        else:
                            notify("Job created and saved to Supabase." if saved_remote else "Job created (this session only).")
                    st.session_state.jobs_view = "list"
                    st.rerun()

# ============================================================
# PAGE — HOME (command center: welcome, live stats, activity, charts, shortlist, reports)
# ============================================================
elif page == "🏠 Home":
    import plotly.graph_objects as go
    from collections import Counter

    all_candidates = st.session_state.candidates
    valid_candidates = [c for c in all_candidates if not c["score"].get("error")]
    ranked = sorted(valid_candidates, key=lambda c: c["score"].get("overall_score", 0), reverse=True)

    all_jobs = get_jobs(include_archived=True)
    active_jobs = [j for j in all_jobs if j.get("status") == "active"]
    completed_jobs = [j for j in all_jobs if j.get("status") == "archived"]

    interviews = get_interviews()
    scheduled_interviews = [i for i in interviews if i.get("status") == "Scheduled"]
    completed_interviews = [i for i in interviews if i.get("status") == "Completed"]
    awaiting_decision = [i for i in completed_interviews if i.get("interview_score") is None]

    selected_candidates = [c for c in valid_candidates if c.get("status") == "Selected"]
    rejected_candidates = [c for c in valid_candidates if c.get("status") == "Rejected"]
    shortlisted_candidates = [c for c in valid_candidates if c["score"].get("overall_score", 0) >= 70]

    avg_ats = round(sum(c["score"].get("overall_score", 0) for c in valid_candidates) / len(valid_candidates), 1) if valid_candidates else 0
    decided = len(selected_candidates) + len(rejected_candidates)
    success_rate = round(len(selected_candidates) / decided * 100) if decided else None

    # ---- Welcome / product description ----
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:26px 30px; margin-bottom:22px; box-shadow:0 1px 3px rgba(15,23,42,0.05);">
        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.6rem; color:#0B1C30; margin-bottom:6px;">
            Welcome back to ICD Platform
        </div>
        <div style="font-family:'Inter',sans-serif; font-size:0.95rem; color:#3E484F; max-width:820px; line-height:1.6;">
            The Intelligent Candidate Discovery Platform helps you discover, evaluate, and manage candidates
            faster by using AI to parse resumes, score candidates against your job requirements, surface
            interview-ready shortlists, and track every stage of recruitment in one place.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Stat cards ----
    st.markdown("""
    <style>
    .home-stat-card {
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:16px 18px;
        box-shadow:0 1px 3px rgba(15,23,42,0.05); transition: transform 0.15s ease, box-shadow 0.15s ease;
        height:100%;
    }
    .home-stat-card:hover { transform: translateY(-3px); box-shadow:0 6px 16px rgba(15,23,42,0.10); }
    .home-stat-icon { font-size:1.15rem; }
    .home-stat-label { font-size:0.74rem; color:#64748B; font-weight:600; margin-bottom:6px; }
    .home-stat-value { font-size:1.55rem; font-weight:800; color:#0B1C30; font-family:'Plus Jakarta Sans',sans-serif; }
    .home-stat-sub { font-size:0.7rem; color:#94A3B8; margin-top:2px; }
    </style>
    """, unsafe_allow_html=True)

    stat_cards = [
        ("📤", "Resumes Uploaded", len(all_candidates), None, "#00506B"),
        ("🧠", "Resumes Analyzed", len(valid_candidates), None, "#0EA5E9"),
        ("⭐", "Shortlisted (≥70)", len(shortlisted_candidates), None, "#F59E0B"),
        ("✅", "Selected", len(selected_candidates), None, "#22C55E"),
        ("❌", "Rejected", len(rejected_candidates), None, "#EF4444"),
        ("🗣️", "Interviewed", len(completed_interviews), None, "#8B5CF6"),
        ("⏳", "Pending Interviews", len(scheduled_interviews), None, "#F97316"),
        ("📋", "Active Job Openings", len(active_jobs), None, "#0891B2"),
        ("🏁", "Completed Recruitments", len(completed_jobs), "archived jobs", "#64748B"),
        ("🎯", "Avg ATS Score", f"{avg_ats}/100" if valid_candidates else "—", None, "#DB2777"),
        ("⏳", "Awaiting Decision", len(awaiting_decision), "interviewed, no score yet", "#7C3AED"),
        ("📈", "Hiring Success Rate", f"{success_rate}%" if success_rate is not None else "—",
         "based on Selected vs Rejected decisions" if success_rate is not None else "no decisions made yet", "#16A34A"),
    ]
    cols_per_row = 4
    for row_start in range(0, len(stat_cards), cols_per_row):
        row = stat_cards[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (icon, label, value, sub, color) in zip(cols, row):
            with col:
                st.markdown(components.stat_card_html(icon, label, value, sub, color), unsafe_allow_html=True)
        st.write("")

    st.divider()

    # ---- Recent activity timeline + charts ----
    timeline_col, chart_col = st.columns([1, 1.4])

    with timeline_col:
        page_header("🕒", "Recent Activity", "")
        events = []
        for c in all_candidates:
            if c.get("screened_at"):
                events.append((c["screened_at"], f"📤 Resume analyzed — **{c.get('name', c.get('filename','—'))}**"))
        for j in all_jobs:
            if j.get("created_at"):
                events.append((j["created_at"], f"📋 Job posted — **{j.get('title') or j.get('role') or 'Untitled role'}**"))
        for i in interviews:
            if i.get("scheduled_at"):
                verb = {"Scheduled": "Interview scheduled", "Completed": "Interview completed", "Cancelled": "Interview cancelled"}.get(i.get("status"), "Interview updated")
                events.append((i["scheduled_at"], f"🗣️ {verb} — **{i.get('candidate_name','—')}**"))
        events.sort(key=lambda e: e[0], reverse=True)

        if not events:
            st.info("No activity yet — upload resumes, post a job, or schedule an interview to get started.")
        else:
            for ts, label in events[:12]:
                try:
                    ts_label = datetime.fromisoformat(ts).strftime("%b %d, %I:%M %p")
                except Exception:
                    ts_label = ts
                st.markdown(f"""
                <div style="border-left:2px solid #38BDF8; padding:2px 0 12px 14px; margin-left:4px;">
                    <div style="font-size:0.85rem; color:#0B1C30;">{label}</div>
                    <div style="font-size:0.72rem; color:#94A3B8;">{ts_label}</div>
                </div>
                """, unsafe_allow_html=True)

    with chart_col:
        page_header("📊", "Recruitment Overview", "")
        if not valid_candidates:
            st.info("Charts will appear here once you've screened at least one candidate.")
        else:
            scores = [c["score"].get("overall_score", 0) for c in valid_candidates]

            tiers = {"Strong Fit (≥75)": 0, "Good Fit (50-74)": 0, "Weak Fit (<50)": 0}
            for s in scores:
                if s >= 75:
                    tiers["Strong Fit (≥75)"] += 1
                elif s >= 50:
                    tiers["Good Fit (50-74)"] += 1
                else:
                    tiers["Weak Fit (<50)"] += 1
            fig_funnel = go.Figure(go.Funnel(
                y=list(tiers.keys()), x=list(tiers.values()),
                marker={"color": ["#22C55E", "#F59E0B", "#EF4444"]},
            ))
            fig_funnel.update_layout(title="Hiring Funnel (by fit tier)", height=280,
                                      margin=dict(l=10, r=10, t=40, b=10),
                                      plot_bgcolor="white", paper_bgcolor="white",
                                      font=dict(family="Inter, sans-serif", color="#0F172A"))
            st.plotly_chart(fig_funnel, width="stretch")

            fig_hist = go.Figure(go.Histogram(x=scores, nbinsx=10, marker_color="#00668A"))
            fig_hist.update_layout(title="ATS Score Distribution", height=280,
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    plot_bgcolor="white", paper_bgcolor="white",
                                    font=dict(family="Inter, sans-serif", color="#0F172A"),
                                    xaxis_title="Overall Score", yaxis_title="Candidates")
            st.plotly_chart(fig_hist, width="stretch")

    all_skills = []
    for c in valid_candidates:
        all_skills.extend(c["profile"].get("skills", []) or [])
    top_skills = Counter(s.strip() for s in all_skills if s and s.strip()).most_common(10)
    edu_counts = Counter((c["profile"].get("education") or "Unknown") for c in valid_candidates)

    chart_col2, chart_col3 = st.columns(2)
    with chart_col2:
        if top_skills:
            fig_skills = go.Figure(go.Bar(
                x=[cnt for _, cnt in top_skills][::-1], y=[s for s, _ in top_skills][::-1],
                orientation="h", marker_color="#38BDF8",
            ))
            fig_skills.update_layout(title="Top 10 Skills Across Candidates", height=320,
                                      margin=dict(l=10, r=10, t=40, b=10),
                                      plot_bgcolor="white", paper_bgcolor="white",
                                      font=dict(family="Inter, sans-serif", color="#0F172A"))
            st.plotly_chart(fig_skills, width="stretch")
        elif valid_candidates:
            st.info("No skill data available yet.")
    with chart_col3:
        if edu_counts:
            top_edu = edu_counts.most_common(8)
            fig_edu = go.Figure(go.Pie(labels=[e for e, _ in top_edu], values=[cnt for _, cnt in top_edu], hole=0.45))
            fig_edu.update_layout(title="Education Distribution", height=320,
                                   margin=dict(l=10, r=10, t=40, b=10),
                                   plot_bgcolor="white", paper_bgcolor="white",
                                   font=dict(family="Inter, sans-serif", color="#0F172A"))
            st.plotly_chart(fig_edu, width="stretch")



elif page == "👥 Candidates":
    all_candidates = [c for c in st.session_state.candidates if not c["score"].get("error")]

    def _ckey(c):
        return c.get("id") or c["filename"]

    def _tier_badge(score):
        if score >= 75:
            return "Strong Fit", "#22C55E", "#DCFCE7"
        elif score >= 50:
            return "Good Fit", "#F59E0B", "#FEF3C7"
        else:
            return "Weak Fit", "#EF4444", "#FEE2E2"

    if not all_candidates:
        st.info("No screened candidates yet. Go to **Resume Screening** first.")
    else:
        # ---------------- Top bar: view switch + compare tray ----------------
        top_l, top_r = st.columns([3, 2])
        with top_l:
            page_header("👥", "Candidates")
            st.caption(f"{len(all_candidates)} screened this session")
        with top_r:
            if st.session_state.compare_list:
                if st.button(f"⚖️ Compare Selected ({len(st.session_state.compare_list)})", type="primary", width="stretch"):
                    st.session_state.candidates_view = "compare"
                    st.rerun()

        # ---------------- PROFILE VIEW ----------------
        if st.session_state.candidates_view == "profile" and st.session_state.selected_candidate_key:
            selected = next((c for c in all_candidates if _ckey(c) == st.session_state.selected_candidate_key), None)
            if not selected:
                st.session_state.candidates_view = "grid"
                st.rerun()
            else:
                if st.button("← Back to all candidates"):
                    st.session_state.candidates_view = "grid"
                    st.rerun()

                p, s = selected["profile"], selected["score"]
                score = s.get("overall_score", 0)
                tier_label, tier_color, tier_bg = _tier_badge(score)

                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:18px; margin:14px 0 18px;">
                    <div style="width:64px; height:64px; border-radius:50%; background:linear-gradient(135deg,#38BDF8,#00668A);
                                color:#fff; display:flex; align-items:center; justify-content:center; font-size:1.5rem; font-weight:800;">
                        {(selected['name'][:1] or '?').upper()}
                    </div>
                    <div>
                        <div style="font-size:1.5rem; font-weight:800; color:var(--text); font-family:'Plus Jakarta Sans',sans-serif;">{selected['name']}</div>
                        <div style="color:var(--text-secondary); font-size:0.9rem;">{p.get('years_experience','—')} · {p.get('education','—')}</div>
                    </div>
                    <div style="margin-left:auto; text-align:right;">
                        <span style="background:{tier_bg}; color:{tier_color}; font-weight:800; border-radius:20px; padding:5px 16px; font-size:0.85rem;">{tier_label}</span>
                        <div style="font-size:1.8rem; font-weight:800; color:var(--primary-dark); font-family:'Plus Jakarta Sans',sans-serif;">{score}/100</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                extraction_flags = p.get("extraction_flags", [])
                if extraction_flags:
                    with st.expander(f"⚠️ {len(extraction_flags)} field(s) may need manual review", expanded=False):
                        st.caption("The AI extraction flagged these — worth a quick double-check against the resume:")
                        for flag in extraction_flags:
                            st.markdown(f"- {flag}")

                tabs = st.tabs(["Overview", "Resume", "Skills", "ATS Analysis", "Experience", "Education", "Interview Prep", "Notes"])

                with tabs[0]:
                    st.markdown("**Recruiter Summary**")
                    st.write(s.get("summary", "—"))
                    bcol1, bcol2, bcol3 = st.columns(3)
                    breakdown = s.get("breakdown", {})
                    bcol1.metric("Skills Match", f"{breakdown.get('skills_match', '—')}/100")
                    bcol2.metric("Experience Fit", f"{breakdown.get('experience_fit', '—')}/100")
                    bcol3.metric("Education Fit", f"{breakdown.get('education_fit', '—')}/100")
                    st.markdown("**Contact**")
                    st.write(f"{p.get('email', '—')} · {p.get('phone', '—')}")

                with tabs[1]:
                    st.markdown("**Extracted Resume Text**")
                    st.text_area("Raw resume text", selected.get("raw_text", ""), height=350, label_visibility="collapsed")

                with tabs[2]:
                    st.markdown("**Matched Skills** (relevant to the job requirements)")
                    components.chip_list(s.get("matched_skills", []), variant="skill", empty_text="None identified")
                    st.markdown("**All Skills Listed on Resume**")
                    components.chip_list(p.get("skills", []), variant="keyword", empty_text="None extracted")
                    st.markdown("**Gaps vs. This Role**")
                    components.chip_list(s.get("gaps", []), variant="gap", empty_text="No significant gaps identified")

                with tabs[3]:
                    st.caption("Formatting/Structure/Readability are computed instantly from the resume text itself (free, no API call). "
                                "Keyword coverage, grammar, and suggestions need judgment, so they're generated on demand below.")

                    local_metrics = compute_local_ats_metrics(selected.get("raw_text", ""))
                    key = _ckey(selected)
                    ats_cache_key = f"ats_{key}"

                    if st.button("🔬 Run Full ATS Analysis", type="primary", key=f"ats_btn_{key}"):
                        with st.spinner("Analyzing keyword coverage and writing quality..."):
                            try:
                                jd_text = f"Job Role: {st.session_state.job_role}\n\n{st.session_state.job_details}"
                                st.session_state[ats_cache_key] = analyze_ats_ai(selected.get("raw_text", ""), jd_text)
                            except Exception as e:
                                st.error(f"ATS analysis failed: {e}")

                    ai_ats = st.session_state.get(ats_cache_key)

                    def _gauge(col, label, value, color):
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=value,
                            title={"text": label, "font": {"size": 13}},
                            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                   "bgcolor": "white", "borderwidth": 1, "bordercolor": "#E2E8F0"},
                        ))
                        fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10))
                        col.plotly_chart(fig, width="stretch")

                    g1, g2, g3 = st.columns(3)
                    _gauge(g1, "Structure", local_metrics["structure_score"], "#00668A")
                    _gauge(g2, "Formatting", local_metrics["formatting_score"], "#38BDF8")
                    _gauge(g3, "Readability", local_metrics["readability_score"], "#3B82F6")

                    st.caption(f"Sections detected: {', '.join(local_metrics['sections_found']) or 'none'} · "
                                f"{local_metrics['word_count']} words · avg {local_metrics['avg_sentence_length']} words/sentence")

                    if ai_ats:
                        g4, g5 = st.columns(2)
                        _gauge(g4, "Keyword Match", ai_ats.get("keyword_score", 0), "#22C55E")
                        _gauge(g5, "Grammar", ai_ats.get("grammar_score", 0), "#F59E0B")

                        compat = ai_ats.get("ats_compatibility", "—")
                        compat_color = {"High": "#22C55E", "Medium": "#F59E0B", "Low": "#EF4444"}.get(compat, "#64748B")
                        st.markdown(f"""
                        <div style="background:{compat_color}1A; border:1px solid {compat_color}; border-radius:10px; padding:10px 16px; margin:10px 0;">
                            <b style="color:{compat_color};">ATS Compatibility: {compat}</b><br>
                            <span style="font-size:0.85rem; color:var(--text);">{ai_ats.get('compatibility_reason','')}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("**Missing Keywords**")
                        components.chip_list(ai_ats.get("missing_keywords", []), variant="gap", empty_text="None — good keyword coverage")

                        st.markdown("**Improvement Suggestions**")
                        for sug in ai_ats.get("suggestions", []):
                            st.write(f"- {sug}")

                with tabs[4]:
                    st.markdown("**Past Roles**")
                    for role in p.get("past_roles", []) or ["—"]:
                        st.write(f"- {role}")
                    st.markdown("**Years of Experience**")
                    st.write(p.get("years_experience", "—"))

                with tabs[5]:
                    st.write(p.get("education", "—"))

                with tabs[6]:
                    render_interview_prep(selected, key_prefix=f"profile_{_ckey(selected)}")

                with tabs[7]:
                    note_val = st.text_area("Recruiter notes for this candidate", value=selected.get("notes", ""), height=150)
                    if st.button("💾 Save Notes"):
                        update_candidate_record(selected["id"], {"notes": note_val})
                        st.success("Notes saved.")

        # ---------------- COMPARE VIEW ----------------
        elif st.session_state.candidates_view == "compare":
            if st.button("← Back to all candidates"):
                st.session_state.candidates_view = "grid"
                st.rerun()

            compare_candidates = [c for c in all_candidates if _ckey(c) in st.session_state.compare_list]
            if len(compare_candidates) < 2:
                st.warning("Select at least 2 candidates from the grid to compare.")
            else:
                page_header("⚖️", f"Comparing {len(compare_candidates)} Candidates")
                cols = st.columns(len(compare_candidates))
                for col, c in zip(cols, compare_candidates):
                    p, s = c["profile"], c["score"]
                    with col:
                        st.markdown(f"**{c['name']}**")
                        st.metric("Overall Score", f"{s.get('overall_score',0)}/100")
                        breakdown = s.get("breakdown", {})
                        st.write(f"Skills: {breakdown.get('skills_match','—')}/100")
                        st.write(f"Experience: {breakdown.get('experience_fit','—')}/100")
                        st.write(f"Education: {breakdown.get('education_fit','—')}/100")
                        st.caption(p.get("years_experience", "—"))
                        st.caption(p.get("education", "—"))
                        st.markdown("**Matched Skills**")
                        components.chip_list(s.get("matched_skills", []), variant="skill")
                        st.markdown("**Gaps**")
                        components.chip_list(s.get("gaps", []), variant="gap")

                # Radar chart comparing breakdown dimensions across candidates — real data.
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for c in compare_candidates:
                        b = c["score"].get("breakdown", {})
                        fig.add_trace(go.Scatterpolar(
                            r=[b.get("skills_match", 0), b.get("experience_fit", 0), b.get("education_fit", 0), b.get("skills_match", 0)],
                            theta=["Skills", "Experience", "Education", "Skills"],
                            fill="toself", name=c["name"],
                        ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                       height=420, margin=dict(l=40, r=40, t=40, b=40))
                    st.plotly_chart(fig, width="stretch")
                except Exception:
                    pass

                if st.button("Clear comparison selection"):
                    st.session_state.compare_list = []
                    st.rerun()

        # ---------------- GRID VIEW (default) ----------------
        else:
            search_term = st.session_state.global_search.strip().lower()
            filtered = all_candidates
            if search_term:
                def _match(c):
                    if search_term in c["name"].lower():
                        return True
                    return any(search_term in str(sk).lower() for sk in c["profile"].get("skills", []))
                filtered = [c for c in all_candidates if _match(c)]

            show_rejected = st.checkbox("Show rejected candidates", value=False)
            if not show_rejected:
                filtered = [c for c in filtered if c.get("status") != "Rejected"]

            filtered = sorted(filtered, key=lambda c: c["score"].get("overall_score", 0), reverse=True)

            n_cols = 3
            rows = [filtered[i:i+n_cols] for i in range(0, len(filtered), n_cols)]
            for row in rows:
                cols = st.columns(n_cols)
                for col, c in zip(cols, row):
                    key = _ckey(c)
                    p, s = c["profile"], c["score"]
                    score = s.get("overall_score", 0)
                    tier_label, tier_color, tier_bg = _tier_badge(score)
                    is_bookmarked = bool(c.get("bookmarked"))
                    is_rejected = c.get("status") == "Rejected"
                    is_in_compare = key in st.session_state.compare_list

                    with col:
                        with st.container(border=True):
                            st.markdown(f"""
                            <div style="display:flex; align-items:center; gap:10px;">
                                <div style="width:44px; height:44px; border-radius:50%; background:linear-gradient(135deg,#38BDF8,#00668A);
                                            color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800;">
                                    {(c['name'][:1] or '?').upper()}
                                </div>
                                <div>
                                    <div style="font-weight:800; color:var(--text); font-size:1.02rem;">{c['name']}{'  🔖' if is_bookmarked else ''}</div>
                                    <div style="color:var(--text-secondary); font-size:0.78rem;">{p.get('years_experience','—')}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown(f"""
                            <div style="margin:10px 0;">
                                <span style="background:{tier_bg}; color:{tier_color}; font-weight:800; border-radius:14px; padding:3px 12px; font-size:0.75rem;">{tier_label}</span>
                                <span style="font-weight:800; color:var(--primary-dark); font-size:1.1rem; float:right;">{score}/100</span>
                            </div>
                            """, unsafe_allow_html=True)

                            if c.get("interview_score") is not None:
                                _avg = round((score + c["interview_score"]) / 2)
                                st.caption(f"🗣️ Interview {c['interview_score']}/100 · Avg {_avg}/100")

                            extraction_flags = p.get("extraction_flags", [])
                            if extraction_flags:
                                with st.popover(f"⚠️ Needs review ({len(extraction_flags)})", use_container_width=True):
                                    st.caption("The AI extraction flagged these for a manual double-check:")
                                    for flag in extraction_flags:
                                        st.markdown(f"- {flag}")

                            st.caption(p.get("education", "—"))
                            top_skills = (p.get("skills", []) or [])[:4]
                            if top_skills:
                                components.chip_list(top_skills, variant="skill")

                            if is_rejected:
                                st.caption("🚫 Marked rejected")

                            b1, b2, b3, b4 = st.columns(4)
                            with b1:
                                if st.button("👁️", key=f"view_{key}", help="View full profile", width="stretch"):
                                    st.session_state.selected_candidate_key = key
                                    st.session_state.candidates_view = "profile"
                                    st.rerun()
                            with b2:
                                if st.button("⚖️" if not is_in_compare else "✅", key=f"cmp_{key}", help="Add/remove from comparison", width="stretch"):
                                    if is_in_compare:
                                        st.session_state.compare_list.remove(key)
                                    elif len(st.session_state.compare_list) < 4:
                                        st.session_state.compare_list.append(key)
                                    st.rerun()
                            with b3:
                                if st.button("🔖" if not is_bookmarked else "★", key=f"bm_{key}", help="Bookmark", width="stretch"):
                                    update_candidate_record(c["id"], {"bookmarked": not is_bookmarked})
                                    st.rerun()
                            with b4:
                                if st.button("↩️" if is_rejected else "🚫", key=f"rej_{key}", help="Un-reject" if is_rejected else "Reject", width="stretch"):
                                    update_candidate_record(c["id"], {"status": "Waiting" if is_rejected else "Rejected"})
                                    st.rerun()



# ============================================================
# PAGE — INTERVIEW (Prep & Schedule + Tracking, combined under one nav item)
# ============================================================
elif page == "🗣️ Interview":
    page_header("🗣️", "Interview", "Generate questions, schedule interviews, and track their status.")
    interview_tab_prep, interview_tab_track = st.tabs(["✨ Prep & Schedule", "📅 Track Interviews"])

    with interview_tab_prep:
        valid_candidates = [c for c in st.session_state.candidates if not c["score"].get("error")]

        if not valid_candidates:
            st.info("No screened candidates yet. Go to **Resume Screening** first.")
        else:
            # Tracked by stable candidate id (not a derived label string) so the
            # selection survives reruns even when the underlying candidate list
            # gets refreshed/reordered from the database — previously, any tiny
            # change there silently reset the picker back to the first candidate.
            candidate_by_id = {c["id"]: c for c in valid_candidates}
            candidate_ids = list(candidate_by_id.keys())

            def _format_candidate(cid):
                c = candidate_by_id[cid]
                return f"{c['name']} — {c['score'].get('overall_score', '—')}/100"

            if st.session_state.get("interview_prep_candidate_id") not in candidate_ids:
                st.session_state.interview_prep_candidate_id = candidate_ids[0]

            selected_id = st.selectbox(
                "Select a candidate", candidate_ids, format_func=_format_candidate,
                key="interview_prep_candidate_id",
            )
            selected = candidate_by_id[selected_id]
            selected_name = selected["name"]

            cap_col, role_col = st.columns([3, 2])
            cap_col.caption(f"Fit score: {selected['score'].get('overall_score', '—')}/100")
            _role = (selected.get("job_role") or "").strip()
            if _role:
                role_col.markdown(f'<div style="text-align:right;"><span class="score-pill" style="background:#EEF2FF; color:#4338CA;">💼 {_role}</span></div>', unsafe_allow_html=True)

            with st.expander("📅 Schedule an interview for this candidate"):
                ic1, ic2 = st.columns(2)
                interview_type = ic1.selectbox("Interview Type", ["Technical", "Behavioral", "HR", "Coding", "Scenario"], key="sched_type")
                interview_date = ic2.date_input("Date", key="sched_date")
                ic3, ic4 = st.columns(2)
                interview_time = ic3.time_input("Time", key="sched_time")
                interview_notes = ic4.text_input("Notes (optional)", key="sched_notes")
                if st.button("📅 Confirm Schedule", key="confirm_schedule"):
                    scheduled_dt = datetime.combine(interview_date, interview_time)
                    _, saved_remote = create_interview({
                        "candidate_name": selected_name,
                        "job_id": st.session_state.selected_job_id,
                        "job_role": st.session_state.job_role,
                        "interview_type": interview_type,
                        "scheduled_at": scheduled_dt.isoformat(),
                        "status": "Scheduled",
                        "notes": interview_notes,
                    })
                    if db.is_configured() and not saved_remote:
                        st.warning(f"Interview saved locally only (this session) — Supabase error: {db.get_last_error()}")
                    else:
                        notify(f"Interview scheduled for {selected_name} on {scheduled_dt.strftime('%b %d, %Y at %I:%M %p')}. "
                                    "View it on the **Track Interviews** tab.")

            render_interview_prep(selected, key_prefix=f"prep_{selected.get('id', selected_name)}")

    with interview_tab_track:
        if not db.is_configured():
            st.caption("🗄️ Not connected to Supabase — interviews are saved for this session only.")

        interviews = get_interviews()
        if not interviews:
            st.info("No interviews scheduled yet. Go to the **Prep & Schedule** tab, pick a candidate, and use "
                    "\"Schedule an interview for this candidate\" to add one.")
        else:
            status_filter = st.radio("Filter", ["All", "Scheduled", "Completed", "Cancelled"], horizontal=True)
            filtered = interviews if status_filter == "All" else [i for i in interviews if i.get("status") == status_filter]
            filtered = sorted(filtered, key=lambda i: i.get("scheduled_at", ""))

            for idx, interview in enumerate(filtered):
                try:
                    dt = datetime.fromisoformat(interview["scheduled_at"])
                    dt_label = dt.strftime("%b %d, %Y · %I:%M %p")
                except Exception:
                    dt_label = interview.get("scheduled_at", "—")

                with st.container(border=True):
                    ic1, ic2, ic3 = st.columns([2.5, 1.5, 1.5])
                    with ic1:
                        st.markdown(f"**{interview.get('candidate_name','—')}**")
                        st.caption(f"{interview.get('interview_type','—')} · {interview.get('job_role') or '—'}")
                    with ic2:
                        st.write(f"🗓️ {dt_label}")
                    with ic3:
                        components.status_chip(interview.get("status", "—"))

                    if interview.get("notes"):
                        st.caption(f"Notes: {interview['notes']}")

                    # idx makes these keys unique even when a locally-generated ID
                    # happens to collide with a Supabase-assigned ID (both start at 1).
                    if interview.get("status") == "Scheduled":
                        a1, a2, a3 = st.columns(3)
                        with a1:
                            if components.styled_button("✅ Mark Completed", key=f"complete_{idx}_{interview['id']}", variant="success", width="stretch"):
                                update_interview_record(interview["id"], {"status": "Completed"})
                                st.rerun()
                        with a2:
                            if components.styled_button("❌ Cancel", key=f"cancel_{idx}_{interview['id']}", variant="danger", width="stretch"):
                                update_interview_record(interview["id"], {"status": "Cancelled"})
                                st.rerun()
                    elif interview.get("status") == "Completed":
                        score_val = interview.get("interview_score")
                        unlock_key = f"iscore_unlocked_{idx}_{interview['id']}"
                        confirm_key = f"iscore_unlock_confirm_{idx}_{interview['id']}"

                        if score_val is None:
                            # Not scored yet — open for entry, saved only on explicit confirmation.
                            score_input = st.number_input(
                                "Interview Score (out of 100)", min_value=0, max_value=100,
                                value=50, step=1, key=f"iscore_{idx}_{interview['id']}",
                            )
                            if st.button("🔒 Save & Lock", key=f"iscore_lock_{idx}_{interview['id']}", type="primary"):
                                if update_interview_record(interview["id"], {"interview_score": score_input}):
                                    for cand in st.session_state.candidates:
                                        if cand.get("name") == interview.get("candidate_name"):
                                            update_candidate_record(cand["id"], {"interview_score": score_input})
                                            cand["interview_score"] = score_input
                                            break
                                    st.rerun()
                                else:
                                    st.error(f"Couldn't save the score: {db.get_last_error() or 'unknown error'}. "
                                             f"If you haven't run migration_full_persistence.sql on your Supabase "
                                             f"project yet, the `interviews` table may be missing the "
                                             f"`interview_score` column — that's the most common cause.")

                        elif st.session_state.get(unlock_key):
                            # Explicitly unlocked — editable again, re-locks automatically once saved.
                            score_input = st.number_input(
                                "Interview Score (out of 100)", min_value=0, max_value=100,
                                value=score_val, step=1, key=f"iscore_{idx}_{interview['id']}",
                            )
                            uc1, uc2 = st.columns(2)
                            if uc1.button("💾 Save & Re-lock", key=f"iscore_save_{idx}_{interview['id']}", type="primary", width="stretch"):
                                if update_interview_record(interview["id"], {"interview_score": score_input}):
                                    for cand in st.session_state.candidates:
                                        if cand.get("name") == interview.get("candidate_name"):
                                            update_candidate_record(cand["id"], {"interview_score": score_input})
                                            cand["interview_score"] = score_input
                                            break
                                    st.session_state[unlock_key] = False
                                    st.rerun()
                                else:
                                    st.error(f"Couldn't save the score: {db.get_last_error() or 'unknown error'}")
                            if uc2.button("Cancel", key=f"iscore_cancel_{idx}_{interview['id']}", width="stretch"):
                                st.session_state[unlock_key] = False
                                st.rerun()

                        else:
                            # Locked — score already recorded, shown read-only.
                            st.markdown(f"""
                            <div style="display:flex; align-items:center; gap:10px; background:#F1F5F9; border:1px solid #E2E8F0;
                                        border-radius:10px; padding:10px 16px; margin:6px 0;">
                                <span style="font-size:1.1rem;">🔒</span>
                                <span style="font-weight:800; color:#0B1C30;">Interview Score: {score_val}/100</span>
                                <span style="color:#64748B; font-size:0.78rem; margin-left:auto;">Locked — can't be accidentally changed</span>
                            </div>
                            """, unsafe_allow_html=True)

                            if not st.session_state.get(confirm_key):
                                if st.button("🔓 Unlock to edit", key=f"iscore_unlock_btn_{idx}_{interview['id']}"):
                                    st.session_state[confirm_key] = True
                                    st.rerun()
                            else:
                                st.warning("This will let you change a recorded interview score. Only unlock if you're sure.")
                                wc1, wc2 = st.columns(2)
                                if wc1.button("✅ Yes, unlock", key=f"iscore_unlock_yes_{idx}_{interview['id']}", width="stretch"):
                                    st.session_state[unlock_key] = True
                                    st.session_state[confirm_key] = False
                                    st.rerun()
                                if wc2.button("Cancel", key=f"iscore_unlock_no_{idx}_{interview['id']}", width="stretch"):
                                    st.session_state[confirm_key] = False
                                    st.rerun()

elif page == "🤖 AI Insights":
    valid_candidates = [c for c in st.session_state.candidates if not c["score"].get("error")]

    page_header("🤖", "AI Recruiting Assistant")
    if valid_candidates:
        st.caption(
            f"Ask about your {len(valid_candidates)} screened candidate(s) — answers are grounded "
            "strictly in the actual screening data. You can also ask how the app works, or for "
            "recruiting/job-description ideas."
        )
    else:
        st.caption(
            "No candidates screened yet — that's fine, ask away. I can explain how the app works, "
            "or help with job-description wording, priority-weighting choices, and interview strategy. "
            "Once you screen some resumes, I can also answer questions about them."
        )

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    placeholder = (
        f"Ask about your {len(valid_candidates)} screened candidate(s), or how the app works..."
        if valid_candidates else
        "Ask how the app works, or for recruiting/job-description ideas..."
    )
    user_q = st.chat_input(placeholder)
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with logo_loading("Thinking..."):
                try:
                    answer = ask_assistant(
                        user_q, valid_candidates,
                        st.session_state.job_role, st.session_state.job_details,
                        st.session_state.chat_history,
                    )
                except Exception as e:
                    answer = f"Sorry, I couldn't get an answer right now: {e}"
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

# ============================================================
# PAGE — SHORTLISTED
# ============================================================
elif page == "⭐ Shortlisted":
    page_header("⭐", "Shortlisted", "Candidates with a combined score ≥ 70 — ATS score alone before an interview, "
                                     "the average of ATS + interview score once interviewed.")

    _valid = [c for c in st.session_state.candidates if not c["score"].get("error")]

    def _combined_score(c):
        ats = c["score"].get("overall_score", 0)
        iscore = c.get("interview_score")
        if iscore is None:
            return ats  # not interviewed yet — go on ATS score alone
        return round((ats + iscore) / 2)  # interviewed — average of both

    def _passes_shortlist(c):
        return _combined_score(c) >= 70

    _shortlisted = sorted(
        [c for c in _valid if _passes_shortlist(c)],
        key=_combined_score, reverse=True,
    )

    if not _shortlisted:
        st.info("No shortlisted candidates yet — candidates appear here once their combined score reaches 70.")
    else:
        outcome_options = ["Waiting", "Selected", "Rejected"]
        filter_choice = st.radio("Filter", ["All"] + outcome_options, horizontal=True, key="shortlisted_filter")
        rows = _shortlisted if filter_choice == "All" else [c for c in _shortlisted if c.get("status", "Waiting") == filter_choice]

        st.caption(f"{len(rows)} of {len(_shortlisted)} shortlisted candidate(s)")

        # Grouped by job role so it's clear which posting each candidate belongs to,
        # instead of one flat mixed list.
        _groups: dict[str, list] = {}
        for c in rows:
            _role = (c.get("job_role") or "").strip() or "Unspecified Role"
            _groups.setdefault(_role, []).append(c)

        for _role in sorted(_groups.keys()):
            _group_rows = _groups[_role]
            st.markdown(f"#### 💼 {_role} <span style='color:#94A3B8; font-weight:400; font-size:0.9rem;'>({len(_group_rows)})</span>", unsafe_allow_html=True)

            for i, c in enumerate(_group_rows):
                score = c["score"].get("overall_score", 0)
                combined = _combined_score(c)
                color = "#1e8e3e" if score >= 75 else "#b8860b"
                bg = "#e6f4ea" if score >= 75 else "#fdf3e2"
                current_status = c.get("status", "Waiting")
                iscore = c.get("interview_score")
                iscore_pill = f'<span class="score-pill" style="background:#EDE9FE; color:#6D28D9;">Interview {iscore}/100</span>' if iscore is not None else ""
                avg_color = "#1e8e3e" if combined >= 75 else "#b8860b"
                avg_bg = "#e6f4ea" if combined >= 75 else "#fdf3e2"
                avg_pill = (
                    f'<span class="score-pill" style="background:{avg_bg}; color:{avg_color}; font-weight:800;">Avg {combined}/100</span>'
                    if iscore is not None else ""
                )

                with st.container():
                    st.markdown(f"""
                    <div class="candidate-card">
                        <div class="candidate-avatar" style="background:linear-gradient(135deg,#38BDF8,#00668A);">⭐</div>
                        <div class="candidate-info">
                            <div class="candidate-rank-label">Shortlisted</div>
                            <div class="candidate-name">{c['name']}</div>
                        </div>
                        {components.status_chip_html(current_status)}
                        <span class="score-pill" style="background:{bg}; color:{color};">ATS {score}/100</span>
                        {iscore_pill}
                        {avg_pill}
                    </div>
                    """, unsafe_allow_html=True)

                    sc1, sc2 = st.columns([1, 3])
                    with sc1:
                        new_status = st.selectbox(
                            "Decision", outcome_options, index=outcome_options.index(current_status) if current_status in outcome_options else 0,
                            key=f"shortlist_status_{_role}_{i}_{c.get('filename','')}", label_visibility="collapsed",
                        )
                        if new_status != current_status:
                            update_candidate_record(c["id"], {"status": new_status})
                            st.rerun()
                    with sc2:
                        skills = c["profile"].get("skills", []) or []
                        if skills:
                            components.chip_list(skills[:6], variant="skill")

            st.markdown("---")

# ============================================================
# PAGE — REPORTS (extracted from Home so it has its own sidebar entry)
# ============================================================
elif page == "📄 Reports":
    page_header("📄", "Reports", "Export candidate, shortlist, and interview data as PDF, Excel, or CSV.")

    report_valid_candidates = [c for c in st.session_state.candidates if not c["score"].get("error")]
    report_interviews = get_interviews()
    _report_job_skills = extract_keywords(f"{st.session_state.job_role} {st.session_state.job_details}") if st.session_state.job_details else []

    report_tabs = st.tabs(["Candidate Report", "Shortlist Report", "✅ Selected Candidates", "Interview Report", "📄 Offer Letter"])

    with report_tabs[0]:
        if not report_valid_candidates:
            st.info("No screened candidates yet.")
        else:
            names = [c["name"] for c in report_valid_candidates]
            pick = st.selectbox("Select a candidate", names, key="report_candidate_pick")
            picked_c = next(c for c in report_valid_candidates if c["name"] == pick)
            if st.button("📄 Generate PDF", key="gen_candidate_pdf"):
                pdf_bytes = reports.build_candidate_report_pdf(
                    picked_c, st.session_state.job_role, st.session_state.job_details,
                    _report_job_skills, st.session_state.weights,
                )
                st.download_button("⬇️ Download Candidate Report (PDF)", data=pdf_bytes,
                                     file_name=f"candidate_report_{pick.replace(' ','_')}.pdf", mime="application/pdf")

    with report_tabs[1]:
        if not report_valid_candidates:
            st.info("No screened candidates yet.")
        else:
            st.caption(f"{len(report_valid_candidates)} candidate(s) in the current shortlist")
            fmt = st.radio("Format", ["PDF", "Excel", "CSV"], horizontal=True, key="shortlist_fmt")
            if st.button("📄 Generate Report", key="gen_shortlist"):
                if fmt == "PDF":
                    data = reports.build_shortlist_report_pdf(
                        report_valid_candidates, st.session_state.job_role, st.session_state.weights,
                        st.session_state.job_details, _report_job_skills,
                    )
                    st.download_button("⬇️ Download Shortlist Report (PDF)", data=data, file_name="shortlist_report.pdf", mime="application/pdf")
                else:
                    df = reports.candidates_to_dataframe(report_valid_candidates)
                    if fmt == "Excel":
                        data = reports.df_to_excel_bytes(df, sheet_name="Shortlist")
                        st.download_button("⬇️ Download Shortlist Report (Excel)", data=data, file_name="shortlist_report.xlsx",
                                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    else:
                        data = reports.df_to_csv_bytes(df)
                        st.download_button("⬇️ Download Shortlist Report (CSV)", data=data, file_name="shortlist_report.csv", mime="text/csv")

    with report_tabs[2]:
        _selected_for_report = [c for c in report_valid_candidates if c.get("status") == "Selected"]
        if not _selected_for_report:
            st.info("No candidates marked **Selected** yet — mark a candidate's decision on the Shortlisted page first.")
        else:
            st.caption(f"{len(_selected_for_report)} selected candidate(s)")
            fmt3 = st.radio("Format", ["PDF", "Excel", "CSV"], horizontal=True, key="selected_fmt")
            if st.button("📄 Generate Report", key="gen_selected"):
                if fmt3 == "PDF":
                    data = reports.build_shortlist_report_pdf(
                        _selected_for_report, st.session_state.job_role, st.session_state.weights,
                        st.session_state.job_details, _report_job_skills,
                    )
                    st.download_button("⬇️ Download Selected Candidates Report (PDF)", data=data,
                                         file_name="selected_candidates_report.pdf", mime="application/pdf")
                else:
                    df = reports.candidates_to_dataframe(_selected_for_report)
                    if fmt3 == "Excel":
                        data = reports.df_to_excel_bytes(df, sheet_name="Selected")
                        st.download_button("⬇️ Download Selected Candidates Report (Excel)", data=data, file_name="selected_candidates_report.xlsx",
                                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    else:
                        data = reports.df_to_csv_bytes(df)
                        st.download_button("⬇️ Download Selected Candidates Report (CSV)", data=data, file_name="selected_candidates_report.csv", mime="text/csv")

    with report_tabs[3]:
        if not report_interviews:
            st.info("No interviews scheduled yet.")
        else:
            st.caption(f"{len(report_interviews)} interview(s) recorded")
            fmt2 = st.radio("Format", ["PDF", "Excel", "CSV"], horizontal=True, key="interview_fmt")
            if st.button("📄 Generate Report", key="gen_interview"):
                if fmt2 == "PDF":
                    data = reports.build_interview_report_pdf(
                        report_interviews, st.session_state.job_role, st.session_state.job_details,
                        _report_job_skills, st.session_state.weights,
                    )
                    st.download_button("⬇️ Download Interview Report (PDF)", data=data, file_name="interview_report.pdf", mime="application/pdf")
                else:
                    df = reports.interviews_to_dataframe(report_interviews)
                    if fmt2 == "Excel":
                        data = reports.df_to_excel_bytes(df, sheet_name="Interviews")
                        st.download_button("⬇️ Download Interview Report (Excel)", data=data, file_name="interview_report.xlsx",
                                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    else:
                        data = reports.df_to_csv_bytes(df)
                        st.download_button("⬇️ Download Interview Report (CSV)", data=data, file_name="interview_report.csv", mime="text/csv")

    with report_tabs[4]:
        saved = local_settings.load_settings()
        using_company_account = auth.is_logged_in() and bool(st.session_state.get("auth_company"))
        _company = st.session_state.get("auth_company", {}) if using_company_account else {}
        offer_company_name = _company.get("name", "") if using_company_account else saved.get("offer_company_name", "")
        offer_logo_bytes = None
        if using_company_account and _company.get("logo_base64"):
            import base64 as _b64
            offer_logo_bytes = _b64.b64decode(_company["logo_base64"])

        if "offer_details_edit_mode" not in st.session_state:
            st.session_state.offer_details_edit_mode = not offer_company_name  # first time -> open for entry

        st.markdown("##### 🏢 Company & Signer Details")
        if using_company_account:
            st.caption("Company name and logo come from your account — signature and contact details below are still yours to fill in.")
        else:
            st.caption("Fill this in once — it's reused for every offer letter until you edit it again.")

        if not st.session_state.offer_details_edit_mode:
            dcol1, dcol2, dcol3 = st.columns([1, 2, 2])
            with dcol1:
                if using_company_account and offer_logo_bytes:
                    st.image(offer_logo_bytes, width=64)
                elif not using_company_account and saved.get("offer_logo_path") and os.path.exists(saved["offer_logo_path"]):
                    st.image(saved["offer_logo_path"], width=64)
                else:
                    st.caption("No logo")
            with dcol2:
                st.markdown(f"**{offer_company_name or '—'}**")
                st.caption(f"{saved.get('offer_company_phone') or '—'} · {saved.get('offer_company_email') or '—'}")
            with dcol3:
                st.markdown(f"**{saved.get('offer_hr_name') or '—'}**")
                st.caption(saved.get("offer_hr_title") or "—")
            _sig_family = SIGNATURE_FONT_FAMILIES.get(saved.get("offer_signature_style", "Elegant Script"), "Great Vibes")
            st.markdown(f"""
            <link href="https://fonts.googleapis.com/css2?family={_sig_family.replace(' ', '+')}&display=swap" rel="stylesheet">
            <div style="font-family:'{_sig_family}',cursive; font-size:2rem; color:#00506B; margin-top:6px;">
                {saved.get('offer_hr_name') or 'Your Signature'}
            </div>
            """, unsafe_allow_html=True)
            if st.button("✏️ Edit details", key="offer_details_edit_btn"):
                st.session_state.offer_details_edit_mode = True
                st.rerun()
        else:
            dc1, dc2 = st.columns(2)
            if using_company_account:
                dc1.text_input("Company Name", value=offer_company_name, disabled=True,
                                help="Set when you signed up — change it from your account settings, not here.")
                company_name_in = offer_company_name
            else:
                company_name_in = dc1.text_input("Company Name", value=saved.get("offer_company_name", ""), key="offer_company_name_in")
            company_email_in = dc2.text_input("Company Email", value=saved.get("offer_company_email", ""), key="offer_company_email_in")
            dc3, dc4 = st.columns(2)
            company_phone_in = dc3.text_input("Company Phone", value=saved.get("offer_company_phone", ""), key="offer_company_phone_in")

            if using_company_account:
                with dc4:
                    st.caption("Logo")
                    if offer_logo_bytes:
                        st.image(offer_logo_bytes, width=48)
                    else:
                        st.caption("No logo set on your account.")
                logo_upload = None
            else:
                logo_upload = dc4.file_uploader("Company Logo (optional)", type=["png", "jpg", "jpeg"], key="offer_logo_upload")
                if saved.get("offer_logo_path") and os.path.exists(saved["offer_logo_path"]) and not logo_upload:
                    st.image(saved["offer_logo_path"], width=56, caption="Current logo")

            dc5, dc6 = st.columns(2)
            hr_name_in = dc5.text_input("Signed By", value=saved.get("offer_hr_name", ""), key="offer_hr_name_in")
            hr_title_in = dc6.text_input("Signer Title", value=saved.get("offer_hr_title", "HR Manager"), key="offer_hr_title_in")

            st.markdown("**Signature style**")
            style_names = list(SIGNATURE_FONT_FAMILIES.keys())
            current_style = saved.get("offer_signature_style", "Elegant Script")
            signature_style_in = st.radio(
                "Signature style", style_names,
                index=style_names.index(current_style) if current_style in style_names else 0,
                key="offer_signature_style_in", label_visibility="collapsed", horizontal=True,
            )
            preview_name = hr_name_in.strip() or "Your Signature"
            font_previews = "".join(
                f"""<link href="https://fonts.googleapis.com/css2?family={fam.replace(' ', '+')}&display=swap" rel="stylesheet">
                <div style="border:2px solid {'#00506B' if name == signature_style_in else '#E2E8F0'};
                            border-radius:10px; padding:8px 16px; margin:4px 0; display:inline-block; width:100%;">
                    <span style="font-size:0.75rem; color:#94A3B8;">{name}</span><br>
                    <span style="font-family:'{fam}',cursive; font-size:1.7rem; color:#00506B;">{preview_name}</span>
                </div>"""
                for name, fam in SIGNATURE_FONT_FAMILIES.items()
            )
            st.markdown(font_previews, unsafe_allow_html=True)

            ecol1, ecol2 = st.columns(2)
            if ecol1.button("💾 Save details", key="offer_details_save_btn", type="primary", width="stretch"):
                if not company_name_in.strip():
                    st.error("Company Name is required.")
                else:
                    settings_to_save = {
                        "offer_company_phone": company_phone_in, "offer_company_email": company_email_in,
                        "offer_hr_name": hr_name_in, "offer_hr_title": hr_title_in,
                        "offer_signature_style": signature_style_in,
                    }
                    if not using_company_account:
                        logo_path_to_save = saved.get("offer_logo_path", "")
                        if logo_upload is not None:
                            logos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
                            os.makedirs(logos_dir, exist_ok=True)
                            ext = os.path.splitext(logo_upload.name)[1] or ".png"
                            logo_path_to_save = os.path.join(logos_dir, f"custom_offer_logo{ext}")
                            with open(logo_path_to_save, "wb") as f:
                                f.write(logo_upload.getbuffer())
                        settings_to_save["offer_company_name"] = company_name_in
                        settings_to_save["offer_logo_path"] = logo_path_to_save
                    local_settings.save_settings(settings_to_save)
                    st.session_state.offer_details_edit_mode = False
                    st.rerun()
            if offer_company_name and ecol2.button("Cancel", key="offer_details_cancel_btn", width="stretch"):
                st.session_state.offer_details_edit_mode = False
                st.rerun()

        st.markdown("---")

        selected_candidates = [c for c in report_valid_candidates if c.get("status") == "Selected"]
        if st.session_state.offer_details_edit_mode:
            st.info("Save your company & signer details above to create offer letters.")
        elif not selected_candidates:
            st.info("No candidates marked **Selected** yet — mark a candidate's decision as Selected on the "
                     "**Shortlisted** page, then come back here to create their offer letter.")
        else:
            names = [c["name"] for c in selected_candidates]
            pick = st.selectbox("Select a candidate", names, key="offer_candidate_pick")
            c = next(cc for cc in selected_candidates if cc["name"] == pick)

            # ---- Editable contact details (email + phone only) — lets a recruiter fix a
            # mis-extracted email/phone before creating/sending the offer letter, without
            # touching anything else about the candidate's screening data. ----
            candidate_email = c.get("profile", {}).get("email", "")
            candidate_phone = c.get("profile", {}).get("phone", "")
            with st.expander(f"✏️ Contact details — {c['name']}", expanded=not candidate_email):
                cd1, cd2 = st.columns(2)
                edited_email = cd1.text_input("Email", value=candidate_email, key=f"offer_contact_email_{c['id']}")
                edited_phone = cd2.text_input("Contact No.", value=candidate_phone, key=f"offer_contact_phone_{c['id']}")
                if st.button("💾 Save contact details", key=f"offer_contact_save_{c['id']}"):
                    updated_profile = dict(c["profile"])
                    updated_profile["email"] = edited_email.strip()
                    updated_profile["phone"] = edited_phone.strip()
                    update_candidate_record(c["id"], {"profile": updated_profile})
                    c["profile"] = updated_profile  # reflect immediately in this run too
                    st.success("Contact details updated.")
                    st.rerun()

            linked_job = next((jb for jb in get_jobs() if jb["id"] == c.get("job_id")), None)
            default_title = linked_job["title"] if linked_job else st.session_state.job_role
            default_location = linked_job.get("location", "") if linked_job else ""
            default_salary = linked_job.get("salary_range", "") if linked_job else ""
            default_benefits = linked_job.get("benefits", "") if linked_job else ""

            oc5, oc6 = st.columns(2)
            job_title = oc5.text_input("Job Title", value=default_title, key="offer_job_title")
            location = oc6.text_input("Location", value=default_location, key="offer_location")
            oc7, oc8 = st.columns(2)
            salary = oc7.text_input("Annual Salary (CTC)", value=default_salary, placeholder="e.g. INR 8,40,000 per annum", key="offer_salary")
            reporting_manager = oc8.text_input("Reporting Manager", key="offer_manager")
            oc9, oc10 = st.columns(2)
            start_date = oc9.date_input("Start Date", key="offer_start_date")
            accept_by = oc10.date_input("Accept By", key="offer_accept_by")
            oc11, oc12 = st.columns(2)
            department = oc11.text_input("Department (optional)", key="offer_department")
            employment_type = oc12.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Internship"], key="offer_employment_type")
            oc13, oc14 = st.columns(2)
            probation_period = oc13.text_input("Probation Period (optional)", placeholder="e.g. 3 months", key="offer_probation")
            work_hours = oc14.text_input("Work Hours (optional)", placeholder="e.g. 9:30 AM – 6:30 PM, Mon–Fri", key="offer_work_hours")
            benefits = st.text_input("Benefits (optional)", value=default_benefits, key="offer_benefits")

            if st.button("📄 Generate Offer Letter PDF", key="offer_generate_btn", type="primary"):
                if not job_title.strip():
                    st.error("Job Title is required.")
                else:
                    offer = {
                        "company_name": offer_company_name,
                        "company_phone": saved.get("offer_company_phone", ""), "company_email": saved.get("offer_company_email", ""),
                        "date": datetime.now().strftime("%d %B %Y"),
                        "job_title": job_title, "location": location, "salary": salary,
                        "department": department, "employment_type": employment_type,
                        "probation_period": probation_period, "work_hours": work_hours,
                        "benefits": benefits,
                        "start_date": start_date.strftime("%B %d, %Y") if start_date else "",
                        "accept_by": accept_by.strftime("%B %d, %Y") if accept_by else "",
                        "reporting_manager": reporting_manager,
                        "hr_name": saved.get("offer_hr_name", ""), "hr_title": saved.get("offer_hr_title", "HR Manager"),
                        "signature_style": saved.get("offer_signature_style", "Elegant Script"),
                    }
                    if using_company_account:
                        pdf_bytes = reports.build_offer_letter_pdf(c, offer, logo_bytes=offer_logo_bytes)
                    else:
                        pdf_bytes = reports.build_offer_letter_pdf(c, offer, logo_path=saved.get("offer_logo_path") or None)
                    # Persisted in session state (not just this button's scope) so the
                    # Send-via-Email button below still has the PDF after its own click/rerun.
                    st.session_state.offer_pdf_bytes = pdf_bytes
                    st.session_state.offer_pdf_filename = f"Offer_Letter_{c['name'].replace(' ', '_')}.pdf"
                    st.session_state.offer_pdf_candidate = c["name"]
                    st.session_state.offer_pdf_job_title = job_title

            # Only show download/email actions for a PDF generated for the currently selected candidate —
            # prevents accidentally emailing a stale PDF after switching candidates.
            if st.session_state.get("offer_pdf_candidate") == c["name"] and st.session_state.get("offer_pdf_bytes"):
                dl_col, mail_col = st.columns(2)
                with dl_col:
                    st.download_button(
                        "⬇️ Download Offer Letter", data=st.session_state.offer_pdf_bytes,
                        file_name=st.session_state.offer_pdf_filename,
                        mime="application/pdf", key="offer_download_btn", width="stretch",
                    )
                with mail_col:
                    with st.popover("📧 Send via Email", use_container_width=True):
                        if not email_utils.is_configured():
                            st.warning("Email sending isn't set up yet.")
                            st.caption(
                                "**Want it sent from your actual Gmail?** Use a Google Apps Script relay — "
                                "5-minute one-time setup, works over HTTPS so campus/office WiFi blocking "
                                "SMTP ports doesn't matter. Full copy-paste steps are in the docstring at "
                                "the top of `email_utils.py`. You'll end up adding:\n\n"
                                "```\nGAS_WEBHOOK_URL = \"https://script.google.com/macros/s/XXXXX/exec\"\n"
                                "GAS_SECRET = \"your-random-string\"\n```\n\n"
                                "**Simpler alternative (not from your own Gmail):**\n\n"
                                "```\nRESEND_API_KEY = \"re_xxxxxxxxxxxx\"\n```\n"
                                "Free at [resend.com](https://resend.com) — 100 emails/day, no setup beyond an API key.\n\n"
                                "**Or, traditional SMTP** (may be blocked on some networks):\n\n"
                                "```\nSMTP_EMAIL = \"you@company.com\"\n"
                                "SMTP_APP_PASSWORD = \"your-app-password\"\n```"
                            )
                        else:
                            candidate_email_default = c.get("profile", {}).get("email", "")
                            to_email = st.text_input("Candidate's email", value=candidate_email_default, key="offer_email_to")
                            if email_utils.is_gas_configured():
                                st.caption("✅ Sending via your Gmail (Google Apps Script)")
                                provider = "Gmail"
                            elif email_utils.is_resend_configured():
                                st.caption("✅ Sending via Resend (HTTPS)")
                                provider = "Gmail"  # unused when Resend/GAS is active, kept for signature compatibility
                            else:
                                provider = st.selectbox("Sending from", list(email_utils.SMTP_PRESETS.keys()) + ["Custom"], key="offer_email_provider")
                            default_subject = f"Job Offer — {st.session_state.offer_pdf_job_title} at {saved.get('offer_company_name', '')}"
                            subject = st.text_input("Subject", value=default_subject, key="offer_email_subject")
                            default_body = (
                                f"Hi {c['name'].split(' ')[0]},\n\n"
                                f"Please find attached your offer letter for the {st.session_state.offer_pdf_job_title} "
                                f"position at {saved.get('offer_company_name', '')}. We're excited about the possibility "
                                f"of you joining the team!\n\n"
                                f"Please review the attached letter and let us know if you have any questions.\n\n"
                                f"Best regards,\n{saved.get('offer_hr_name', '')}\n{saved.get('offer_hr_title', '')}"
                            )
                            body_text = st.text_area("Message", value=default_body, height=180, key="offer_email_body")
                            if st.button("📤 Send", key="offer_email_send_btn", type="primary"):
                                if not to_email.strip():
                                    st.error("Enter the candidate's email address.")
                                else:
                                    with st.spinner("Sending..."):
                                        success, message = email_utils.send_email_with_pdf(
                                            to_email.strip(), subject, body_text,
                                            st.session_state.offer_pdf_bytes, st.session_state.offer_pdf_filename,
                                            provider=provider,
                                        )
                                    if success:
                                        st.success(f"✅ {message}")
                                    else:
                                        st.error(message)

# ----------------------------- FLOATING CHAT WIDGET (every page except AI Insights) -----------------------------
if page != "🤖 AI Insights":
    if "show_fab_chat" not in st.session_state:
        st.session_state.show_fab_chat = False

    st.markdown("""
    <style>
    .st-key-fab_toggle {
        background: transparent !important;
    }
    .st-key-fab_toggle button, .st-key-fab_toggle button:focus, .st-key-fab_toggle button:active {
        position: fixed !important; bottom: 24px !important; right: 24px !important; z-index: 9999 !important;
        width: 60px !important; height: 60px !important; border-radius: 50% !important;
        font-size: 1.5rem !important; padding: 0 !important;
        background: linear-gradient(135deg, #38BDF8 0%, #00506B 100%) !important;
        color: #ffffff !important; border: none !important;
        box-shadow: 0 6px 20px rgba(0,80,107,0.45) !important;
    }
    .st-key-fab_toggle button:hover {
        transform: scale(1.06);
        box-shadow: 0 8px 24px rgba(0,80,107,0.55) !important;
        background: linear-gradient(135deg, #0EA5E9 0%, #00394D 100%) !important;
    }
    .st-key-fab_panel {
        position: fixed; bottom: 96px; right: 24px; z-index: 9998;
        width: 350px; max-height: 60vh; overflow-y: auto;
        background: #0F172A; border: 1px solid #1E293B;
        border-radius: 16px; padding: 16px;
        box-shadow: 0 12px 32px rgba(2,136,209,0.28);
    }
    .st-key-fab_panel p, .st-key-fab_panel span, .st-key-fab_panel strong,
    .st-key-fab_panel .stMarkdown, .st-key-fab_panel label {
        color: #F1F5F9 !important;
    }
    .st-key-fab_panel [data-testid="stCaptionContainer"] { color: #94A3B8 !important; }
    .st-key-fab_panel [data-testid="stChatMessage"] {
        background: #1E293B !important; border-radius: 10px; border: 1px solid #334155;
    }
    .st-key-fab_panel [data-testid="stChatInput"] textarea {
        background: #FFFFFF !important; color: #0B1C30 !important; border-color: #CBD5E1 !important;
    }
    .st-key-fab_panel [data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important; opacity: 1 !important;
    }
    .st-key-fab_panel [data-testid="stChatInput"] {
        background: #FFFFFF !important; border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    def _toggle_fab_chat():
        st.session_state.show_fab_chat = not st.session_state.show_fab_chat

    st.button("✕" if st.session_state.show_fab_chat else "💬", key="fab_toggle", on_click=_toggle_fab_chat)

    if st.session_state.show_fab_chat:
        with st.container(key="fab_panel"):
            st.markdown("**🤖 Quick Assistant**")
            st.caption(f"Ask about the **{page}** page, or anything about your candidates.")

            for turn in st.session_state.chat_history[-6:]:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])

            fab_q = st.chat_input("Ask a quick question...", key="fab_chat_input")
            if fab_q:
                st.session_state.chat_history.append({"role": "user", "content": fab_q})
                valid_candidates = [c for c in st.session_state.candidates if not c["score"].get("error")]
                with logo_loading("Thinking..."):
                    try:
                        fab_answer = ask_assistant(
                            f"(The recruiter is currently on the '{page}' page.) {fab_q}",
                            valid_candidates,
                            st.session_state.job_role, st.session_state.job_details,
                            st.session_state.chat_history,
                        )
                    except Exception as e:
                        fab_answer = f"Sorry, I couldn't get an answer right now: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": fab_answer})
                st.rerun()
