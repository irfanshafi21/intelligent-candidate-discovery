"""
Intelligent Candidate Discovery Platform
AI-Powered Resume Screening and Candidate Ranking

Built with Streamlit + Google Gemini API
"""

import streamlit as st
import json
import time
import random
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
from resume_parser import (
    extract_text_from_file,
    extract_text_from_bytes,
    extract_files_from_zip,
    heuristic_resume_check,
    compute_local_ats_metrics,
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
import reports
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


def update_interview_record(interview_id, updates: dict) -> None:
    if db.is_configured():
        db.update_interview(interview_id, updates)
    for i in st.session_state.local_interviews:
        if i["id"] == interview_id:
            i.update(updates)

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="Intelligent Candidate Discovery Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- STYLING -----------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        /* Core palette (accurate ICD Platform Design System tokens) */
        --primary: #00668A;
        --primary-dark: #004965;
        --secondary: #E0F2FE;
        --secondary-container: #39B8FD;
        --accent: #38BDF8;
        --bg: #F8F9FF;
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

# ----------------------------- SESSION STATE -----------------------------
if "candidates" not in st.session_state:
    st.session_state.candidates = []  # list of dicts: {name, raw_text, profile, score}
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
if "bookmarked" not in st.session_state:
    st.session_state.bookmarked = set()
if "rejected" not in st.session_state:
    st.session_state.rejected = set()
if "candidate_notes" not in st.session_state:
    st.session_state.candidate_notes = {}
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
                "**Recommended: Groq (primary, 30 req/min free)**\n"
                "1. Get a free key at **console.groq.com** → API Keys → Create API Key\n\n"
                "**Optional: Gemini (automatic fallback)**\n"
                "1. Get a free key at **aistudio.google.com** → Get API key\n\n"
                "Then create `.streamlit/secrets.toml`:\n"
                "```toml\nGROQ_API_KEY = \"your-groq-key\"\nGEMINI_API_KEY = \"your-gemini-key\"\n```\n"
                "You only need one — restart the app after saving."
            )

    if st.session_state.candidates:
        n_ok = len([c for c in st.session_state.candidates if not c["score"].get("error")])
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); border-radius:10px; padding:10px 14px; margin-top:8px;">
            <span style="color:#E2E8F0; font-size:0.8rem;">📋 <b>{n_ok}</b> candidate(s) screened this session</span>
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

    nav_items = ["🏠 Home", "👥 Candidates", "📤 Resume Screening", "📋 Jobs", "🗣️ Interview", "📄 Reports", "🤖 AI Insights"]
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
            st.warning("Remove all screened candidates from this session? This can't be undone.")
            cc1, cc2 = st.columns(2)
            if cc1.button("✅ Yes, clear", width="stretch"):
                st.session_state.candidates = []
                st.session_state.processed = False
                st.session_state.chat_history = []
                st.session_state.confirm_clear_candidates = False
                st.rerun()
            if cc2.button("Cancel", width="stretch"):
                st.session_state.confirm_clear_candidates = False
                st.rerun()

# ----------------------------- TOP NAVBAR -----------------------------
st.markdown("""
<style>
.st-key-topnavbar {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 16px 26px; margin: 6px 0 28px;
    box-shadow: var(--shadow-1);
    position: sticky; top: 0; z-index: 999;
}
.navbar-logo { font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.25rem; color: var(--text); }
.navbar-logo span { color: var(--primary); }
.navbar-breadcrumb { font-size:0.78rem; color: var(--text-secondary); margin-top:2px; }
.navbar-breadcrumb b { color: var(--primary); font-weight:700; }
.navbar-profile {
    display:flex; align-items:center; gap:10px; justify-content:flex-end;
    font-weight:700; color: var(--text); font-size:0.9rem; height: 100%;
}
.navbar-profile .avatar {
    width:36px; height:36px; border-radius:50%;
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color:#fff; display:flex; align-items:center; justify-content:center; font-size:0.85rem; font-weight:800;
}
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
    nav_logo, nav_action, nav_profile = st.columns([4.2, 1.5, 1.6], vertical_alignment="center")

    with nav_logo:
        current_label = st.session_state.get("current_page", "🏠 Home")
        breadcrumb = "🏠 Home" if current_label == "🏠 Home" else f"🏠 Home / <b>{current_label}</b>"
        st.markdown(f"""
        <div class="navbar-logo">🎯 <span>ICD</span> Platform</div>
        <div class="navbar-breadcrumb">{breadcrumb}</div>
        """, unsafe_allow_html=True)

    with nav_action:
        if st.button("➕ Quick Action", key="nav_quick_action", width="stretch"):
            st.session_state.current_page = "📤 Resume Screening"
            st.rerun()

    with nav_profile:
        st.markdown("""
        <div class="navbar-profile">
            <div class="avatar">MS</div>
            <div>Recruiter<br><span style="font-weight:500; color:var(--text-secondary); font-size:0.72rem;">Guest session</span></div>
        </div>
        """, unsafe_allow_html=True)

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
                    "screened_at": datetime.now().isoformat(), "status": "In Review",
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
            return {
                "filename": name,
                "name": profile.get("name") or name,
                "raw_text": raw_text,
                "profile": profile,
                "score": score_result,
                "screened_at": datetime.now().isoformat(),
                "status": "In Review",
            }

        # Step 2: parse + score all confirmed resumes concurrently — each
        # candidate is independent, so there's no need to wait for one before
        # starting the next. Capped at 5 workers to stay within free-tier RPM limits.
        valid_items = list(confirmed_texts.items())
        total = len(raw_items)  # includes excluded/failed files in the progress denominator
        completed = total - len(valid_items)  # extraction failures + excluded non-resumes already "done"
        with ThreadPoolExecutor(max_workers=min(5, max(1, len(valid_items)))) as executor:
            futures = {
                executor.submit(process_one, name, text): name
                for name, text in valid_items
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    st.session_state.candidates.append(result)
                    # Best-effort persistence for Analytics — silently skipped if
                    # Supabase isn't configured; never blocks or errors the UI.
                    db.save_screening_record(result, job_role, job_details, job_id=st.session_state.selected_job_id)
                except Exception as e:
                    st.session_state.candidates.append({
                        "filename": name, "name": name, "raw_text": "",
                        "profile": {}, "score": {"overall_score": 0, "error": str(e)},
                        "screened_at": datetime.now().isoformat(), "status": "In Review",
                    })
                completed += 1
                progress.progress(completed / max(total, 1), text=f"Screened {completed}/{total}...")

        progress.empty()
        st.session_state.processed = True
        if valid_items:
            saved_note = " Saved to persistent history for Analytics." if db.is_configured() else ""
            notify(f"Screened {len(valid_items)} candidate(s).{saved_note} Head to the Home tab to view rankings.")
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
    rated_interviews = [i for i in completed_interviews if i.get("score_locked") and i.get("interview_score") is not None]

    selected_candidates = [c for c in valid_candidates if c.get("status") == "Selected"]
    rejected_candidates = [c for c in valid_candidates if c.get("status") == "Rejected"]
    shortlisted_candidates = [c for c in valid_candidates if c["score"].get("overall_score", 0) >= 70]

    avg_ats = round(sum(c["score"].get("overall_score", 0) for c in valid_candidates) / len(valid_candidates), 1) if valid_candidates else 0
    avg_interview_score = round(sum(i.get("interview_score", 0) for i in rated_interviews) / len(rated_interviews), 1) if rated_interviews else None
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
        ("📤", "Resumes Uploaded", len(all_candidates), None),
        ("🧠", "Resumes Analyzed", len(valid_candidates), None),
        ("⭐", "Shortlisted (≥70)", len(shortlisted_candidates), None),
        ("✅", "Selected", len(selected_candidates), None),
        ("❌", "Rejected", len(rejected_candidates), None),
        ("🗣️", "Interviewed", len(completed_interviews), None),
        ("⏳", "Pending Interviews", len(scheduled_interviews), None),
        ("📋", "Active Job Openings", len(active_jobs), None),
        ("🏁", "Completed Recruitments", len(completed_jobs), "archived jobs"),
        ("🎯", "Avg ATS Score", f"{avg_ats}/100" if valid_candidates else "—", None),
        ("🗣️", "Avg Interview Score", f"{avg_interview_score}/100" if avg_interview_score is not None else "—",
         "based on locked scores" if avg_interview_score is not None else "no locked scores yet"),
        ("📈", "Hiring Success Rate", f"{success_rate}%" if success_rate is not None else "—",
         "based on Selected vs Rejected decisions" if success_rate is not None else "no decisions made yet"),
    ]
    cols_per_row = 4
    for row_start in range(0, len(stat_cards), cols_per_row):
        row = stat_cards[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (icon, label, value, sub) in zip(cols, row):
            with col:
                st.markdown(components.stat_card_html(icon, label, value, sub), unsafe_allow_html=True)
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

    st.divider()

    # ---- Ranked shortlist (preserved from the previous Dashboard page) ----
    page_header("🏆", f"Ranked Shortlist — Top {st.session_state.top_n}")
    if not valid_candidates:
        st.info("No screened candidates yet. Go to **Resume Screening** first.")
    else:
        st.caption("Ranked using your priority weighting — not a flat average. Mark candidates Selected or Rejected to track hiring decisions.")

        filter_col, sort_col = st.columns([2, 1])
        with filter_col:
            min_score = st.slider("Minimum score filter", 0, 100, 0, key="home_min_score")
        with sort_col:
            st.write("")

        search_term = st.session_state.global_search.strip().lower()
        if search_term:
            def _matches_search(c):
                if search_term in c["name"].lower():
                    return True
                skills = c["profile"].get("skills", []) or []
                return any(search_term in str(s).lower() for s in skills)
            ranked_filtered = [c for c in ranked if _matches_search(c)]
            st.caption(f"🔍 Filtering by \"{st.session_state.global_search}\" — {len(ranked_filtered)} match(es)")
        else:
            ranked_filtered = ranked

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

            status_label = c.get("status", "In Review")
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
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View details — {c['name']}"):
                    profile = c["profile"]
                    score_data = c["score"]

                    dec1, dec2, dec3 = st.columns([1, 1, 2])
                    with dec1:
                        if components.styled_button("✅ Mark Selected", key=f"home_select_{idx}_{c.get('filename','')}", variant="success", width="stretch"):
                            c["status"] = "Selected"
                            st.rerun()
                    with dec2:
                        if components.styled_button("❌ Mark Rejected", key=f"home_reject_{idx}_{c.get('filename','')}", variant="danger", width="stretch"):
                            c["status"] = "Rejected"
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

    st.divider()

    # ---- Reports teaser (full export UI now lives on its own Reports page) ----
    st.markdown("### 📄 Reports")
    st.caption("Export candidate, shortlist, and interview data as PDF, Excel, or CSV.")
    if st.button("Go to Reports →", key="home_goto_reports"):
        st.session_state.current_page = "📄 Reports"
        st.rerun()

# ============================================================
# PAGE — CANDIDATES (grid / profile / compare)
# ============================================================
elif page == "👥 Candidates":
    all_candidates = [c for c in st.session_state.candidates if not c["score"].get("error")]

    def _ckey(c):
        return c["filename"]

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
                    if st.button("✨ Generate Interview Questions", type="primary", key="profile_gen_q"):
                        with st.spinner("Generating tailored questions..."):
                            try:
                                selected["questions"] = generate_interview_questions(p, s, f"Job Role: {st.session_state.job_role}\n\n{st.session_state.job_details}")
                            except Exception as e:
                                st.error(f"Couldn't generate questions: {e}")
                    if "questions" in selected:
                        for section, qs in selected["questions"].items():
                            st.markdown(f"**{section}**")
                            for item in qs:
                                q_text = item.get("question", "") if isinstance(item, dict) else str(item)
                                guidance = item.get("what_good_looks_like", "") if isinstance(item, dict) else ""
                                st.markdown(f"**Q:** {q_text}")
                                if guidance:
                                    st.caption(f"What a strong answer covers: {guidance}")

                with tabs[7]:
                    key = _ckey(selected)
                    note_val = st.text_area("Recruiter notes for this candidate", value=st.session_state.candidate_notes.get(key, ""), height=150)
                    if st.button("💾 Save Notes"):
                        st.session_state.candidate_notes[key] = note_val
                        st.success("Notes saved for this session.")

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
                filtered = [c for c in filtered if _ckey(c) not in st.session_state.rejected]

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
                    is_bookmarked = key in st.session_state.bookmarked
                    is_rejected = key in st.session_state.rejected
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
                                    if is_bookmarked:
                                        st.session_state.bookmarked.discard(key)
                                    else:
                                        st.session_state.bookmarked.add(key)
                                    st.rerun()
                            with b4:
                                if st.button("↩️" if is_rejected else "🚫", key=f"rej_{key}", help="Un-reject" if is_rejected else "Reject", width="stretch"):
                                    if is_rejected:
                                        st.session_state.rejected.discard(key)
                                    else:
                                        st.session_state.rejected.add(key)
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
            names = [c["name"] for c in valid_candidates]
            selected_name = st.selectbox("Select a candidate", names)
            selected = next(c for c in valid_candidates if c["name"] == selected_name)

            st.caption(f"Fit score: {selected['score'].get('overall_score', '—')}/100")

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

            if st.button("✨ Generate Interview Questions", type="primary"):
                with st.spinner("Generating tailored questions..."):
                    questions = generate_interview_questions(
                        selected["profile"], selected["score"],
                        f"Job Role: {st.session_state.job_role}\n\nKey Requirements:\n{st.session_state.job_details}"
                    )
                    selected["questions"] = questions

            if "questions" in selected:
                st.caption("Each question includes what a strong, evidence-based answer should cover — "
                            "use it as a rubric while listening, not a script the candidate should recite.")
                for section, qs in selected["questions"].items():
                    st.markdown(f"**{section}**")
                    for item in qs:
                        if isinstance(item, dict):
                            q_text = item.get("question", "")
                            guidance = item.get("what_good_looks_like", "")
                        else:
                            q_text, guidance = str(item), ""
                        st.markdown(f"**Q:** {q_text}")
                        if guidance:
                            st.markdown(f"""
                            <div style="background:#E3F5FD; border-left:3px solid #38BDF8; border-radius:8px; padding:8px 14px; margin:4px 0 14px; font-size:0.85rem; color:#12314A;">
                                <b>What a strong answer covers:</b> {guidance}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.write("")

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
                        if interview.get("score_locked"):
                            score = interview.get("interview_score")
                            st.markdown(f"""
                            <div style="display:flex; align-items:center; gap:8px; background:#F1F5F9; border:1px solid #E2E8F0;
                                        border-radius:10px; padding:10px 14px; margin-top:4px;">
                                <span style="font-size:1.1rem;">🔒</span>
                                <span style="font-weight:800; color:#0B1C30;">Final Interview Score: {score}/100</span>
                                <span style="font-size:0.75rem; color:#64748B;">— locked, cannot be edited</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            score_input = st.number_input(
                                "Interview Score (out of 100)", min_value=0, max_value=100,
                                value=interview.get("interview_score") or 50, step=1,
                                key=f"score_{idx}_{interview['id']}",
                            )
                            st.caption("Once submitted, the score is final and cannot be changed — double-check before submitting.")
                            if components.styled_button("🔒 Submit Final Score", key=f"submit_score_{idx}_{interview['id']}", variant="warning"):
                                update_interview_record(interview["id"], {"interview_score": score_input, "score_locked": True})
                                st.success(f"Score locked at {score_input}/100.")
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
            with st.spinner("Thinking..."):
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
# PAGE — REPORTS (extracted from Home so it has its own sidebar entry)
# ============================================================
elif page == "📄 Reports":
    page_header("📄", "Reports", "Export candidate, shortlist, and interview data as PDF, Excel, or CSV.")

    report_valid_candidates = [c for c in st.session_state.candidates if not c["score"].get("error")]
    report_interviews = get_interviews()

    report_tabs = st.tabs(["Candidate Report", "Shortlist Report", "Interview Report"])

    with report_tabs[0]:
        if not report_valid_candidates:
            st.info("No screened candidates yet.")
        else:
            names = [c["name"] for c in report_valid_candidates]
            pick = st.selectbox("Select a candidate", names, key="report_candidate_pick")
            picked_c = next(c for c in report_valid_candidates if c["name"] == pick)
            if st.button("📄 Generate PDF", key="gen_candidate_pdf"):
                pdf_bytes = reports.build_candidate_report_pdf(picked_c, st.session_state.job_role)
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
                    data = reports.build_shortlist_report_pdf(report_valid_candidates, st.session_state.job_role)
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
        if not report_interviews:
            st.info("No interviews scheduled yet.")
        else:
            st.caption(f"{len(report_interviews)} interview(s) recorded")
            fmt2 = st.radio("Format", ["PDF", "Excel", "CSV"], horizontal=True, key="interview_fmt")
            if st.button("📄 Generate Report", key="gen_interview"):
                if fmt2 == "PDF":
                    data = reports.build_interview_report_pdf(report_interviews)
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

# ----------------------------- FLOATING CHAT WIDGET (every page except AI Insights) -----------------------------
if page != "🤖 AI Insights":
    if "show_fab_chat" not in st.session_state:
        st.session_state.show_fab_chat = False

    st.markdown("""
    <style>
    .st-key-fab_toggle button {
        position: fixed; bottom: 24px; right: 24px; z-index: 9999;
        width: 60px; height: 60px; border-radius: 50%;
        font-size: 1.5rem; padding: 0;
        background: linear-gradient(135deg, #38BDF8 0%, #00668A 100%);
        color: #ffffff !important; border: none;
        box-shadow: 0 6px 20px rgba(2,136,209,0.45);
    }
    .st-key-fab_toggle button:hover {
        transform: scale(1.06);
        box-shadow: 0 8px 24px rgba(2,136,209,0.55);
    }
    .st-key-fab_panel {
        position: fixed; bottom: 96px; right: 24px; z-index: 9998;
        width: 350px; max-height: 60vh; overflow-y: auto;
        background: #ffffff; border: 1px solid var(--sky-border);
        border-radius: 16px; padding: 16px;
        box-shadow: 0 12px 32px rgba(2,136,209,0.28);
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("✕" if st.session_state.show_fab_chat else "💬", key="fab_toggle"):
        st.session_state.show_fab_chat = not st.session_state.show_fab_chat

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
