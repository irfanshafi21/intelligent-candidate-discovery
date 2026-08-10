"""
Supabase persistence layer — stores every successfully screened candidate so
Analytics can show real trends across sessions, not just this one.

Design principle: persistence is best-effort and never blocks the core
screening flow. If Supabase isn't configured, or a network call fails, these
functions return empty/None rather than raising — the app should degrade to
session-only data, not crash.
"""

import os
import json
import streamlit as st

_last_error = None


def get_last_error() -> str | None:
    """The most recent Supabase error, if any — surfaced in the UI so
    failures aren't silently swallowed into an untraceable fallback."""
    return _last_error


def _get_secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def is_configured() -> bool:
    return bool(_get_secret("SUPABASE_URL") and _get_secret("SUPABASE_KEY"))


def _get_client():
    """
    Supabase client, cached per browser session (st.session_state) — NOT
    globally with st.cache_resource. This matters a lot now that the client
    can carry an authenticated user session (see auth.py): st.cache_resource
    caches ONE object shared across every visitor to this app on the same
    server process, which would leak one user's login into another user's
    requests. Row Level Security policies that check auth.uid() are only
    correct if each browser session gets its own client instance.
    """
    global _last_error
    if "_supabase_client" in st.session_state:
        return st.session_state._supabase_client
    from supabase import create_client
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        client = create_client(url, key)
        st.session_state._supabase_client = client
        return client
    except Exception as e:
        _last_error = f"Client creation failed: {e}"
        return None


def _current_company_id():
    """The logged-in company's id (set by auth.py after login), or None if
    auth isn't in use. Read directly from session_state rather than
    importing auth.py, to avoid a circular import between the two modules."""
    company = st.session_state.get("auth_company")
    return company.get("id") if company else None


def save_screening_record(candidate: dict, job_role: str, job_details: str, job_id=None) -> dict | None:
    """
    Persist one screened candidate to Supabase. Returns the inserted row
    (including its Supabase-assigned "id", needed later to update decision
    status / bookmark / notes / interview score) or None on failure —
    callers should treat None as a no-op / fall back to local session
    storage, never as an error to surface to the user mid-screening.
    job_id is optional — links this screening to a saved job posting (Phase 5)
    when one was selected; ad-hoc screenings without a saved job simply omit it.

    Stores the FULL profile/score/raw_text as jsonb/text so a candidate can be
    reconstructed exactly as-is after the app restarts — not just enough for
    Analytics aggregates like the earlier version of this function did.
    """
    global _last_error
    client = _get_client()
    if client is None:
        return None
    try:
        p, s = candidate["profile"], candidate["score"]
        row = {
            "company_id": _current_company_id(),
            "job_role": job_role,
            "job_details": job_details,
            "job_id": job_id,
            "candidate_name": candidate.get("name"),
            "filename": candidate.get("filename"),
            "email": p.get("email"),
            "phone": p.get("phone"),
            "years_experience": p.get("years_experience"),
            "education": p.get("education"),
            "skills": json.dumps(p.get("skills", [])),
            "past_roles": json.dumps(p.get("past_roles", [])),
            "overall_score": s.get("overall_score"),
            "skills_match": s.get("breakdown", {}).get("skills_match"),
            "experience_fit": s.get("breakdown", {}).get("experience_fit"),
            "education_fit": s.get("breakdown", {}).get("education_fit"),
            "matched_skills": json.dumps(s.get("matched_skills", [])),
            "gaps": json.dumps(s.get("gaps", [])),
            "recruiter_summary": s.get("summary"),
            "questions_json": json.dumps(candidate.get("questions", {}) or {}),
            "raw_text": candidate.get("raw_text", ""),
            "profile_json": json.dumps(p),
            "score_json": json.dumps(s),
            "screened_at": candidate.get("screened_at"),
            "status": "active",
            "decision_status": candidate.get("status", "Waiting"),
            "bookmarked": False,
            "interview_score": candidate.get("interview_score"),
            "notes": candidate.get("notes", ""),
        }
        result = client.table("screening_history").insert(row).execute()
        _last_error = None
        saved_row = (result.data or [None])[0]
        return _parse_row_json_fields(saved_row) if saved_row else None
    except Exception as e:
        _last_error = f"save_screening_record failed: {e}"
        return None


def _parse_row_json_fields(row: dict) -> dict:
    """Some Supabase client/insert-response paths return jsonb columns as
    already-decoded dicts, others as raw JSON text (e.g. right after an
    insert echoes back exactly what was sent). Normalize both cases so every
    caller — fetch or save — always gets real dicts/lists back, never a
    string that later code calls .get()/.items() on and crashes."""
    for field in ("skills", "past_roles", "matched_skills", "gaps"):
        if isinstance(row.get(field), str):
            try:
                row[field] = json.loads(row[field])
            except Exception:
                row[field] = []
    for field in ("profile_json", "score_json", "questions_json"):
        if isinstance(row.get(field), str):
            try:
                row[field] = json.loads(row[field])
            except Exception:
                row[field] = {}
    return row


def update_screening_record(record_id, updates: dict) -> bool:
    """Update a single persisted candidate row (decision status, bookmark,
    notes, interview score, etc). Used so every page that lets a recruiter
    change a candidate's state writes straight through to Supabase."""
    global _last_error
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("screening_history").update(updates).eq("id", record_id).execute()
        _last_error = None
        return True
    except Exception as e:
        _last_error = f"update_screening_record failed: {e}"
        return False


def clear_screening_records(job_id=None) -> bool:
    """Soft-delete: marks active candidate rows as 'cleared' instead of
    deleting them, so 'Clear all candidates' is recoverable and doesn't
    break historical Analytics trends. Pass job_id to clear only candidates
    tied to that job; omit to clear everything currently active."""
    global _last_error
    client = _get_client()
    if client is None:
        return False
    try:
        q = client.table("screening_history").update({"status": "cleared"}).eq("status", "active")
        if job_id is not None:
            q = q.eq("job_id", job_id)
        company_id = _current_company_id()
        if company_id is not None:
            q = q.eq("company_id", company_id)
        q.execute()
        _last_error = None
        return True
    except Exception as e:
        _last_error = f"clear_screening_records failed: {e}"
        return False


def fetch_screening_history(limit: int = 1000, include_cleared: bool = False) -> list[dict]:
    """
    Fetch historical screening records across all past sessions. Returns an
    empty list if Supabase isn't configured or the query fails — the app
    should fall back to session-only data in that case, not error out.
    By default only "active" (not cleared) records are returned so the app's
    normal views reflect what "Clear all candidates" removed.
    """
    client = _get_client()
    if client is None:
        return []
    try:
        q = client.table("screening_history").select("*")
        if not include_cleared:
            q = q.eq("status", "active")
        company_id = _current_company_id()
        if company_id is not None:
            q = q.eq("company_id", company_id)
        result = q.order("screened_at", desc=True).limit(limit).execute()
        rows = result.data or []
        return [_parse_row_json_fields(row) for row in rows]
    except Exception:
        return []


# ============================================================
# JOBS — Phase 5
# ============================================================

def save_job(job: dict) -> dict | None:
    """Create a new job posting. Returns the inserted row (with its id) or None on failure."""
    global _last_error
    client = _get_client()
    if client is None:
        return None
    try:
        row = {**job, "status": job.get("status", "active"), "company_id": _current_company_id()}
        result = client.table("jobs").insert(row).execute()
        _last_error = None
        return (result.data or [None])[0]
    except Exception as e:
        _last_error = f"save_job failed: {e}"
        return None


def update_job(job_id: int, updates: dict) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        from datetime import datetime, timezone
        updates = {**updates, "updated_at": datetime.now(timezone.utc).isoformat()}
        client.table("jobs").update(updates).eq("id", job_id).execute()
        return True
    except Exception:
        return False


def delete_job(job_id: int) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("jobs").delete().eq("id", job_id).execute()
        return True
    except Exception:
        return False


def fetch_jobs(include_archived: bool = True) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        q = client.table("jobs").select("*").order("created_at", desc=True)
        if not include_archived:
            q = q.eq("status", "active")
        company_id = _current_company_id()
        if company_id is not None:
            q = q.eq("company_id", company_id)
        result = q.execute()
        rows = result.data or []
        for row in rows:
            if isinstance(row.get("required_skills"), str):
                try:
                    row["required_skills"] = json.loads(row["required_skills"])
                except Exception:
                    row["required_skills"] = []
        return rows
    except Exception:
        return []


# ============================================================
# INTERVIEWS — Phase 6
# ============================================================

def save_interview(interview: dict) -> dict | None:
    global _last_error
    client = _get_client()
    if client is None:
        return None
    try:
        row = {**interview, "company_id": _current_company_id()}
        result = client.table("interviews").insert(row).execute()
        _last_error = None
        return (result.data or [None])[0]
    except Exception as e:
        _last_error = f"save_interview failed: {e}"
        return None


def update_interview(interview_id: int, updates: dict) -> bool:
    global _last_error
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("interviews").update(updates).eq("id", interview_id).execute()
        _last_error = None
        return True
    except Exception as e:
        _last_error = f"update_interview failed: {e}"
        return False


def fetch_interviews(limit: int = 500) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        q = client.table("interviews").select("*").order("scheduled_at", desc=False).limit(limit)
        company_id = _current_company_id()
        if company_id is not None:
            q = q.eq("company_id", company_id)
        result = q.execute()
        return result.data or []
    except Exception:
        return []
