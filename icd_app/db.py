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


@st.cache_resource
def _get_client():
    """Cached Supabase client — created once per app session, not per call."""
    global _last_error
    from supabase import create_client
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        _last_error = f"Client creation failed: {e}"
        return None


def save_screening_record(candidate: dict, job_role: str, job_details: str, job_id=None) -> bool:
    """
    Persist one screened candidate to Supabase. Returns True on success,
    False otherwise (including "not configured") — callers should treat
    False as a no-op, never as an error to surface to the user mid-screening.
    job_id is optional — links this screening to a saved job posting (Phase 5)
    when one was selected; ad-hoc screenings without a saved job simply omit it.
    """
    client = _get_client()
    if client is None:
        return False
    try:
        p, s = candidate["profile"], candidate["score"]
        row = {
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
        }
        client.table("screening_history").insert(row).execute()
        return True
    except Exception:
        return False


def fetch_screening_history(limit: int = 1000) -> list[dict]:
    """
    Fetch historical screening records across all past sessions. Returns an
    empty list if Supabase isn't configured or the query fails — Analytics
    should fall back to session-only data in that case, not error out.
    """
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("screening_history")
            .select("*")
            .order("screened_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = result.data or []
        # jsonb columns come back as native lists/dicts already via supabase-py,
        # but guard against string-encoded values just in case.
        for row in rows:
            for field in ("skills", "past_roles", "matched_skills", "gaps"):
                if isinstance(row.get(field), str):
                    try:
                        row[field] = json.loads(row[field])
                    except Exception:
                        row[field] = []
        return rows
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
        row = {**job, "status": job.get("status", "active")}
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
        result = client.table("interviews").insert(interview).execute()
        _last_error = None
        return (result.data or [None])[0]
    except Exception as e:
        _last_error = f"save_interview failed: {e}"
        return None


def update_interview(interview_id: int, updates: dict) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("interviews").update(updates).eq("id", interview_id).execute()
        return True
    except Exception:
        return False


def fetch_interviews(limit: int = 500) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("interviews")
            .select("*")
            .order("scheduled_at", desc=False)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception:
        return []
