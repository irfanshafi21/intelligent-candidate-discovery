"""
Report generation — Candidate, Shortlist, Interview, and Hiring reports,
exportable as PDF, Excel, or CSV. All functions operate on real data passed
in (candidates already screened, interviews already scheduled) — nothing
here fabricates figures.
"""

import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ----------------------------- DataFrame builders -----------------------------

def candidates_to_dataframe(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        if c["score"].get("error"):
            continue
        p, s = c["profile"], c["score"]
        b = s.get("breakdown", {})
        rows.append({
            "Name": c["name"],
            "Overall Score": s.get("overall_score"),
            "Skills Match": b.get("skills_match"),
            "Experience Fit": b.get("experience_fit"),
            "Education Fit": b.get("education_fit"),
            "Years Experience": p.get("years_experience"),
            "Education": p.get("education"),
            "Matched Skills": ", ".join(s.get("matched_skills", [])),
            "Gaps": ", ".join(s.get("gaps", [])),
            "Email": p.get("email"),
            "Phone": p.get("phone"),
        })
    return pd.DataFrame(rows)


def interviews_to_dataframe(interviews: list[dict]) -> pd.DataFrame:
    rows = []
    for i in interviews:
        rows.append({
            "Candidate": i.get("candidate_name"),
            "Job Role": i.get("job_role"),
            "Type": i.get("interview_type"),
            "Scheduled At": i.get("scheduled_at"),
            "Status": i.get("status"),
            "Interview Score": i.get("interview_score"),
            "Score Locked": "Yes" if i.get("score_locked") else "No",
            "Notes": i.get("notes"),
        })
    return pd.DataFrame(rows)


# ----------------------------- Export helpers -----------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


# ----------------------------- PDF reports -----------------------------

def _pdf_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("RTitle", parent=styles["Heading1"], fontSize=18, spaceAfter=6, textColor=colors.HexColor("#0F172A")),
        "sub": ParagraphStyle("RSub", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#64748B"), spaceAfter=14),
        "h2": ParagraphStyle("RH2", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#0EA5E9")),
        "body": ParagraphStyle("RBody", parent=styles["Normal"], fontSize=10.5, leading=15),
    }


def build_candidate_report_pdf(candidate: dict, job_role: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch,
                             leftMargin=0.7*inch, rightMargin=0.7*inch)
    st_ = _pdf_styles()
    p, s = candidate["profile"], candidate["score"]
    b = s.get("breakdown", {})

    content = [
        Paragraph(f"Candidate Report — {candidate['name']}", st_["title"]),
        Paragraph(f"Screened for: {job_role or '—'}", st_["sub"]),
        Paragraph("Overall Assessment", st_["h2"]),
        Paragraph(f"Overall Score: <b>{s.get('overall_score','—')}/100</b>", st_["body"]),
        Paragraph(f"Skills Match: {b.get('skills_match','—')}/100 &nbsp;|&nbsp; "
                   f"Experience Fit: {b.get('experience_fit','—')}/100 &nbsp;|&nbsp; "
                   f"Education Fit: {b.get('education_fit','—')}/100", st_["body"]),
        Paragraph("Recruiter Summary", st_["h2"]),
        Paragraph(s.get("summary", "—"), st_["body"]),
        Paragraph("Matched Skills", st_["h2"]),
        Paragraph(", ".join(s.get("matched_skills", [])) or "—", st_["body"]),
        Paragraph("Gaps", st_["h2"]),
        Paragraph(", ".join(s.get("gaps", [])) or "—", st_["body"]),
        Paragraph("Profile Details", st_["h2"]),
        Paragraph(f"Experience: {p.get('years_experience','—')}", st_["body"]),
        Paragraph(f"Education: {p.get('education','—')}", st_["body"]),
        Paragraph(f"Contact: {p.get('email','—')} · {p.get('phone','—')}", st_["body"]),
    ]
    doc.build(content)
    return buf.getvalue()


def build_shortlist_report_pdf(candidates: list[dict], job_role: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch,
                             leftMargin=0.6*inch, rightMargin=0.6*inch)
    st_ = _pdf_styles()
    ranked = sorted(
        [c for c in candidates if not c["score"].get("error")],
        key=lambda c: c["score"].get("overall_score", 0), reverse=True,
    )

    content = [
        Paragraph("Shortlist Report", st_["title"]),
        Paragraph(f"Job Role: {job_role or '—'} · {len(ranked)} candidate(s)", st_["sub"]),
    ]

    table_data = [["#", "Name", "Score", "Skills", "Experience", "Education"]]
    for idx, c in enumerate(ranked, start=1):
        p, s = c["profile"], c["score"]
        b = s.get("breakdown", {})
        table_data.append([
            str(idx), c["name"], f"{s.get('overall_score','—')}",
            f"{b.get('skills_match','—')}", f"{b.get('experience_fit','—')}", f"{b.get('education_fit','—')}",
        ])

    table = Table(table_data, colWidths=[0.3*inch, 1.8*inch, 0.6*inch, 0.7*inch, 0.9*inch, 0.9*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0EA5E9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    content.append(table)
    doc.build(content)
    return buf.getvalue()


def build_interview_report_pdf(interviews: list[dict]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch,
                             leftMargin=0.6*inch, rightMargin=0.6*inch)
    st_ = _pdf_styles()
    content = [
        Paragraph("Interview Report", st_["title"]),
        Paragraph(f"{len(interviews)} interview(s)", st_["sub"]),
    ]
    table_data = [["Candidate", "Type", "Scheduled", "Status", "Score", "Locked"]]
    for i in interviews:
        score = i.get("interview_score")
        table_data.append([
            i.get("candidate_name", "—"), i.get("interview_type", "—"),
            str(i.get("scheduled_at", "—"))[:16].replace("T", " "),
            i.get("status", "—"), f"{score}/100" if score is not None else "—",
            "Locked" if i.get("score_locked") else "No",
        ])
    table = Table(table_data, colWidths=[1.4*inch, 1.0*inch, 1.4*inch, 0.9*inch, 0.7*inch, 0.6*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0EA5E9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    content.append(table)
    doc.build(content)
    return buf.getvalue()
