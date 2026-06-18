import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import html

# 1 GB upload limit for Streamlit file uploaders (value is in MB)
UPLOAD_LIMIT_MB = 1024

st.set_page_config(
    page_title="AI Recruiter | India Runs Hackathon",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root{
    --bg:#F8FAFC;
    --primary:#0F172A;
    --primary-soft:#1E293B;
    --accent:#06B6D4;
    --accent-soft:#E0F7FB;
    --card:#FFFFFF;
    --text:#111827;
    --muted:#64748B;
    --border:#E2E8F0;
    --success:#059669;
    --warning:#D97706;
}

html, body, [class*="css"]{
    font-family:'Inter',sans-serif;
}

#MainMenu, footer, header{
    visibility:hidden;
}

.stApp{
    background:var(--bg);
}

.block-container{
    padding:.75rem 1.15rem 1.3rem !important;
    max-width:1480px !important;
}

.hero{
    position:relative;
    overflow:hidden;
    border-radius:20px;
    padding:1.05rem 1.35rem;
    margin:.15rem 0 .7rem;
    color:#FFFFFF;
    background:var(--primary);
    border:1px solid var(--primary-soft);
    box-shadow:0 18px 42px rgba(15,23,42,.14);
}

.hero > *{
    position:relative;
    z-index:1;
}

.hero-badge{
    display:inline-flex;
    align-items:center;
    gap:7px;
    background:rgba(6,182,212,.16);
    border:1px solid rgba(6,182,212,.32);
    color:#CFFAFE;
    font-size:11.5px;
    font-weight:800;
    padding:5px 13px;
    border-radius:999px;
}

.hero h1{
    font-family:'Space Grotesk',sans-serif;
    font-size:clamp(1.65rem,2.7vw,2.7rem);
    line-height:1.05;
    font-weight:900;
    margin:.5rem 0 .25rem;
    letter-spacing:-.8px;
    color:#FFFFFF;
}

.hero p{
    color:#CBD5E1;
    max-width:980px;
    font-size:.93rem;
    line-height:1.4;
    margin:0;
}

.panel-card{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:20px;
    padding:.82rem .95rem;
    box-shadow:0 12px 30px rgba(15,23,42,.06);
    margin-bottom:.65rem;
}

.control-title{
    font-family:'Space Grotesk',sans-serif;
    font-weight:900;
    font-size:1rem;
    color:var(--primary);
    margin:0 0 .6rem;
}

.small-title{
    font-family:'Space Grotesk',sans-serif;
    font-weight:900;
    font-size:.88rem;
    color:var(--primary);
    margin:0 0 .35rem;
}

.mini-label{
    font-size:11px;
    font-weight:800;
    color:#64748B;
    letter-spacing:.4px;
    margin-bottom:4px;
}

[data-testid="stTextArea"] textarea{
    border-radius:16px !important;
    border:1px solid var(--border) !important;
    background:#FFFFFF !important;
    box-shadow:inset 0 1px 2px rgba(15,23,42,.04) !important;
    color:var(--text) !important;
    font-size:.86rem !important;
    line-height:1.35 !important;
}

[data-testid="stTextArea"] textarea:focus{
    border-color:var(--accent) !important;
    box-shadow:0 0 0 3px rgba(6,182,212,.12) !important;
}

.tag{
    display:inline-flex;
    align-items:center;
    background:var(--accent-soft);
    color:#0E7490;
    font-size:10.3px;
    font-weight:800;
    padding:5px 9px;
    border-radius:999px;
    margin:3px;
    border:1px solid #BAE6FD;
}

.weight-mini{
    background:#F8FAFC;
    border:1px solid var(--border);
    border-radius:15px;
    padding:.46rem .62rem;
    margin-top:.08rem;
}

[data-testid="stSlider"]{
    padding:.02rem 0 .06rem !important;
}

[data-testid="stSlider"] label{
    font-size:.76rem !important;
    color:#334155 !important;
    font-weight:800 !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"]{
    margin-top:0 !important;
}

.stSlider label, .stNumberInput label, .stFileUploader label{
    font-weight:800 !important;
    color:#334155 !important;
}

.upload-card{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:20px;
    padding:.82rem .95rem .9rem;
    box-shadow:0 12px 30px rgba(15,23,42,.06);
    margin-bottom:.65rem;
}

[data-testid="stFileUploader"] section{
    border-radius:17px !important;
    border:1.5px dashed #94A3B8 !important;
    background:#F8FAFC !important;
    padding:.7rem !important;
}

[data-testid="stFileUploader"] section:hover{
    border-color:var(--accent) !important;
    background:#F0FDFA !important;
}

[data-testid="stFileUploader"] small{
    color:var(--muted) !important;
}

.run-card{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:20px;
    padding:.72rem .95rem;
    box-shadow:0 12px 30px rgba(15,23,42,.06);
    margin-bottom:.75rem;
}

.stButton > button{
    background:var(--primary) !important;
    color:#FFFFFF !important;
    border:1px solid var(--primary) !important;
    border-radius:18px !important;
    font-weight:900 !important;
    font-size:1.05rem !important;
    letter-spacing:.3px !important;
    min-height:64px !important;
    padding:.85rem 1.5rem !important;
    box-shadow:0 16px 34px rgba(15,23,42,.18) !important;
    transition:all .18s ease !important;
}

.stButton > button:hover{
    background:#164E63 !important;
    border-color:#164E63 !important;
    transform:translateY(-1px);
    box-shadow:0 20px 42px rgba(22,78,99,.22) !important;
}

.stDownloadButton > button{
    background:#FFFFFF !important;
    color:var(--primary) !important;
    border:1px solid var(--border) !important;
    border-radius:16px !important;
    font-weight:850 !important;
    min-height:48px !important;
}

.stDownloadButton > button:hover{
    border-color:var(--accent) !important;
    color:#0E7490 !important;
}

.section-title{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.03rem;
    font-weight:900;
    color:var(--primary);
    margin:.05rem 0 .52rem;
}

.metric-row{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:.85rem;
    margin:.75rem 0 1rem;
}

.metric-card{
    background:#FFFFFF;
    border:1px solid var(--border);
    border-top:4px solid var(--accent);
    border-radius:18px;
    padding:1rem .85rem;
    text-align:center;
    box-shadow:0 12px 28px rgba(15,23,42,.06);
}

.metric-card .label{
    font-size:10.5px;
    font-weight:900;
    color:#0E7490;
    text-transform:uppercase;
    letter-spacing:.85px;
    margin-bottom:7px;
}

.metric-card .value{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.85rem;
    font-weight:900;
    color:var(--primary);
    line-height:1;
}

.metric-card .sub{
    font-size:11.5px;
    color:var(--muted);
    margin-top:6px;
}

.candidate-card{
    background:#FFFFFF;
    border:1px solid var(--border);
    border-radius:17px;
    padding:.78rem .95rem;
    margin-bottom:9px;
    display:flex;
    align-items:center;
    gap:1rem;
    box-shadow:0 10px 24px rgba(15,23,42,.055);
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}

.candidate-card:hover{
    transform:translateY(-2px);
    box-shadow:0 18px 36px rgba(15,23,42,.10);
    border-color:#BAE6FD;
}

.recommend-card{
    background:#F8FAFC;
    border:1px solid #BAE6FD;
    border-left:5px solid var(--accent);
    border-radius:18px;
    padding:1rem 1.1rem;
    margin:.25rem 0 1rem;
    box-shadow:0 12px 28px rgba(15,23,42,.055);
}

.recommend-kicker{
    font-size:10.5px;
    font-weight:900;
    color:#0E7490;
    text-transform:uppercase;
    letter-spacing:.8px;
    margin-bottom:6px;
}

.recommend-title{
    font-family:'Space Grotesk',sans-serif;
    color:var(--primary);
    font-size:1.22rem;
    font-weight:900;
    margin-bottom:4px;
}

.recommend-body{
    color:#475569;
    font-size:.88rem;
    line-height:1.45;
    font-weight:650;
}

.why-line{
    display:flex;
    align-items:center;
    flex-wrap:wrap;
    gap:5px;
    margin-top:7px;
}

.why-label{
    color:#0F172A;
    font-size:10.5px;
    font-weight:900;
}

.why-chip{
    background:#E0F7FB;
    color:#0E7490;
    border:1px solid #BAE6FD;
    border-radius:999px;
    padding:3px 7px;
    font-size:10px;
    font-weight:850;
}

.why-pill{
    background:#F1F5F9;
    color:#334155;
    border:1px solid #E2E8F0;
    border-radius:999px;
    padding:3px 7px;
    font-size:10px;
    font-weight:800;
}

.rank-badge{
    width:40px;
    height:40px;
    border-radius:13px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:900;
    font-size:14px;
    flex-shrink:0;
}

.rank-1{ background:#F59E0B; color:#FFFFFF; }
.rank-2{ background:#64748B; color:#FFFFFF; }
.rank-3{ background:#B45309; color:#FFFFFF; }
.rank-other{ background:#E0F7FB; color:#0E7490; }
.score-bar-bg{ background:#E2E8F0; border-radius:999px; height:8px; width:100%; overflow:hidden; }
.score-bar-fill{ height:100%; border-radius:999px; background:var(--accent); }
hr{ display:none !important; }

[data-testid="stAlert"]{
    border-radius:16px !important;
    border:1px solid var(--border) !important;
}

@media(max-width:1120px){
    .metric-row{ grid-template-columns:repeat(2,1fr); }
}

@media(max-width:760px){
    .metric-row{ grid-template-columns:1fr; }
    .candidate-card{ align-items:flex-start; flex-direction:column; }
    .hero{ padding:1rem; }
    .block-container{ padding:.6rem .75rem 1.1rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🇮🇳 India Runs Hackathon 2026 · Track 01 · Data &amp; AI</div>
    <h1>Intelligent Candidate Discovery</h1>
    <p>AI-powered recruiter that shortlists the best candidates for any job role in seconds using NLP & machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ── ML Skills & Honeypot config ──────────────────────────────────────────────
ML_SKILLS = {
    'python','machine learning','deep learning','tensorflow','pytorch','scikit-learn',
    'nlp','natural language processing','llm','transformer','bert','gpt',
    'fine-tuning llms','lora','neural networks','mlops','model deployment',
    'feature engineering','data analysis','sql','vector databases','milvus',
    'faiss','pinecone','aws','gcp','azure','data pipelines','speech recognition',
    'image classification','computer vision','weights & biases','wandb','bentoml',
    'gans','statistical modeling','spark','airflow','kafka','rlhf','hugging face',
    'xgboost','lightgbm','random forest','mlflow','docker','kubernetes',
    'reinforcement learning','time series','anomaly detection','recommendation systems',
    'graph neural networks','object detection','yolo','opencv','pyspark'
}

HONEYPOT_TITLES = [
    'marketing manager','operations manager','hr manager','content writer',
    'business analyst','accountant','mechanical engineer','logistics manager',
    'sales manager','finance manager','supply chain','procurement manager',
    'administrative','customer service','receptionist','retail manager',
    'project coordinator','office manager','legal counsel','civil engineer'
]

proficiency_map = {'advanced':3,'intermediate':2,'beginner':1}
tier_map = {'tier_1':100,'tier_2':70,'tier_3':40}
degree_map = {'ph.d':100,'phd':100,'m.tech':85,'m.sc':80,'mca':75,'b.tech':60,'b.e.':60,'b.sc':55,'m.e.':85,'m.s.':82}
ai_fields = ['artificial intelligence','machine learning','computer science','data science','deep learning','statistics']

def is_honeypot(c):
    title = c['profile'].get('current_title','').lower()
    skills = [s['name'].lower() for s in c.get('skills',[])]
    skill_text = ' '.join(skills)
    career_text = ' '.join([(j.get('description','')+' '+j.get('title','')).lower() for j in c.get('career_history',[])])
    is_non_ml = any(f in title for f in HONEYPOT_TITLES)
    ml_skill_count = sum(1 for s in ML_SKILLS if s in skill_text)
    career_ml = sum(1 for kw in ['machine learning','python','ai ','nlp','deep learning','data science','neural'] if kw in career_text)
    if is_non_ml and ml_skill_count < 3 and career_ml < 2:
        return True
    if ml_skill_count > 12 and career_ml < 1:
        return True
    return False

def score_jsonl_candidate(c, jd_vec, vectorizer):
    p = c['profile']; sig = c.get('redrob_signals',{}); skills = c.get('skills',[]); career = c.get('career_history',[]); edu = c.get('education',[])
    skill_text = ' '.join([s['name'] for s in skills]+[p.get('headline',''),p.get('summary','')])
    try:
        cv = vectorizer.transform([skill_text])
        ss = cosine_similarity(jd_vec, cv).flatten()[0]*100
    except: ss = 0
    ml_cnt=0; ml_prof=0
    for s in skills:
        if s['name'].lower() in ML_SKILLS:
            ml_cnt+=1; ml_prof+=proficiency_map.get(s.get('proficiency','beginner'),1); ml_prof+=min(s.get('endorsements',0)/20,2)
    ml_depth=min(ml_cnt*5+ml_prof*2,100)
    ml_months=sum(j.get('duration_months',0) for j in career if any(kw in (j.get('description','')+' '+j.get('title','')).lower() for kw in ['machine learning','ml ','ai ','deep learning','nlp','data science','python','neural']))
    ml_exp=min(ml_months/36*100,100)
    exp_s=min(p.get('years_of_experience',0)/10*100,100)
    gh=sig.get('github_activity_score',-1)
    act=(min(sig.get('profile_completeness_score',0),100)*0.15+(100 if sig.get('open_to_work_flag') else 30)*0.10+(min(gh,100) if gh>0 else 0)*0.20+min(sig.get('profile_views_received_30d',0)/50*100,100)*0.10+sig.get('interview_completion_rate',0)*100*0.15+sig.get('recruiter_response_rate',0)*100*0.10+(100 if sig.get('verified_email') else 0)*0.10+min(sig.get('saved_by_recruiters_30d',0)/10*100,100)*0.10)
    edu_s=max([tier_map.get(e.get('tier','tier_3'),40)*0.5+degree_map.get(e.get('degree','').lower(),50)*0.3+(20 if any(f in e.get('field_of_study','').lower() for f in ai_fields) else 0) for e in edu],default=0)
    asm=sig.get('skill_assessment_scores',{})
    assess=np.mean(list(asm.values())) if asm else 0
    final=ss*0.25+ml_depth*0.25+ml_exp*0.15+exp_s*0.10+act*0.15+edu_s*0.05+assess*0.05
    top_skills=[s['name'] for s in skills if s['name'].lower() in ML_SKILLS][:4]
    skill_str=', '.join(top_skills) if top_skills else 'technical skills'
    strength="Strong" if final>=70 else "Good" if final>=50 else "Moderate"
    reasoning=f"{strength} ML/AI candidate with {p.get('years_of_experience',0):.1f} yrs as {p.get('current_title','')}; key skills: {skill_str}. ML depth {ml_depth:.0f}/100, assessment {assess:.0f}/100."
    return {
        'candidate_id':c['candidate_id'],'name':p.get('anonymized_name',''),
        'job_title':p.get('current_title',''),'experience_years':p.get('years_of_experience',0),
        'skills':' '.join([s['name'] for s in skills]),
        'education':edu[0].get('degree','') if edu else '',
        'skill_match_score':round(ss,2),'ml_depth':round(ml_depth,2),
        'activity_score':round(act,2),'final_score':round(final,2),'reasoning':reasoning
    }

def pretty_keyword(term):
    term = str(term).strip()
    upper_terms = {'ai','ml','nlp','llm','gpt','bert','cnn','sql','aws','gcp','api','rest','ui','ux','etl','mlops','automl','xgboost','opencv','faiss'}
    if term.lower() in upper_terms:
        return term.upper()
    if term.lower() == 'tensorflow':
        return 'TensorFlow'
    if term.lower() == 'scikit-learn':
        return 'scikit-learn'
    return term.title()

def get_candidate_reason_parts(row, job_keywords):
    skills_text = str(row.get('skills', '')).lower()
    matched_terms = []
    for kw in job_keywords:
        clean_kw = str(kw).strip().lower()
        if clean_kw and clean_kw in skills_text and clean_kw not in matched_terms:
            matched_terms.append(clean_kw)
        if len(matched_terms) >= 5:
            break

    if not matched_terms:
        fallback_terms = []
        for term in str(row.get('skills', '')).split():
            clean_term = term.strip('.,()').lower()
            if len(clean_term) > 3 and clean_term not in fallback_terms:
                fallback_terms.append(clean_term)
            if len(fallback_terms) >= 4:
                break
        matched_terms = fallback_terms

    try:
        activity_value = int(row.get('activity_score', 0))
    except Exception:
        activity_value = row.get('activity_score', 0)

    exp_reason = f"{row.get('experience_years', '')} yrs exp"
    act_reason = f"Activity {activity_value}/100"
    return matched_terms, exp_reason, act_reason

def make_reason_chips(matched_terms, exp_reason, act_reason):
    skill_chips = ''.join(
        f'<span class="why-chip">{html.escape(pretty_keyword(term))}</span>'
        for term in matched_terms[:5]
    )
    return (
        f'{skill_chips}'
        f'<span class="why-pill">{html.escape(str(exp_reason))}</span>'
        f'<span class="why-pill">{html.escape(str(act_reason))}</span>'
    )

def create_ranked_pdf_report(df_ranked, top_n, recommendation_reason, explanation_keywords, skill_weight, exp_weight, act_weight):
    """Create a compact recruiter PDF report as bytes for st.download_button."""
    buffer = BytesIO()
    top_pdf = df_ranked.head(top_n).copy()

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        top1 = df_ranked.iloc[0]
        fig.text(0.07, 0.94, 'Intelligent Candidate Discovery', fontsize=19, fontweight='bold', color='#0F172A')
        fig.text(0.07, 0.915, 'Ranked candidate report generated from AI recruiter dashboard', fontsize=9.5, color='#64748B')

        fig.text(0.07, 0.865, 'Recruiter Recommendation', fontsize=12.5, fontweight='bold', color='#0F172A')
        fig.text(0.07, 0.842, f"Recommended Hire: {top1.get('name', '')}", fontsize=11, fontweight='bold', color='#0E7490')
        fig.text(0.07, 0.812, recommendation_reason, fontsize=8.8, color='#334155', wrap=True)

        summary_lines = [
            f"Top Score: {df_ranked['final_score'].max():.1f}/100",
            f"Total Candidates: {len(df_ranked)}",
            f"Average Score: {df_ranked['final_score'].mean():.1f}",
            f"Weights: Skill {skill_weight}% | Experience {exp_weight}% | Activity {act_weight}%"
        ]
        y = 0.765
        for line in summary_lines:
            fig.text(0.07, y, line, fontsize=9, color='#0F172A')
            y -= 0.022

        fig.text(0.07, 0.67, f'Top {top_n} Ranked Candidates', fontsize=12.5, fontweight='bold', color='#0F172A')

        table_cols = ['Rank', 'Name', 'Job Title', 'Exp', 'Skill %', 'Activity', 'Final']
        table_rows = []
        for _, row in top_pdf.iterrows():
            table_rows.append([
                int(row.get('rank', 0)),
                str(row.get('name', ''))[:22],
                str(row.get('job_title', ''))[:26],
                str(row.get('experience_years', '')),
                f"{row.get('skill_match_score', 0):.1f}",
                f"{row.get('activity_score', 0):.0f}",
                f"{row.get('final_score', 0):.1f}",
            ])

        table_ax = fig.add_axes([0.06, 0.26, 0.88, 0.38])
        table_ax.axis('off')
        table = table_ax.table(
            cellText=table_rows,
            colLabels=table_cols,
            loc='upper center',
            cellLoc='left',
            colLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.35)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#E2E8F0')
            if r == 0:
                cell.set_facecolor('#0F172A')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#FFFFFF' if r % 2 else '#F8FAFC')
                cell.set_text_props(color='#0F172A')

        fig.text(0.07, 0.20, 'Why top candidates ranked high', fontsize=12.5, fontweight='bold', color='#0F172A')
        y = 0.175
        for _, row in df_ranked.head(min(5, top_n)).iterrows():
            matched_terms, exp_reason, act_reason = get_candidate_reason_parts(row, explanation_keywords)
            terms = ', '.join(pretty_keyword(term) for term in matched_terms[:4]) if matched_terms else 'Relevant skills'
            line = f"#{int(row.get('rank', 0))} {row.get('name', '')}: {terms}; {exp_reason}; {act_reason}; Final {row.get('final_score', 0):.1f}"
            fig.text(0.07, y, line, fontsize=8.2, color='#334155')
            y -= 0.023

        fig.text(0.07, 0.055, 'Generated by Intelligent Candidate Discovery - India Runs Hackathon 2026', fontsize=7.8, color='#64748B')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()

# ── Horizontal Control Section ────────────────────────────────────────────────
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">⚙️ Recruiter Control Panel</div>', unsafe_allow_html=True)

jd_col, key_col, weight_col, select_col = st.columns([2.05, 1.35, 2.35, .9], gap="medium")

default_job_description = """We are looking for a Senior AI/ML Engineer with strong Python skills.
Required: machine learning, deep learning, TensorFlow, PyTorch, scikit-learn,
natural language processing, LLMs, transformers, BERT, GPT, model deployment, MLOps.
Fine-tuning LLMs, LoRA, vector databases preferred.
Minimum 3 years ML/AI experience. Strong problem solving skills required."""

current_job_text = st.session_state.get("job_description_text", default_job_description)
line_count = max(4, len(current_job_text.splitlines()))
auto_textarea_height = min(210, max(118, line_count * 25 + 28))

with jd_col:
    st.markdown('<div class="small-title">📋 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste or type any job description:",
        value=default_job_description,
        height=auto_textarea_height,
        label_visibility="collapsed",
        key="job_description_text"
    )

with key_col:
    st.markdown('<div class="small-title">🔍 Keywords</div>', unsafe_allow_html=True)
    keywords = []
    if job_description:
        stop_words = {'and','or','the','a','an','is','are','we','for','with','in',
                      'of','to','be','experience','skills','required','preferred',
                      'minimum','years','strong','looking','problem','solving'}
        words = [w.strip('.,()') for w in job_description.lower().split()
                 if len(w) > 3 and w.lower().strip('.,()') not in stop_words]
        keywords = list(dict.fromkeys(words))[:10]
        tags_html = "".join([f'<span class="tag">{k}</span>' for k in keywords])
        st.markdown(tags_html, unsafe_allow_html=True)

with weight_col:
    st.markdown('<div class="small-title">⚖️ Scoring Weights</div>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3, gap="small")
    with w1:
        skill_weight = st.slider("🎯 Skill", 0, 100, 70, step=5)
    with w2:
        exp_weight = st.slider("📅 Exp", 0, 100, 20, step=5)
    with w3:
        act_weight = st.slider("⚡ Act", 0, 100, 10, step=5)
    total = skill_weight + exp_weight + act_weight
    if total != 100:
        st.error(f"⚠️ Weights = {total}% · Must be 100%")
    else:
        st.markdown(f"""
        <div class='weight-mini'>
            <div style='display:flex;gap:4px;height:8px;border-radius:8px;overflow:hidden'>
                <div style='width:{skill_weight}%;background:#0F172A'></div>
                <div style='width:{exp_weight}%;background:#06B6D4'></div>
                <div style='width:{act_weight}%;background:#94A3B8'></div>
            </div>
            <div style='display:flex;gap:9px;margin-top:6px;font-size:10.5px;color:#475569;flex-wrap:wrap;font-weight:800'>
                <span>Skill {skill_weight}%</span>
                <span>Exp {exp_weight}%</span>
                <span>Act {act_weight}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with select_col:
    st.markdown('<div class="small-title">👥 Top N</div>', unsafe_allow_html=True)
    top_n = st.slider("Number of Candidates", 3, 20, 10, label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

# ── Upload Section (single unified slot — JSONL, JSON, CSV all accepted) ──────
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="small-title">📂 Upload Candidate Data</div>', unsafe_allow_html=True)
st.caption(f"Accepts .jsonl · .json · .csv — select one or more files · Upload limit: {UPLOAD_LIMIT_MB} MB (1 GB) per file")

st.markdown('<div class="mini-label">CANDIDATE FILES (JSONL / JSON / CSV — SELECT ANY COMBINATION)</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload candidate files",
    type=["jsonl", "json", "csv"],
    accept_multiple_files=True,
    max_upload_size=UPLOAD_LIMIT_MB,
    label_visibility="collapsed"
)

# Split uploads by extension so downstream logic stays identical
jsonl_file = next(
    (f for f in (uploaded_files or []) if f.name.lower().endswith(('.jsonl', '.json'))),
    None
)
csv_files = [f for f in (uploaded_files or []) if f.name.lower().endswith('.csv')]

st.markdown('</div>', unsafe_allow_html=True)

# ── Full-width Run Section ─────────────────────────────────────────────────────
st.markdown('<div class="run-card">', unsafe_allow_html=True)
run = st.button("🚀 RUN AI RANKING", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data
def get_default_data():
    np.random.seed(42)
    data = {
        'candidate_id': range(1, 51),
        'name': [f'Candidate_{i}' for i in range(1, 51)],
        'skills': [
            'Python machine learning deep learning TensorFlow data analysis',
            'Java Spring Boot REST API microservices Docker',
            'Python data science pandas numpy scikit-learn visualization',
            'JavaScript React Node.js HTML CSS frontend development',
            'Python AI NLP transformer BERT GPT language models',
            'SQL database MySQL PostgreSQL data warehousing ETL',
            'Python TensorFlow keras neural networks computer vision',
            'C++ algorithms data structures competitive programming',
            'Python scikit-learn random forest XGBoost feature engineering',
            'AWS cloud computing DevOps CI/CD Jenkins Kubernetes',
            'Python NLP text classification sentiment analysis spacy',
            'R statistics data analysis regression modeling visualization',
            'Python deep learning CNN image recognition OpenCV',
            'JavaScript TypeScript Angular frontend UI UX design',
            'Python reinforcement learning OpenAI gym reward optimization',
            'Tableau Power BI data visualization business intelligence',
            'Python time series forecasting ARIMA LSTM prediction',
            'Java Android mobile development Kotlin Firebase',
            'Python recommendation system collaborative filtering',
            'Cybersecurity ethical hacking penetration testing network',
            'Python pandas data cleaning wrangling EDA exploratory',
            'Machine learning model deployment Flask FastAPI REST',
            'Python graph neural networks knowledge graphs embeddings',
            'Blockchain Solidity smart contracts Web3 cryptocurrency',
            'Python AutoML hyperparameter tuning model optimization',
            'Data engineering Apache Spark Kafka Hadoop big data',
            'Python GANs generative adversarial networks image synthesis',
            'iOS Swift mobile development Xcode Apple development',
            'Python anomaly detection fraud detection outlier analysis',
            'Natural language processing question answering chatbot LLM',
            'Python ensemble methods bagging boosting stacking models',
            'UI UX design Figma Adobe XD prototyping user research',
            'Python transfer learning pretrained models fine-tuning BERT',
            'Robotics ROS automation embedded systems IoT sensors',
            'Python multimodal AI vision language models CLIP DALL-E',
            'Data science statistics hypothesis testing A/B testing',
            'Python object detection YOLO real-time video processing',
            'Game development Unity C# 3D modeling game design',
            'Python causal inference econometrics experimental design',
            'Search engine Elasticsearch Solr information retrieval ranking',
            'Python vector databases FAISS Pinecone semantic search',
            'Agile project management Scrum product management roadmap',
            'Python MLOps model monitoring Weights Biases MLflow',
            'Network engineering TCP/IP routing switching firewall',
            'Python speech recognition audio processing whisper NLP',
            'Business analytics Excel VBA financial modeling forecasting',
            'Python federated learning privacy preserving distributed ML',
            'Cloud ML GCP Vertex AI Azure ML SageMaker deployment',
            'Python knowledge distillation model compression edge AI',
            'Full stack Python Django React PostgreSQL REST API deployment'
        ],
        'experience_years': [5,3,4,2,6,7,5,2,4,6,3,5,4,2,3,6,5,2,4,3,
                             4,5,6,2,3,7,4,2,5,6,3,2,5,4,3,6,4,2,5,7,
                             4,3,5,2,4,6,3,5,4,6],
        'job_title': [
            'ML Engineer','Java Developer','Data Scientist','Frontend Dev',
            'AI Engineer','Database Admin','Deep Learning Engineer','Software Dev',
            'ML Engineer','Cloud Engineer','NLP Engineer','Data Analyst',
            'Computer Vision Engineer','Frontend Dev','RL Engineer','BI Analyst',
            'Data Scientist','Mobile Dev','ML Engineer','Security Engineer',
            'Data Analyst','ML Engineer','AI Researcher','Blockchain Dev',
            'AutoML Engineer','Data Engineer','AI Researcher','iOS Dev',
            'Fraud Detection Engineer','NLP Engineer','ML Engineer','UI Designer',
            'NLP Engineer','Robotics Engineer','Multimodal AI Engineer',
            'Data Scientist','Computer Vision Engineer','Game Dev','Data Scientist',
            'Search Engineer','ML Engineer','Product Manager','MLOps Engineer',
            'Network Engineer','Speech Engineer','Business Analyst','ML Researcher',
            'Cloud ML Engineer','ML Engineer','Full Stack Developer'
        ],
        'activity_score': np.random.randint(60, 100, 50).tolist(),
        'education': ['B.Tech CS','MCA','M.Tech AI','B.Tech IT','PhD AI',
                      'B.Tech CS','M.Tech ML','B.Tech CS','M.Tech DS','B.Tech CS',
                      'M.Tech NLP','MSc Stats','M.Tech AI','B.Tech IT','PhD ML',
                      'MBA Analytics','M.Tech DS','B.Tech IT','M.Tech CS','B.Tech CS',
                      'B.Tech CS','M.Tech CS','PhD AI','B.Tech CS','M.Tech AI',
                      'M.Tech DE','PhD AI','B.Tech IT','M.Tech CS','PhD NLP',
                      'M.Tech ML','B.Des','PhD NLP','B.Tech Robotics','PhD AI',
                      'MSc Stats','M.Tech CS','B.Tech CS','PhD Stats','M.Tech CS',
                      'M.Tech AI','MBA','M.Tech CS','B.Tech CS','M.Tech CS',
                      'MBA','PhD ML','M.Tech CS','M.Tech AI','B.Tech CS']
    }
    return pd.DataFrame(data)

# ── Run Logic ───────────────────────────────────────────────────────────────
if run:
    if total != 100:
        st.error("❌ Weights must add up to 100%")
    elif not job_description.strip():
        st.error("❌ Please enter a job description")
    else:
        mode = None
        df = None
        df_ranked = None
        honeypots = 0

        with st.spinner("🤖 Analyzing candidates..."):

            # ── JSONL mode ───────────────────────────────────────────────
            if jsonl_file:
                mode = "jsonl"
                content = jsonl_file.read().decode('utf-8')
                candidates = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
                st.info(f"📂 Loaded {len(candidates):,} candidates from JSONL file")

                vectorizer = TfidfVectorizer(stop_words='english', max_features=8000)
                vectorizer.fit([job_description])
                jd_vec = vectorizer.transform([job_description])

                results = []
                for c in candidates:
                    if is_honeypot(c):
                        honeypots += 1
                        continue
                    results.append(score_jsonl_candidate(c, jd_vec, vectorizer))

                df_ranked = pd.DataFrame(results).sort_values('final_score', ascending=False).reset_index(drop=True)
                df_ranked['rank'] = df_ranked.index + 1

            # ── CSV mode ─────────────────────────────────────────────────
            else:
                mode = "csv"
                if csv_files:
                    dfs = [pd.read_csv(f) for f in csv_files]
                    df = pd.concat(dfs, ignore_index=True)
                    if len(df) < 5:
                        st.warning(f"⚠️ Only {len(df)} candidates found. Using default dataset.")
                        df = get_default_data()
                    else:
                        st.success(f"✅ {len(df)} candidates loaded from {len(csv_files)} file(s)!")
                else:
                    df = get_default_data()

                vectorizer = TfidfVectorizer(stop_words='english')
                all_text = [job_description] + list(df['skills'])
                tfidf_matrix = vectorizer.fit_transform(all_text)
                similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                df2 = df.copy()
                df2['skill_match_score'] = similarity_scores * 100
                df2['exp_score'] = (df2['experience_years'] / df2['experience_years'].max()) * 100
                df2['final_score'] = (
                    df2['skill_match_score'] * (skill_weight / 100) +
                    df2['exp_score']         * (exp_weight  / 100) +
                    df2['activity_score']    * (act_weight  / 100)
                )
                df2['rank'] = df2['final_score'].rank(ascending=False).astype(int)
                df_ranked = df2.sort_values('rank').reset_index(drop=True)

        st.success(
            f"✅ Ranking complete! {len(df_ranked):,} candidates scored" +
            (f", {honeypots:,} honeypots filtered." if mode == "jsonl" else ".")
        )

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        top1 = df_ranked.iloc[0]

        if mode == "jsonl":
            m3_label, m3_value, m3_sub = "📊 Scored", f"{len(df_ranked):,}", "candidates"
            m4_label, m4_value, m4_sub = "🚫 Filtered", f"{honeypots:,}", "honeypots removed"
        else:
            m3_label, m3_value, m3_sub = "📊 Total", f"{len(df_ranked):,}", "candidates analyzed"
            m4_label, m4_value, m4_sub = "📈 Avg Score", f"{df_ranked['final_score'].mean():.1f}", "across all"

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="label">👑 Top Candidate</div>
                <div class="value" style="font-size:1.1rem">{html.escape(str(top1['name']))}</div>
                <div class="sub">{html.escape(str(top1.get('job_title','')))}</div>
            </div>
            <div class="metric-card">
                <div class="label">🎯 Top Score</div>
                <div class="value">{df_ranked['final_score'].max():.1f}</div>
                <div class="sub">out of 100</div>
            </div>
            <div class="metric-card">
                <div class="label">{m3_label}</div>
                <div class="value">{m3_value}</div>
                <div class="sub">{m3_sub}</div>
            </div>
            <div class="metric-card">
                <div class="label">{m4_label}</div>
                <div class="value">{m4_value}</div>
                <div class="sub">{m4_sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        top1_terms, top1_exp_reason, top1_act_reason = get_candidate_reason_parts(top1, keywords)
        top1_terms_text = ', '.join(pretty_keyword(term) for term in top1_terms[:4]) if top1_terms else 'relevant role skills'
        recommendation_reason = (
            f"Highest final score ({top1['final_score']:.1f}/100), matched {top1_terms_text}, "
            f"{top1.get('experience_years', '')} years experience, and {top1_act_reason.lower()}."
        )
        st.markdown(f"""
        <div class="recommend-card">
            <div class="recommend-kicker">Recruiter Recommendation</div>
            <div class="recommend-title">Recommended Hire: {html.escape(str(top1['name']))}</div>
            <div class="recommend-body">{html.escape(recommendation_reason)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="section-title">🏆 Top {top_n} Candidates</div>', unsafe_allow_html=True)

        for i, row in df_ranked.head(top_n).iterrows():
            rank = int(row['rank'])
            badge_cls = "rank-1" if rank==1 else "rank-2" if rank==2 else "rank-3" if rank==3 else "rank-other"
            rank_icon = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else str(rank)
            score = row['final_score']
            skill = row['skill_match_score']
            matched_terms, exp_reason, act_reason = get_candidate_reason_parts(row, keywords)
            reason_html = make_reason_chips(matched_terms, exp_reason, act_reason)
            st.markdown(f"""
            <div class="candidate-card">
                <div class="rank-badge {badge_cls}">{rank_icon}</div>
                <div style="flex:1">
                    <div style="font-weight:600;color:#0F172A;font-size:15px">{html.escape(str(row['name']))}</div>
                    <div style="color:#64748B;font-size:12px;margin-top:2px">
                        {html.escape(str(row.get('job_title','')))} · {html.escape(str(row.get('experience_years','')))} yrs · {html.escape(str(row.get('education','')))}
                    </div>
                    <div class="why-line"><span class="why-label">Why ranked high:</span> {reason_html}</div>
                </div>
                <div style="min-width:140px">
                    <div style="font-size:11px;color:#0E7490;margin-bottom:4px">Skill Match: {skill:.1f}%</div>
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{min(int(score),100)}%"></div></div>
                </div>
                <div style="text-align:right;min-width:70px">
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#0E7490">{score:.1f}</div>
                    <div style="font-size:10px;color:#64748B">Final Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots(figsize=(5.2, 2.9))
            fig1.patch.set_facecolor('#FFFFFF')
            ax1.set_facecolor('#FFFFFF')
            top10 = df_ranked.head(10)
            colors = ['#0F172A' if i==0 else '#06B6D4' if i<3 else '#CBD5E1' for i in range(len(top10))]
            bars = ax1.barh(top10['name'], top10['final_score'], color=colors, height=0.42)
            ax1.set_xlabel('Final Score', color='#64748B', fontsize=7.5)
            ax1.set_title('Top 10 Candidates', color='#0F172A', fontsize=9.5, fontweight='bold', pad=5)
            ax1.invert_yaxis()
            ax1.set_xlim(0, 100)
            ax1.tick_params(colors='#64748B', labelsize=6.8)
            ax1.grid(axis='x', alpha=0.15, linewidth=0.7)
            for sp in ['top','right']: ax1.spines[sp].set_visible(False)
            ax1.spines['left'].set_color('#E2E8F0')
            ax1.spines['bottom'].set_color('#E2E8F0')
            for bar, val in zip(bars, top10['final_score']):
                ax1.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f'{val:.1f}', va='center', fontsize=6.8, color='#0E7490')
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(5.2, 2.9))
            fig2.patch.set_facecolor('#FFFFFF')
            ax2.set_facecolor('#FFFFFF')
            ax2.scatter(df_ranked['skill_match_score'], df_ranked['experience_years'], color='#06B6D4', s=38, alpha=0.82, edgecolors='#0F172A', linewidths=0.25)
            ax2.set_xlabel('Skill Match Score', color='#64748B', fontsize=7.5)
            ax2.set_ylabel('Experience (years)', color='#64748B', fontsize=7.5)
            ax2.set_title('Skill vs Experience', color='#0F172A', fontsize=9.5, fontweight='bold', pad=5)
            ax2.tick_params(colors='#64748B', labelsize=6.8)
            ax2.grid(alpha=0.15, linewidth=0.7)
            for sp in ['top','right']: ax2.spines[sp].set_visible(False)
            ax2.spines['left'].set_color('#E2E8F0')
            ax2.spines['bottom'].set_color('#E2E8F0')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        st.write("")
        out_cols = [c for c in ['rank','candidate_id','name','job_title','experience_years','skill_match_score','activity_score','final_score','reasoning'] if c in df_ranked.columns]
        csv_out = df_ranked[out_cols].round(2).to_csv(index=False).encode('utf-8')

        sub_cols = [c for c in ['rank','candidate_id','final_score','reasoning'] if c in df_ranked.columns]
        sub_out = df_ranked.head(100)[sub_cols].round(2).to_csv(index=False).encode('utf-8')

        pdf_report = create_ranked_pdf_report(
            df_ranked=df_ranked,
            top_n=top_n,
            recommendation_reason=recommendation_reason,
            explanation_keywords=keywords,
            skill_weight=skill_weight,
            exp_weight=exp_weight,
            act_weight=act_weight
        )

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "⬇️ Full Ranked CSV",
                data=csv_out,
                file_name="full_ranked_candidates.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl2:
            st.download_button(
                "⬇️ Submission CSV (Top 100)",
                data=sub_out,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl3:
            st.download_button(
                "📄 PDF Report",
                data=pdf_report,
                file_name="ranked_candidates_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.markdown("""
        <div style='text-align:center;padding:1.5rem 0 0.5rem;color:#64748B;font-size:12px'>
            🤖 Intelligent Candidate Discovery · India Runs Hackathon 2026 · Built by Irfan Shafi
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
