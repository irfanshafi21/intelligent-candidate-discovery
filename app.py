import warnings
warnings.filterwarnings("ignore")

import html
import json
import re
from heapq import nlargest
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import orjson  # optional: much faster if installed
except Exception:
    orjson = None

UPLOAD_LIMIT_MB = 1024
TOP_KEEP_FOR_UI = 5000          # keep only best 5000 rows in memory for UI/download speed
UI_UPDATE_EVERY = 25000         # updating Streamlit too often makes processing slow

st.set_page_config(
    page_title="AI Recruiter | Fast Ranking",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700;800&display=swap');
:root{--bg:#F8FAFC;--primary:#0F172A;--accent:#06B6D4;--card:#FFFFFF;--text:#111827;--muted:#64748B;--border:#E2E8F0;}
html, body, [class*="css"]{font-family:'Inter',sans-serif;} #MainMenu, footer, header{visibility:hidden;}
.stApp{background:var(--bg);} .block-container{padding:.75rem 1.15rem 1.3rem !important;max-width:1480px !important;}
.hero{border-radius:20px;padding:1.05rem 1.35rem;margin:.15rem 0 .7rem;color:white;background:var(--primary);box-shadow:0 18px 42px rgba(15,23,42,.14);}
.hero h1{font-family:'Space Grotesk',sans-serif;font-size:clamp(1.65rem,2.7vw,2.7rem);font-weight:900;margin:.45rem 0 .2rem;}
.hero p{color:#CBD5E1;font-size:.93rem;line-height:1.4;margin:0;}
.hero-badge{display:inline-flex;background:rgba(6,182,212,.16);border:1px solid rgba(6,182,212,.32);color:#CFFAFE;font-size:11.5px;font-weight:800;padding:5px 13px;border-radius:999px;}
.panel-card,.upload-card,.run-card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:.82rem .95rem;box-shadow:0 12px 30px rgba(15,23,42,.06);margin-bottom:.65rem;}
.section-title,.small-title{font-family:'Space Grotesk',sans-serif;font-weight:900;color:var(--primary);}.section-title{font-size:1.03rem;margin:.05rem 0 .52rem}.small-title{font-size:.88rem;margin:0 0 .35rem;}
.tag{display:inline-flex;background:#E0F7FB;color:#0E7490;font-size:10.3px;font-weight:800;padding:5px 9px;border-radius:999px;margin:3px;border:1px solid #BAE6FD;}
.stButton > button{background:var(--primary) !important;color:#fff !important;border-radius:18px !important;font-weight:900 !important;min-height:60px !important;}
.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.85rem;margin:.75rem 0 1rem}.metric-card{background:#fff;border:1px solid var(--border);border-top:4px solid var(--accent);border-radius:18px;padding:1rem .85rem;text-align:center}.metric-card .label{font-size:10.5px;font-weight:900;color:#0E7490;text-transform:uppercase}.metric-card .value{font-family:'Space Grotesk',sans-serif;font-size:1.65rem;font-weight:900;color:var(--primary)}
.candidate-card{background:#fff;border:1px solid var(--border);border-radius:17px;padding:.78rem .95rem;margin-bottom:9px;display:flex;align-items:center;gap:1rem;box-shadow:0 10px 24px rgba(15,23,42,.055)}
.rank-badge{width:40px;height:40px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-weight:900;flex-shrink:0}.rank-1{background:#F59E0B;color:#fff}.rank-2{background:#64748B;color:#fff}.rank-3{background:#B45309;color:#fff}.rank-other{background:#E0F7FB;color:#0E7490}.score-bar-bg{background:#E2E8F0;border-radius:999px;height:8px;width:100%;overflow:hidden}.score-bar-fill{height:100%;border-radius:999px;background:var(--accent)}
.why-chip{background:#E0F7FB;color:#0E7490;border:1px solid #BAE6FD;border-radius:999px;padding:3px 7px;font-size:10px;font-weight:850;margin-right:4px}.why-pill{background:#F1F5F9;color:#334155;border:1px solid #E2E8F0;border-radius:999px;padding:3px 7px;font-size:10px;font-weight:800;margin-right:4px}
@media(max-width:760px){.metric-row{grid-template-columns:1fr}.candidate-card{align-items:flex-start;flex-direction:column}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-badge">🇮🇳 India Runs Hackathon 2026 · Fast JSONL Ranking</div>
  <h1>Intelligent Candidate Discovery</h1>
  <p>Optimized for huge JSONL candidate files. Uses streaming parsing, fast keyword scoring, and low-frequency UI updates.</p>
</div>
""", unsafe_allow_html=True)

ML_SKILLS = {
    'python','machine learning','deep learning','tensorflow','pytorch','scikit-learn','nlp','natural language processing','llm','transformer','bert','gpt','fine-tuning llms','lora','neural networks','mlops','model deployment','feature engineering','data analysis','sql','vector databases','milvus','faiss','pinecone','aws','gcp','azure','data pipelines','speech recognition','image classification','computer vision','weights & biases','wandb','bentoml','gans','statistical modeling','spark','airflow','kafka','rlhf','hugging face','xgboost','lightgbm','random forest','mlflow','docker','kubernetes','reinforcement learning','time series','anomaly detection','recommendation systems','graph neural networks','object detection','yolo','opencv','pyspark'
}
HONEYPOT_TITLES = {'marketing manager','operations manager','hr manager','content writer','business analyst','accountant','mechanical engineer','logistics manager','sales manager','finance manager','supply chain','procurement manager','administrative','customer service','receptionist','retail manager','project coordinator','office manager','legal counsel','civil engineer'}
proficiency_map = {'advanced':3,'intermediate':2,'beginner':1}
tier_map = {'tier_1':100,'tier_2':70,'tier_3':40}
degree_map = {'ph.d':100,'phd':100,'m.tech':85,'m.sc':80,'mca':75,'b.tech':60,'b.e.':60,'b.sc':55,'m.e.':85,'m.s.':82}
ai_fields = ('artificial intelligence','machine learning','computer science','data science','deep learning','statistics')

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.#-]{1,}")
STOP_WORDS = {'and','or','the','a','an','is','are','we','for','with','in','of','to','be','experience','skills','required','preferred','minimum','years','strong','looking','problem','solving','candidate','engineer','developer','role','work','good','must','should'}

def safe_get_profile(c):
    p = c.get('profile', {}) if isinstance(c, dict) else {}
    return p if isinstance(p, dict) else {}

def safe_list(value):
    return value if isinstance(value, list) else []

def parse_json_fast(raw_line):
    if orjson is not None:
        return orjson.loads(raw_line)
    if isinstance(raw_line, bytes):
        return json.loads(raw_line.decode('utf-8', errors='ignore'))
    return json.loads(raw_line)

def extract_job_terms(text):
    words = [w.lower() for w in WORD_RE.findall(text or '')]
    terms = []
    for w in words:
        if len(w) > 2 and w not in STOP_WORDS and w not in terms:
            terms.append(w)
    # include important multi-word ML terms when present in JD
    text_l = (text or '').lower()
    for term in ML_SKILLS:
        if ' ' in term and term in text_l and term not in terms:
            terms.append(term)
    return terms[:80]

def is_honeypot_fast(c, skill_text_l=None, career_text_l=None):
    p = safe_get_profile(c)
    title = str(p.get('current_title','')).lower()
    if skill_text_l is None:
        skills = safe_list(c.get('skills', []))
        skill_text_l = ' '.join(str(s.get('name','')).lower() for s in skills if isinstance(s, dict))
    if career_text_l is None:
        career = safe_list(c.get('career_history', []))
        career_text_l = ' '.join((str(j.get('description',''))+' '+str(j.get('title',''))).lower() for j in career if isinstance(j, dict))
    is_non_ml = any(t in title for t in HONEYPOT_TITLES)
    ml_skill_count = sum(1 for s in ML_SKILLS if s in skill_text_l)
    career_ml = sum(1 for kw in ('machine learning','python','ai ','nlp','deep learning','data science','neural') if kw in career_text_l)
    return (is_non_ml and ml_skill_count < 3 and career_ml < 2) or (ml_skill_count > 12 and career_ml < 1)

def score_candidate_fast(c, job_terms, weights):
    p = safe_get_profile(c)
    sig = c.get('redrob_signals', {}) if isinstance(c.get('redrob_signals', {}), dict) else {}
    skills = safe_list(c.get('skills', []))
    career = safe_list(c.get('career_history', []))
    edu = safe_list(c.get('education', []))

    skill_names = [str(s.get('name','')) for s in skills if isinstance(s, dict)]
    skill_text = ' '.join(skill_names + [str(p.get('headline','')), str(p.get('summary',''))])
    skill_text_l = skill_text.lower()
    career_text_l = ' '.join((str(j.get('description',''))+' '+str(j.get('title',''))).lower() for j in career if isinstance(j, dict))

    if is_honeypot_fast(c, skill_text_l, career_text_l):
        return None

    matched_terms = [kw for kw in job_terms if kw and kw in skill_text_l]
    ss = min((len(matched_terms) / max(len(job_terms[:30]), 1)) * 100, 100)

    ml_cnt = 0
    ml_prof = 0.0
    for s in skills:
        if isinstance(s, dict) and str(s.get('name','')).lower() in ML_SKILLS:
            ml_cnt += 1
            ml_prof += proficiency_map.get(str(s.get('proficiency','beginner')).lower(), 1)
            try:
                ml_prof += min(float(s.get('endorsements', 0)) / 20, 2)
            except Exception:
                pass
    ml_depth = min(ml_cnt * 5 + ml_prof * 2, 100)

    ml_months = 0
    for j in career:
        if isinstance(j, dict) and any(kw in (str(j.get('description',''))+' '+str(j.get('title',''))).lower() for kw in ('machine learning','ml ','ai ','deep learning','nlp','data science','python','neural')):
            try:
                ml_months += float(j.get('duration_months', 0) or 0)
            except Exception:
                pass
    ml_exp = min(ml_months / 36 * 100, 100)
    exp_s = min(float(p.get('years_of_experience', 0) or 0) / 10 * 100, 100)

    gh = sig.get('github_activity_score', -1)
    try:
        gh = float(gh)
    except Exception:
        gh = -1
    act = (
        min(float(sig.get('profile_completeness_score', 0) or 0), 100) * 0.15 +
        (100 if sig.get('open_to_work_flag') else 30) * 0.10 +
        (min(gh, 100) if gh > 0 else 0) * 0.20 +
        min(float(sig.get('profile_views_received_30d', 0) or 0) / 50 * 100, 100) * 0.10 +
        float(sig.get('interview_completion_rate', 0) or 0) * 100 * 0.15 +
        float(sig.get('recruiter_response_rate', 0) or 0) * 100 * 0.10 +
        (100 if sig.get('verified_email') else 0) * 0.10 +
        min(float(sig.get('saved_by_recruiters_30d', 0) or 0) / 10 * 100, 100) * 0.10
    )

    edu_scores = []
    for e in edu:
        if isinstance(e, dict):
            degree = str(e.get('degree','')).lower()
            field = str(e.get('field_of_study','')).lower()
            edu_scores.append(tier_map.get(e.get('tier','tier_3'), 40) * 0.5 + degree_map.get(degree, 50) * 0.3 + (20 if any(f in field for f in ai_fields) else 0))
    edu_s = max(edu_scores, default=0)

    asm = sig.get('skill_assessment_scores', {})
    assess = float(np.mean(list(asm.values()))) if isinstance(asm, dict) and asm else 0

    # Keep original detailed score, then blend with UI weights.
    base_final = ss*0.25 + ml_depth*0.25 + ml_exp*0.15 + exp_s*0.10 + act*0.15 + edu_s*0.05 + assess*0.05
    final = base_final*(weights['skill']/100) + exp_s*(weights['exp']/100) + act*(weights['act']/100)

    top_skills = [s for s in skill_names if s.lower() in ML_SKILLS][:4]
    skill_str = ', '.join(top_skills) if top_skills else 'technical skills'
    strength = 'Strong' if final >= 70 else 'Good' if final >= 50 else 'Moderate'
    return {
        'candidate_id': c.get('candidate_id',''),
        'name': p.get('anonymized_name',''),
        'job_title': p.get('current_title',''),
        'experience_years': round(float(p.get('years_of_experience', 0) or 0), 2),
        'skills': skill_text[:1200],
        'education': edu[0].get('degree','') if edu and isinstance(edu[0], dict) else '',
        'skill_match_score': round(ss, 2),
        'ml_depth': round(ml_depth, 2),
        'activity_score': round(act, 2),
        'final_score': round(final, 2),
        'reasoning': f"{strength} ML/AI candidate with {p.get('years_of_experience',0)} yrs as {p.get('current_title','')}; key skills: {skill_str}; matched {len(matched_terms)} JD terms."
    }

def iter_jsonl_candidates(uploaded_file):
    uploaded_file.seek(0)
    bad = []
    for line_no, raw_line in enumerate(uploaded_file, start=1):
        if not raw_line or raw_line in (b'\n', '\n'):
            continue
        try:
            obj = parse_json_fast(raw_line)
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item
        except Exception as e:
            if len(bad) < 10:
                bad.append((line_no, str(e)))
    st.session_state['json_bad_lines'] = bad

def load_json_candidates(uploaded_file):
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    data = parse_json_fast(raw)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ('candidates','data','records','items'):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
        return [data]
    return []

def pretty_keyword(term):
    t = str(term).strip()
    return t.upper() if t.lower() in {'ai','ml','nlp','llm','gpt','bert','sql','aws','gcp','api','mlops','xgboost','opencv','faiss'} else t.title()

def create_ranked_pdf_report(df_ranked, top_n, recommendation_reason):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
        top1 = df_ranked.iloc[0]
        fig.text(0.07, 0.94, 'Intelligent Candidate Discovery', fontsize=19, fontweight='bold', color='#0F172A')
        fig.text(0.07, 0.915, 'Fast ranked candidate report', fontsize=9.5, color='#64748B')
        fig.text(0.07, 0.865, f"Recommended Hire: {top1.get('name','')}", fontsize=12, fontweight='bold', color='#0E7490')
        fig.text(0.07, 0.835, recommendation_reason, fontsize=8.8, color='#334155', wrap=True)
        table_cols = ['Rank','Name','Job Title','Exp','Skill %','Activity','Final']
        table_rows = []
        for _, row in df_ranked.head(top_n).iterrows():
            table_rows.append([int(row.rank), str(row.name)[:22], str(row.job_title)[:26], row.experience_years, row.skill_match_score, round(row.activity_score), row.final_score])
        table_ax = fig.add_axes([0.06, 0.25, 0.88, 0.52]); table_ax.axis('off')
        table = table_ax.table(cellText=table_rows, colLabels=table_cols, loc='upper center', cellLoc='left', colLoc='left')
        table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1, 1.35)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()

# Controls
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">⚙️ Recruiter Control Panel</div>', unsafe_allow_html=True)
jd_col, key_col, weight_col, select_col = st.columns([2.05, 1.35, 2.35, .9], gap='medium')
default_job_description = """We are looking for a Senior AI/ML Engineer with strong Python skills.
Required: machine learning, deep learning, TensorFlow, PyTorch, scikit-learn,
natural language processing, LLMs, transformers, BERT, GPT, model deployment, MLOps.
Fine-tuning LLMs, LoRA, vector databases preferred.
Minimum 3 years ML/AI experience. Strong problem solving skills required."""
with jd_col:
    st.markdown('<div class="small-title">📋 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area('Paste or type any job description:', value=default_job_description, height=145, label_visibility='collapsed')
job_terms = extract_job_terms(job_description)
with key_col:
    st.markdown('<div class="small-title">🔍 Keywords</div>', unsafe_allow_html=True)
    st.markdown(''.join(f'<span class="tag">{html.escape(k)}</span>' for k in job_terms[:12]), unsafe_allow_html=True)
with weight_col:
    st.markdown('<div class="small-title">⚖️ Scoring Weights</div>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3, gap='small')
    with w1: skill_weight = st.slider('🎯 Skill', 0, 100, 70, step=5)
    with w2: exp_weight = st.slider('📅 Exp', 0, 100, 20, step=5)
    with w3: act_weight = st.slider('⚡ Act', 0, 100, 10, step=5)
    total = skill_weight + exp_weight + act_weight
    if total != 100: st.error(f'⚠️ Weights = {total}% · Must be 100%')
with select_col:
    st.markdown('<div class="small-title">👥 Top N</div>', unsafe_allow_html=True)
    top_n = st.slider('Number of Candidates', 3, 20, 10, label_visibility='collapsed')
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="small-title">📂 Upload Candidate Data</div>', unsafe_allow_html=True)
st.caption(f'Accepts .jsonl · .json · .csv. For 450 MB+ files, JSONL is fastest. Set server.maxUploadSize={UPLOAD_LIMIT_MB} in .streamlit/config.toml.')
uploaded_files = st.file_uploader('Upload candidate files', type=['jsonl','json','csv'], accept_multiple_files=True, label_visibility='collapsed')
json_file = next((f for f in (uploaded_files or []) if f.name.lower().endswith(('.jsonl','.json'))), None)
csv_files = [f for f in (uploaded_files or []) if f.name.lower().endswith('.csv')]
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="run-card">', unsafe_allow_html=True)
run = st.button('🚀 RUN AI RANKING', use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if run:
    if total != 100:
        st.error('❌ Weights must add up to 100%')
        st.stop()
    if not job_description.strip():
        st.error('❌ Please enter a job description')
        st.stop()

    weights = {'skill': skill_weight, 'exp': exp_weight, 'act': act_weight}
    results, loaded_count, honeypots, errors = [], 0, 0, []
    progress_bar = st.progress(0)
    status_box = st.empty()

    with st.spinner('⚡ Fast analyzing candidates...'):
        if json_file:
            mode = 'jsonl' if json_file.name.lower().endswith('.jsonl') else 'json'
            file_size = getattr(json_file, 'size', 0) or 0
            candidate_iter = iter_jsonl_candidates(json_file) if mode == 'jsonl' else iter(load_json_candidates(json_file))

            for c in candidate_iter:
                loaded_count += 1
                try:
                    row = score_candidate_fast(c, job_terms, weights)
                    if row is None:
                        honeypots += 1
                    else:
                        results.append(row)
                except Exception as e:
                    if len(errors) < 10:
                        errors.append((loaded_count, str(e)))

                if loaded_count % UI_UPDATE_EVERY == 0:
                    # Keep only top rows to avoid memory explosion on very large files.
                    if len(results) > TOP_KEEP_FOR_UI * 2:
                        results = nlargest(TOP_KEEP_FOR_UI, results, key=lambda r: r['final_score'])
                    if mode == 'jsonl' and file_size:
                        progress_bar.progress(min(json_file.tell() / file_size, 1.0))
                    status_box.info(f'📂 Processed {loaded_count:,} candidates... kept best {len(results):,}')

            progress_bar.progress(1.0)
            if len(results) > TOP_KEEP_FOR_UI:
                results = nlargest(TOP_KEEP_FOR_UI, results, key=lambda r: r['final_score'])

            bad = st.session_state.get('json_bad_lines', [])
            if bad: st.warning(f'⚠️ Skipped invalid JSONL lines. First errors: {bad[:3]}')
            if errors: st.warning(f'⚠️ Skipped invalid candidate records. First errors: {errors[:3]}')
            if not results:
                st.error('❌ No valid candidates found in uploaded file.')
                st.stop()
            df_ranked = pd.DataFrame(results).sort_values('final_score', ascending=False).reset_index(drop=True)
        elif csv_files:
            mode = 'csv'
            df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            skills_l = df['skills'].fillna('').astype(str).str.lower()
            df['skill_match_score'] = [min(sum(kw in text for kw in job_terms) / max(len(job_terms[:30]), 1) * 100, 100) for text in skills_l]
            max_exp = max(float(df['experience_years'].max() or 1), 1)
            df['exp_score'] = df['experience_years'].fillna(0) / max_exp * 100
            df['final_score'] = df['skill_match_score']*(skill_weight/100) + df['exp_score']*(exp_weight/100) + df['activity_score'].fillna(0)*(act_weight/100)
            if 'name' not in df: df['name'] = df.index.astype(str)
            if 'job_title' not in df: df['job_title'] = ''
            if 'education' not in df: df['education'] = ''
            if 'candidate_id' not in df: df['candidate_id'] = df.index + 1
            if 'reasoning' not in df: df['reasoning'] = 'Ranked from CSV skills, experience and activity score.'
            df_ranked = df.sort_values('final_score', ascending=False).head(TOP_KEEP_FOR_UI).reset_index(drop=True)
            loaded_count = len(df)
        else:
            st.error('❌ Please upload your JSONL/JSON/CSV candidate file.')
            st.stop()

    df_ranked['rank'] = df_ranked.index + 1
    st.success(f'✅ Ranking complete! Processed {loaded_count:,} candidates, showing best {len(df_ranked):,}' + (f', filtered {honeypots:,} honeypots.' if json_file else '.'))

    top1 = df_ranked.iloc[0]
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="label">👑 Top Candidate</div><div class="value" style="font-size:1.1rem">{html.escape(str(top1.get('name','')))}</div></div>
      <div class="metric-card"><div class="label">🎯 Top Score</div><div class="value">{df_ranked['final_score'].max():.1f}</div></div>
      <div class="metric-card"><div class="label">📂 Processed</div><div class="value">{loaded_count:,}</div></div>
      <div class="metric-card"><div class="label">🚫 Filtered</div><div class="value">{honeypots:,}</div></div>
    </div>
    """, unsafe_allow_html=True)

    recommendation_reason = f"Highest final score ({top1['final_score']:.1f}/100), skill match {top1.get('skill_match_score',0):.1f}%, {top1.get('experience_years','')} years experience."
    st.info('🏆 Recommended Hire: ' + str(top1.get('name','')) + ' — ' + recommendation_reason)

    st.markdown(f'<div class="section-title">🏆 Top {top_n} Candidates</div>', unsafe_allow_html=True)
    for _, row in df_ranked.head(top_n).iterrows():
        rank = int(row['rank'])
        badge_cls = 'rank-1' if rank == 1 else 'rank-2' if rank == 2 else 'rank-3' if rank == 3 else 'rank-other'
        rank_icon = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else str(rank)
        score = float(row.get('final_score', 0))
        st.markdown(f"""
        <div class="candidate-card">
          <div class="rank-badge {badge_cls}">{rank_icon}</div>
          <div style="flex:1"><div style="font-weight:700;color:#0F172A;font-size:15px">{html.escape(str(row.get('name','')))}</div>
          <div style="color:#64748B;font-size:12px;margin-top:2px">{html.escape(str(row.get('job_title','')))} · {html.escape(str(row.get('experience_years','')))} yrs · {html.escape(str(row.get('education','')))}</div>
          <div style="margin-top:6px"><span class="why-pill">Skill {row.get('skill_match_score',0):.1f}%</span><span class="why-pill">Activity {row.get('activity_score',0):.0f}/100</span></div></div>
          <div style="min-width:140px"><div style="font-size:11px;color:#0E7490;margin-bottom:4px">Final Score</div><div class="score-bar-bg"><div class="score-bar-fill" style="width:{min(int(score),100)}%"></div></div></div>
          <div style="text-align:right;min-width:70px"><div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#0E7490">{score:.1f}</div></div>
        </div>
        """, unsafe_allow_html=True)

    out_cols = [c for c in ['rank','candidate_id','name','job_title','experience_years','skill_match_score','activity_score','final_score','reasoning'] if c in df_ranked.columns]
    csv_out = df_ranked[out_cols].round(2).to_csv(index=False).encode('utf-8')
    sub_out = df_ranked.head(100)[out_cols].round(2).to_csv(index=False).encode('utf-8')
    pdf_report = create_ranked_pdf_report(df_ranked, top_n, recommendation_reason)

    dl1, dl2, dl3 = st.columns(3)
    with dl1: st.download_button('⬇️ Ranked CSV', csv_out, 'fast_ranked_candidates.csv', 'text/csv', use_container_width=True)
    with dl2: st.download_button('⬇️ Submission CSV Top 100', sub_out, 'submission.csv', 'text/csv', use_container_width=True)
    with dl3: st.download_button('📄 PDF Report', pdf_report, 'ranked_candidates_report.pdf', 'application/pdf', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
