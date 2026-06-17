import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Recruiter | India Runs Hackathon",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 1.5rem 2rem 1.5rem !important; }
.hero {
    background: linear-gradient(135deg, #1a0533 0%, #2d1b69 50%, #0f3460 100%);
    border-radius: 16px; padding: 2rem 2.5rem;
    margin-bottom: 1.5rem; margin-top: 1rem;
}
.hero-badge {
    display: inline-block; background: rgba(139,92,246,0.25);
    border: 1px solid rgba(139,92,246,0.5); color: #c4b5fd;
    font-size: 12px; font-weight: 500; padding: 4px 14px;
    border-radius: 20px; margin-bottom: 1rem;
}
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#fff; margin:0 0 0.5rem 0; }
.hero p { color:#a5b4fc; font-size:0.95rem; margin:0; }
.control-box {
    background: linear-gradient(135deg, #1a0533 0%, #1e1b4b 100%);
    border-radius: 14px; padding: 1.2rem; height: 100%;
}
.control-box h3 { font-family:'Space Grotesk',sans-serif; color:#fff; font-size:1rem; margin:0 0 0.75rem 0; }
.tag { display:inline-block; background:rgba(139,92,246,0.3); color:#e9d5ff; font-size:11px; font-weight:500; padding:3px 10px; border-radius:20px; margin:2px; border:1px solid rgba(139,92,246,0.5); }
.metric-row { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.5rem 0; }
.metric-card { background:linear-gradient(135deg,#f8f7ff 0%,#ede9fe 100%); border:1px solid #c4b5fd; border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.metric-card .label { font-size:11px; font-weight:500; color:#7c3aed; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px; }
.metric-card .value { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:#1e1b4b; line-height:1; }
.metric-card .sub { font-size:11px; color:#6d28d9; margin-top:4px; }
.candidate-card { background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:8px; display:flex; align-items:center; gap:1.2rem; }
.rank-badge { width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:13px; flex-shrink:0; }
.rank-1 { background:linear-gradient(135deg,#fbbf24,#f59e0b); color:#fff; }
.rank-2 { background:linear-gradient(135deg,#9ca3af,#6b7280); color:#fff; }
.rank-3 { background:linear-gradient(135deg,#cd7c2e,#b45309); color:#fff; }
.rank-other { background:#f3f4f6; color:#6b7280; }
.score-bar-bg { background:#f3f4f6; border-radius:20px; height:7px; width:100%; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:20px; background:linear-gradient(90deg,#7c3aed,#a855f7); }
.stButton>button { background:linear-gradient(135deg,#7c3aed 0%,#6d28d9 100%) !important; color:white !important; border:none !important; border-radius:10px !important; font-weight:600 !important; font-size:1rem !important; padding:0.75rem 2rem !important; box-shadow:0 4px 14px rgba(124,58,237,0.4) !important; }
.stDownloadButton>button { background:#f3f4f6 !important; color:#1e1b4b !important; border:1px solid #d1d5db !important; border-radius:8px !important; font-weight:500 !important; }
.control-box .stTextArea textarea { background-color:#2d1b69 !important; color:#ffffff !important; border:1px solid rgba(139,92,246,0.5) !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🇮🇳 India Runs Hackathon 2026 · Track 01 · Data &amp; AI</div>
    <h1>🤖 Intelligent Candidate Discovery</h1>
    <p>AI-powered recruiter that understands who fits the role — not just keyword matching. Uses NLP, semantic similarity &amp; 7 scoring signals.</p>
</div>
""", unsafe_allow_html=True)

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

proficiency_map = {'advanced':3,'intermediate':2,'beginner':1}
tier_map = {'tier_1':100,'tier_2':70,'tier_3':40}
degree_map = {'ph.d':100,'phd':100,'m.tech':85,'m.sc':80,'mca':75,'b.tech':60,'b.e.':60,'b.sc':55,'m.e.':85,'m.s.':82}
ai_fields = ['artificial intelligence','machine learning','computer science','data science','deep learning','statistics']

left, right = st.columns([1, 2.5])

with left:
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    st.markdown('<h3>📋 Job Description</h3>', unsafe_allow_html=True)
    job_description = st.text_area("", value="""We are looking for a Senior AI/ML Engineer with strong Python skills.
Required: machine learning, deep learning, TensorFlow, PyTorch, scikit-learn,
NLP, LLMs, transformers, BERT, GPT, model deployment, MLOps, feature engineering.
Fine-tuning LLMs, LoRA, vector databases preferred.
Minimum 3 years ML/AI experience.""", height=170, label_visibility="collapsed")

    if job_description:
        stop_words = {'and','or','the','a','an','is','are','we','for','with','in','of','to','be','experience','skills','required','preferred','minimum','years','strong','looking'}
        words = [w.strip('.,()') for w in job_description.lower().split() if len(w)>3 and w.lower().strip('.,()') not in stop_words]
        keywords = list(dict.fromkeys(words))[:10]
        tags_html = "".join([f'<span class="tag">{k}</span>' for k in keywords])
        st.markdown("**🔍 Keywords:**")
        st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h3>⚖️ Scoring Weights</h3>', unsafe_allow_html=True)
    skill_w  = st.slider("🎯 Skill Match",    0, 100, 25, step=5)
    depth_w  = st.slider("🧠 ML Depth",       0, 100, 25, step=5)
    mlexp_w  = st.slider("📅 ML Experience",  0, 100, 15, step=5)
    exp_w    = st.slider("🏢 Total Exp",      0, 100, 10, step=5)
    act_w    = st.slider("⚡ Activity",       0, 100, 15, step=5)
    edu_w    = st.slider("🎓 Education",      0, 100,  5, step=5)
    assess_w = st.slider("📝 Assessment",     0, 100,  5, step=5)
    total = skill_w+depth_w+mlexp_w+exp_w+act_w+edu_w+assess_w
    if total != 100:
        st.error(f"⚠️ Weights = {total}% · Must be 100%")
    else:
        st.success("✅ Weights = 100%")

    st.markdown("---")
    top_n = st.slider("👥 Top N Candidates", 10, 100, 10)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("### 📂 Upload Candidate Data")
    st.markdown("Upload `candidates.jsonl` or a CSV file")
    c1, c2 = st.columns(2)
    with c1:
        jsonl_file = st.file_uploader("Upload JSONL file", type=["jsonl","json"])
    with c2:
        csv_file = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=True)

    st.markdown("---")
    run = st.button("🚀 Run AI Ranking", use_container_width=True)

    if run:
        if total != 100:
            st.error("❌ Weights must add up to 100%")
        elif not job_description.strip():
            st.error("❌ Please enter a job description")
        elif not jsonl_file and not csv_file:
            st.error("❌ Please upload a candidate file (JSONL or CSV)")
        else:
            with st.spinner("🤖 Analyzing candidates — this may take a moment for large files..."):
                # Load candidates
                candidates = []
                if jsonl_file:
                    content = jsonl_file.read().decode('utf-8')
                    for line in content.strip().split('\n'):
                        if line.strip():
                            candidates.append(json.loads(line))
                    mode = "jsonl"
                elif csv_file:
                    dfs = [pd.read_csv(f) for f in csv_file]
                    df_csv = pd.concat(dfs, ignore_index=True)
                    # Convert CSV to candidate-like format
                    for _, row in df_csv.iterrows():
                        c = {
                            'candidate_id': str(row.get('candidate_id', row.get('name','unknown'))),
                            'profile': {
                                'anonymized_name': str(row.get('name','')),
                                'current_title': str(row.get('job_title','')),
                                'years_of_experience': float(row.get('experience_years',0)),
                                'headline': str(row.get('job_title','')),
                                'summary': str(row.get('skills',''))
                            },
                            'skills': [{'name': s.strip(), 'proficiency':'intermediate','endorsements':0,'duration_months':12}
                                       for s in str(row.get('skills','')).split()],
                            'career_history': [],
                            'education': [{'degree': str(row.get('education','')), 'tier':'tier_2', 'field_of_study':'Computer Science'}],
                            'redrob_signals': {
                                'profile_completeness_score': 70,
                                'open_to_work_flag': True,
                                'github_activity_score': float(row.get('activity_score',50)),
                                'profile_views_received_30d': 20,
                                'interview_completion_rate': 0.7,
                                'recruiter_response_rate': 0.5,
                                'verified_email': True,
                                'saved_by_recruiters_30d': 3,
                                'skill_assessment_scores': {}
                            }
                        }
                        candidates.append(c)
                    mode = "csv"

                # Build vectorizer
                vectorizer = TfidfVectorizer(stop_words='english', max_features=8000)
                vectorizer.fit([job_description])
                jd_vec = vectorizer.transform([job_description])

                results = []
                honeypots = 0
                for c in candidates:
                    if is_honeypot(c):
                        honeypots += 1
                        continue
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
                    final=ss*(skill_w/100)+ml_depth*(depth_w/100)+ml_exp*(mlexp_w/100)+exp_s*(exp_w/100)+act*(act_w/100)+edu_s*(edu_w/100)+assess*(assess_w/100)
                    top_skills=[s['name'] for s in skills if s['name'].lower() in ML_SKILLS][:4]
                    skill_str=', '.join(top_skills) if top_skills else 'technical skills'
                    strength="Strong" if final>=70 else "Good" if final>=50 else "Moderate"
                    reasoning=f"{strength} ML/AI candidate with {p.get('years_of_experience',0):.1f} yrs as {p.get('current_title','')}; key skills: {skill_str}. ML depth {ml_depth:.0f}/100, assessment {assess:.0f}/100."
                    results.append({'candidate_id':c['candidate_id'],'name':p.get('anonymized_name',''),'title':p.get('current_title',''),'years_exp':p.get('years_of_experience',0),'final_score':round(final,2),'skill_match':round(ss,2),'ml_depth':round(ml_depth,2),'activity':round(act,2),'reasoning':reasoning})

                df_r = pd.DataFrame(results).sort_values('final_score',ascending=False).reset_index(drop=True)
                df_r['rank'] = df_r.index+1

            st.success(f"✅ Ranking complete! {len(df_r)} candidates scored, {honeypots} honeypots filtered.")

            top1=df_r.iloc[0]
            st.markdown(f"""<div class="metric-row">
                <div class="metric-card"><div class="label">👑 Top Candidate</div><div class="value" style="font-size:1rem">{top1['name']}</div><div class="sub">{top1['title']}</div></div>
                <div class="metric-card"><div class="label">🎯 Top Score</div><div class="value">{df_r['final_score'].max():.1f}</div><div class="sub">out of 100</div></div>
                <div class="metric-card"><div class="label">📊 Scored</div><div class="value">{len(df_r)}</div><div class="sub">valid candidates</div></div>
                <div class="metric-card"><div class="label">🚫 Filtered</div><div class="value">{honeypots}</div><div class="sub">honeypots removed</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"### 🏆 Top {top_n} Candidates")
            for _, row in df_r.head(top_n).iterrows():
                rank=int(row['rank'])
                bc="rank-1" if rank==1 else "rank-2" if rank==2 else "rank-3" if rank==3 else "rank-other"
                ri="🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else str(rank)
                score=row['final_score']; skill=row['skill_match']
                st.markdown(f"""<div class="candidate-card">
                    <div class="rank-badge {bc}">{ri}</div>
                    <div style="flex:1">
                        <div style="font-weight:600;color:#1e1b4b;font-size:14px">{row['name']}</div>
                        <div style="color:#6b7280;font-size:11px;margin-top:2px">{row['title']} · {row['years_exp']:.1f} yrs</div>
                        <div style="color:#9ca3af;font-size:11px;margin-top:2px;font-style:italic">{row['reasoning'][:100]}...</div>
                    </div>
                    <div style="min-width:130px">
                        <div style="font-size:10px;color:#7c3aed;margin-bottom:3px">Skill: {skill:.1f}%</div>
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{min(int(score),100)}%"></div></div>
                    </div>
                    <div style="text-align:right;min-width:65px">
                        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;color:#7c3aed">{score:.1f}</div>
                        <div style="font-size:10px;color:#9ca3af">Score</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            ch1, ch2 = st.columns(2)
            with ch1:
                fig1, ax1 = plt.subplots(figsize=(6,4))
                fig1.patch.set_facecolor('#faf9ff'); ax1.set_facecolor('#faf9ff')
                t10=df_r.head(10)
                clrs=['#7c3aed' if i==0 else '#a78bfa' if i<3 else '#c4b5fd' for i in range(len(t10))]
                bars=ax1.barh(t10['name'],t10['final_score'],color=clrs,height=0.6)
                ax1.set_xlabel('Final Score',color='#6b7280',fontsize=10); ax1.set_title('Top 10 Candidates',color='#1e1b4b',fontsize=12,fontweight='bold')
                ax1.invert_yaxis(); ax1.set_xlim(0,100); ax1.tick_params(colors='#6b7280',labelsize=8)
                for sp in ['top','right']: ax1.spines[sp].set_visible(False)
                for bar,val in zip(bars,t10['final_score']):
                    ax1.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,f'{val:.1f}',va='center',fontsize=7,color='#7c3aed')
                plt.tight_layout(); st.pyplot(fig1)
            with ch2:
                fig2, ax2 = plt.subplots(figsize=(6,4))
                fig2.patch.set_facecolor('#faf9ff'); ax2.set_facecolor('#faf9ff')
                sc=ax2.scatter(df_r['skill_match'],df_r['years_exp'],c=df_r['final_score'],cmap='RdPu',s=40,alpha=0.6,edgecolors='white',linewidths=0.3)
                ax2.set_xlabel('Skill Match Score',color='#6b7280',fontsize=10); ax2.set_ylabel('Experience (yrs)',color='#6b7280',fontsize=10)
                ax2.set_title('Skill vs Experience',color='#1e1b4b',fontsize=12,fontweight='bold')
                ax2.tick_params(colors='#6b7280',labelsize=9)
                for sp in ['top','right']: ax2.spines[sp].set_visible(False)
                plt.colorbar(sc,ax=ax2,label='Final Score'); plt.tight_layout(); st.pyplot(fig2)

            st.markdown("---")
            # Download full ranked
            full_csv = df_r.round(2).to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Full Ranked CSV", data=full_csv, file_name="full_ranked_candidates.csv", mime="text/csv", use_container_width=True)
            # Download submission top 100
            sub_csv = df_r.head(100)[['rank','candidate_id','final_score','reasoning']].round(2).to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Submission CSV (Top 100)", data=sub_csv, file_name="submission.csv", mime="text/csv", use_container_width=True)

            st.markdown("<div style='text-align:center;padding:1rem 0;color:#9ca3af;font-size:12px'>🤖 Intelligent Candidate Discovery · India Runs Hackathon 2026 · Built by Irfan Shafi</div>", unsafe_allow_html=True)
