import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root{
    --primary:#6D28D9;
    --primary2:#8B5CF6;
    --pink:#EC4899;
    --cyan:#06B6D4;
    --ink:#0F172A;
    --muted:#64748B;
    --card:rgba(255,255,255,.82);
    --border:rgba(139,92,246,.16);
}

html, body, [class*="css"]{
    font-family:'Inter',sans-serif;
}

#MainMenu, footer, header{
    visibility:hidden;
}

.stApp{
    background:
        radial-gradient(circle at 8% 8%, rgba(139,92,246,.23), transparent 32%),
        radial-gradient(circle at 92% 2%, rgba(6,182,212,.20), transparent 30%),
        radial-gradient(circle at 55% 95%, rgba(236,72,153,.10), transparent 34%),
        linear-gradient(135deg,#F8FAFC 0%,#EEF2FF 48%,#FDF2F8 100%);
}

.block-container{
    padding:1rem 1.5rem 2rem !important;
    max-width:1580px !important;
}

.hero{
    position:relative;
    overflow:hidden;
    border-radius:24px;
    padding:1.35rem 1.75rem;
    margin:.35rem 0 1rem;
    color:white;
    background:linear-gradient(135deg,#0B1028 0%,#312E81 50%,#0891B2 100%);
    border:1px solid rgba(255,255,255,.18);
    box-shadow:0 22px 55px rgba(49,46,129,.25);
}

.hero:before{
    content:"";
    position:absolute;
    right:7%;
    top:-85px;
    width:260px;
    height:260px;
    border-radius:50%;
    background:linear-gradient(135deg,rgba(236,72,153,.62),rgba(6,182,212,.52));
    filter:blur(18px);
    opacity:.65;
}

.hero > *{
    position:relative;
    z-index:1;
}

.hero-badge{
    display:inline-flex;
    align-items:center;
    gap:7px;
    background:rgba(255,255,255,.13);
    border:1px solid rgba(255,255,255,.24);
    color:#EDE9FE;
    font-size:12px;
    font-weight:800;
    padding:6px 14px;
    border-radius:999px;
}

.hero h1{
    font-family:'Space Grotesk',sans-serif;
    font-size:clamp(1.8rem,3.1vw,3rem);
    line-height:1.05;
    font-weight:900;
    margin:.55rem 0 .32rem;
    letter-spacing:-1px;
    background:linear-gradient(90deg,#FFFFFF,#DDD6FE,#A5F3FC);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero p{
    color:#E0E7FF;
    max-width:980px;
    font-size:.98rem;
    line-height:1.45;
    margin:0;
}

.glass-card{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:24px;
    padding:1.1rem;
    box-shadow:0 18px 48px rgba(79,70,229,.11);
    backdrop-filter:blur(16px);
    margin-bottom:1rem;
}

.control-title{
    font-family:'Space Grotesk',sans-serif;
    font-weight:900;
    font-size:1rem;
    color:var(--ink);
    margin:0 0 .6rem;
}

.small-title{
    font-family:'Space Grotesk',sans-serif;
    font-weight:900;
    font-size:.92rem;
    color:var(--ink);
    margin:.15rem 0 .45rem;
}

[data-testid="stTextArea"] textarea{
    min-height:138px !important;
    border-radius:18px !important;
    border:1px solid rgba(139,92,246,.18) !important;
    background:rgba(255,255,255,.88) !important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.6), 0 10px 22px rgba(15,23,42,.04) !important;
    color:#111827 !important;
    font-size:.88rem !important;
}

.tag{
    display:inline-flex;
    align-items:center;
    background:linear-gradient(135deg,rgba(109,40,217,.13),rgba(6,182,212,.13));
    color:#4C1D95;
    font-size:11px;
    font-weight:800;
    padding:6px 10px;
    border-radius:999px;
    margin:3px;
    border:1px solid rgba(139,92,246,.18);
    box-shadow:0 8px 18px rgba(109,40,217,.07);
}

.weight-mini{
    background:rgba(255,255,255,.65);
    border:1px solid rgba(139,92,246,.12);
    border-radius:18px;
    padding:.7rem .8rem;
    margin-top:.35rem;
}

.stSlider label, .stNumberInput label, .stFileUploader label{
    font-weight:700 !important;
    color:#334155 !important;
}

.stButton > button{
    background:linear-gradient(135deg,#6D28D9 0%,#EC4899 55%,#06B6D4 100%) !important;
    color:white !important;
    border:0 !important;
    border-radius:18px !important;
    font-weight:900 !important;
    font-size:1rem !important;
    padding:.85rem 1.5rem !important;
    box-shadow:0 18px 40px rgba(109,40,217,.30) !important;
    transition:all .2s ease !important;
}

.stButton > button:hover{
    transform:translateY(-2px);
    box-shadow:0 24px 55px rgba(236,72,153,.32) !important;
}

.stDownloadButton > button{
    background:rgba(255,255,255,.92) !important;
    color:#111827 !important;
    border:1px solid rgba(139,92,246,.22) !important;
    border-radius:16px !important;
    font-weight:800 !important;
}

[data-testid="stFileUploader"] section{
    border-radius:20px !important;
    border:1.5px dashed rgba(139,92,246,.35) !important;
    background:rgba(255,255,255,.72) !important;
    box-shadow:0 14px 35px rgba(79,70,229,.08);
}

.section-title{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.18rem;
    font-weight:900;
    color:#111827;
    margin:.2rem 0 .75rem;
}

.metric-row{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:1rem;
    margin:1.1rem 0;
}

.metric-card{
    position:relative;
    overflow:hidden;
    background:rgba(255,255,255,.90);
    border:1px solid rgba(139,92,246,.16);
    border-radius:22px;
    padding:1.2rem 1rem;
    text-align:center;
    box-shadow:0 18px 42px rgba(79,70,229,.12);
}

.metric-card:before{
    content:"";
    position:absolute;
    inset:0 0 auto 0;
    height:5px;
    background:linear-gradient(90deg,var(--primary),var(--pink),var(--cyan));
}

.metric-card .label{
    font-size:11px;
    font-weight:900;
    color:#6D28D9;
    text-transform:uppercase;
    letter-spacing:1px;
    margin-bottom:8px;
}

.metric-card .value{
    font-family:'Space Grotesk',sans-serif;
    font-size:2rem;
    font-weight:900;
    color:#111827;
    line-height:1;
}

.metric-card .sub{
    font-size:12px;
    color:#64748B;
    margin-top:6px;
}

.candidate-card{
    background:rgba(255,255,255,.94);
    border:1px solid rgba(226,232,240,.95);
    border-radius:20px;
    padding:1rem 1.15rem;
    margin-bottom:12px;
    display:flex;
    align-items:center;
    gap:1.15rem;
    box-shadow:0 14px 35px rgba(15,23,42,.08);
    transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}

.candidate-card:hover{
    transform:translateY(-3px);
    box-shadow:0 22px 50px rgba(79,70,229,.18);
    border-color:rgba(139,92,246,.35);
}

.rank-badge{
    width:44px;
    height:44px;
    border-radius:15px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:900;
    font-size:15px;
    flex-shrink:0;
}

.rank-1{ background:linear-gradient(135deg,#FDE68A,#F59E0B); color:#fff; }
.rank-2{ background:linear-gradient(135deg,#E5E7EB,#6B7280); color:#fff; }
.rank-3{ background:linear-gradient(135deg,#FDBA74,#B45309); color:#fff; }
.rank-other{ background:linear-gradient(135deg,#EEF2FF,#DDD6FE); color:#5B21B6; }
.score-bar-bg{ background:#EEF2FF; border-radius:999px; height:10px; width:100%; overflow:hidden; }
.score-bar-fill{ height:100%; border-radius:999px; background:linear-gradient(90deg,var(--primary),var(--pink),var(--cyan)); }
hr{ display:none !important; }

@media(max-width:1100px){
    .metric-row{ grid-template-columns:repeat(2,1fr); }
}

@media(max-width:700px){
    .metric-row{ grid-template-columns:1fr; }
    .candidate-card{ align-items:flex-start; flex-direction:column; }
    .hero{ padding:1.25rem; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">🇮🇳 India Runs Hackathon 2026 · Track 01 · Data & AI</div>
    <h1>Intelligent Candidate Discovery</h1>
    <p>AI-powered recruiter that shortlists the best candidates for any job role in seconds using NLP & machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ── Horizontal Control Section ────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">⚙️ Recruiter Control Panel</div>', unsafe_allow_html=True)

jd_col, key_col, weight_col, select_col = st.columns([2.15, 1.55, 2.05, 1.05], gap="large")

with jd_col:
    st.markdown('<div class="small-title">📋 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste or type any job description:",
        value="""We are looking for an AI/ML Engineer with strong Python skills.
Required: machine learning, deep learning, TensorFlow, scikit-learn,
natural language processing, data analysis, model deployment.
Experience with neural networks, transformers, and LLM is preferred.
Minimum 3 years experience. Strong problem solving skills required.""",
        height=138,
        label_visibility="collapsed"
    )

with key_col:
    st.markdown('<div class="small-title">🔍 Keywords</div>', unsafe_allow_html=True)
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
    skill_weight = st.slider("🎯 Skill Match", 0, 100, 70, step=5)
    exp_weight   = st.slider("📅 Experience",  0, 100, 20, step=5)
    act_weight   = st.slider("⚡ Activity",    0, 100, 10, step=5)
    total = skill_weight + exp_weight + act_weight
    if total != 100:
        st.error(f"⚠️ Weights = {total}% · Must be 100%")
    else:
        st.markdown(f"""
        <div class='weight-mini'>
            <div style='display:flex;gap:4px;height:9px;border-radius:8px;overflow:hidden'>
                <div style='width:{skill_weight}%;background:#6D28D9'></div>
                <div style='width:{exp_weight}%;background:#06B6D4'></div>
                <div style='width:{act_weight}%;background:#10B981'></div>
            </div>
            <div style='display:flex;gap:10px;margin-top:7px;font-size:11px;color:#475569;flex-wrap:wrap;font-weight:800'>
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

# ── Upload + Run Section ──────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
upload_col, run_col = st.columns([3, 1.15], gap="large")

with upload_col:
    st.markdown('<div class="small-title">📂 Candidate Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload candidates CSV files (select multiple)",
        type=["csv"],
        accept_multiple_files=True
    )

with run_col:
    st.markdown('<div class="small-title">🚀 Action</div>', unsafe_allow_html=True)
    st.write("")
    run = st.button("Run AI Ranking", use_container_width=True)

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

if uploaded:
    dfs = [pd.read_csv(f) for f in uploaded]
    df_upload = pd.concat(dfs, ignore_index=True)
    if len(df_upload) < 5:
        st.warning(f"⚠️ Only {len(df_upload)} candidates found. Using default dataset.")
        df = get_default_data()
    else:
        df = df_upload
        st.success(f"✅ {len(df)} candidates loaded from {len(uploaded)} file(s)!")
else:
    df = get_default_data()

if run:
    if total != 100:
        st.error("❌ Weights must add up to 100%")
    elif not job_description.strip():
        st.error("❌ Please enter a job description")
    else:
        with st.spinner("🤖 Analyzing candidates..."):
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

        st.success("✅ Ranking complete!")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        top1 = df_ranked.iloc[0]
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="label">👑 Top Candidate</div>
                <div class="value" style="font-size:1.1rem">{top1['name']}</div>
                <div class="sub">{top1.get('job_title','')}</div>
            </div>
            <div class="metric-card">
                <div class="label">🎯 Top Score</div>
                <div class="value">{df_ranked['final_score'].max():.1f}</div>
                <div class="sub">out of 100</div>
            </div>
            <div class="metric-card">
                <div class="label">📊 Total</div>
                <div class="value">{len(df2)}</div>
                <div class="sub">candidates analyzed</div>
            </div>
            <div class="metric-card">
                <div class="label">📈 Avg Score</div>
                <div class="value">{df_ranked['final_score'].mean():.1f}</div>
                <div class="sub">across all</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="section-title">🏆 Top {top_n} Candidates</div>', unsafe_allow_html=True)

        for i, row in df_ranked.head(top_n).iterrows():
            rank = int(row['rank'])
            badge_cls = "rank-1" if rank==1 else "rank-2" if rank==2 else "rank-3" if rank==3 else "rank-other"
            rank_icon = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else str(rank)
            score = row['final_score']
            skill = row['skill_match_score']
            st.markdown(f"""
            <div class="candidate-card">
                <div class="rank-badge {badge_cls}">{rank_icon}</div>
                <div style="flex:1">
                    <div style="font-weight:600;color:#1e1b4b;font-size:15px">{row['name']}</div>
                    <div style="color:#6b7280;font-size:12px;margin-top:2px">
                        {row.get('job_title','')} · {row.get('experience_years','')} yrs · {row.get('education','')}
                    </div>
                </div>
                <div style="min-width:140px">
                    <div style="font-size:11px;color:#7c3aed;margin-bottom:4px">Skill Match: {skill:.1f}%</div>
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{min(int(score),100)}%"></div></div>
                </div>
                <div style="text-align:right;min-width:70px">
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#7c3aed">{score:.1f}</div>
                    <div style="font-size:10px;color:#9ca3af">Final Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_facecolor('#faf9ff')
            ax1.set_facecolor('#faf9ff')
            top10 = df_ranked.head(10)
            colors = ['#7c3aed' if i==0 else '#a78bfa' if i<3 else '#c4b5fd' for i in range(len(top10))]
            bars = ax1.barh(top10['name'], top10['final_score'], color=colors, height=0.6)
            ax1.set_xlabel('Final Score', color='#6b7280', fontsize=10)
            ax1.set_title('Top 10 Candidates', color='#1e1b4b', fontsize=12, fontweight='bold')
            ax1.invert_yaxis()
            ax1.set_xlim(0, 100)
            ax1.tick_params(colors='#6b7280', labelsize=9)
            for sp in ['top','right']: ax1.spines[sp].set_visible(False)
            for bar, val in zip(bars, top10['final_score']):
                ax1.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f'{val:.1f}', va='center', fontsize=8, color='#7c3aed')
            plt.tight_layout()
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#faf9ff')
            ax2.set_facecolor('#faf9ff')
            sc = ax2.scatter(df2['skill_match_score'], df2['experience_years'], c=df2['final_score'], cmap='RdPu', s=80, alpha=0.85, edgecolors='white', linewidths=0.5)
            ax2.set_xlabel('Skill Match Score', color='#6b7280', fontsize=10)
            ax2.set_ylabel('Experience (years)', color='#6b7280', fontsize=10)
            ax2.set_title('Skill vs Experience', color='#1e1b4b', fontsize=12, fontweight='bold')
            ax2.tick_params(colors='#6b7280', labelsize=9)
            for sp in ['top','right']: ax2.spines[sp].set_visible(False)
            plt.colorbar(sc, ax=ax2, label='Final Score')
            plt.tight_layout()
            st.pyplot(fig2)

        st.write("")
        out_cols = [c for c in ['rank','name','job_title','experience_years','skill_match_score','activity_score','final_score'] if c in df_ranked.columns]
        csv = df_ranked[out_cols].round(2).to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Ranked Results CSV", data=csv, file_name="ranked_candidates.csv", mime="text/csv", use_container_width=True)

        st.markdown("""
        <div style='text-align:center;padding:1.5rem 0 0.5rem;color:#9ca3af;font-size:12px'>
            🤖 Intelligent Candidate Discovery · India Runs Hackathon 2026 · Built by Irfan Shafi
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
