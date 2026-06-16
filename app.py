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
    --bg:#070A1A;
    --panel:#0D1230;
    --panel2:#111944;
    --card:#FFFFFF;
    --muted:#9CA3AF;
    --text:#111827;
    --purple:#8B5CF6;
    --pink:#EC4899;
    --cyan:#06B6D4;
    --green:#10B981;
    --gold:#F59E0B;
}

html, body, [class*="css"] { font-family:'Inter',sans-serif; }
#MainMenu, footer, header { visibility:hidden; }
.stApp{
    background:
        radial-gradient(circle at top left, rgba(139,92,246,.30), transparent 32%),
        radial-gradient(circle at top right, rgba(6,182,212,.22), transparent 28%),
        linear-gradient(135deg,#F8FAFC 0%,#EEF2FF 45%,#FDF2F8 100%);
}
.block-container{ padding:1.2rem 1.8rem 2rem !important; max-width:1500px !important; }

/* Premium glass hero */
.hero{
    position:relative;
    overflow:hidden;
    border-radius:30px;
    padding:2.4rem 2.7rem;
    margin:0.8rem 0 1.5rem;
    color:white;
    background:
        linear-gradient(135deg, rgba(13,18,48,.98), rgba(49,18,98,.96) 48%, rgba(3,105,161,.94)),
        url('');
    box-shadow:0 30px 80px rgba(49,46,129,.32);
    border:1px solid rgba(255,255,255,.18);
}
.hero:before{
    content:"";
    position:absolute; inset:-90px -60px auto auto;
    width:280px; height:280px;
    background:linear-gradient(135deg,rgba(236,72,153,.75),rgba(6,182,212,.65));
    filter:blur(12px); opacity:.55; border-radius:50%;
}
.hero:after{
    content:"";
    position:absolute; inset:auto auto -120px -70px;
    width:260px; height:260px;
    background:linear-gradient(135deg,rgba(139,92,246,.75),rgba(16,185,129,.45));
    filter:blur(18px); opacity:.50; border-radius:50%;
}
.hero > *{ position:relative; z-index:1; }
.hero-badge{
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.24);
    color:#EDE9FE;
    font-size:12px; font-weight:800; letter-spacing:.5px;
    padding:8px 16px; border-radius:999px;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.18);
    backdrop-filter:blur(14px);
}
.hero h1{
    font-family:'Space Grotesk',sans-serif;
    font-size:clamp(2.2rem,4vw,4rem);
    line-height:1.02;
    font-weight:800;
    margin:.9rem 0 .55rem;
    letter-spacing:-1.4px;
    background:linear-gradient(90deg,#FFFFFF,#DDD6FE,#A5F3FC);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p{ color:#DDE7FF; max-width:850px; font-size:1.05rem; line-height:1.65; margin:0; }

/* Control sidebar */
.control-box{
    position:sticky; top:1rem;
    background:linear-gradient(180deg,rgba(13,18,48,.98),rgba(31,18,76,.98));
    border:1px solid rgba(255,255,255,.16);
    border-radius:26px;
    padding:1.35rem;
    height:100%;
    box-shadow:0 24px 60px rgba(15,23,42,.26);
    backdrop-filter:blur(18px);
}
.control-box h3{
    font-family:'Space Grotesk',sans-serif;
    color:#fff;
    font-size:1.05rem;
    font-weight:800;
    margin:0 0 .85rem;
}
.control-label{ color:#C4B5FD; font-size:12px; font-weight:700; margin-bottom:4px; }
.tag{
    display:inline-flex;
    align-items:center;
    background:linear-gradient(135deg,rgba(139,92,246,.25),rgba(6,182,212,.16));
    color:#F5F3FF;
    font-size:11px; font-weight:700;
    padding:6px 11px; border-radius:999px;
    margin:3px;
    border:1px solid rgba(196,181,253,.38);
    box-shadow:0 8px 18px rgba(0,0,0,.12);
}

/* Section titles */
.section-title{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.25rem;
    font-weight:800;
    color:#111827;
    margin:1.15rem 0 .8rem;
    display:flex; align-items:center; gap:10px;
}
/* Section underline removed for cleaner UI */

/* Metrics */
.metric-row{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:1rem;
    margin:1.4rem 0;
}
.metric-card{
    position:relative; overflow:hidden;
    background:rgba(255,255,255,.86);
    border:1px solid rgba(139,92,246,.16);
    border-radius:24px;
    padding:1.35rem 1.2rem;
    text-align:center;
    box-shadow:0 20px 45px rgba(79,70,229,.12);
    backdrop-filter:blur(14px);
}
.metric-card:before{
    content:"";
    position:absolute; inset:0 0 auto 0; height:5px;
    background:linear-gradient(90deg,var(--purple),var(--pink),var(--cyan));
}
.metric-card .label{
    font-size:11px; font-weight:900; color:#6D28D9;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;
}
.metric-card .value{
    font-family:'Space Grotesk',sans-serif;
    font-size:2.15rem; font-weight:800; color:#111827; line-height:1;
}
.metric-card .sub{ font-size:12px; color:#6B7280; margin-top:6px; }

/* Candidate cards */
.candidate-card{
    background:rgba(255,255,255,.92);
    border:1px solid rgba(226,232,240,.95);
    border-radius:22px;
    padding:1rem 1.15rem;
    margin-bottom:12px;
    display:flex; align-items:center; gap:1.2rem;
    box-shadow:0 14px 35px rgba(15,23,42,.08);
    transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.candidate-card:hover{
    transform:translateY(-3px);
    box-shadow:0 22px 50px rgba(79,70,229,.18);
    border-color:rgba(139,92,246,.35);
}
.rank-badge{
    width:46px; height:46px; border-radius:16px;
    display:flex; align-items:center; justify-content:center;
    font-weight:900; font-size:15px; flex-shrink:0;
    box-shadow:0 12px 25px rgba(15,23,42,.18);
}
.rank-1{ background:linear-gradient(135deg,#FDE68A,#F59E0B); color:#fff; }
.rank-2{ background:linear-gradient(135deg,#E5E7EB,#6B7280); color:#fff; }
.rank-3{ background:linear-gradient(135deg,#FDBA74,#B45309); color:#fff; }
.rank-other{ background:linear-gradient(135deg,#EEF2FF,#DDD6FE); color:#5B21B6; }
.score-bar-bg{
    background:#EEF2FF; border-radius:999px; height:10px; width:100%; overflow:hidden;
    box-shadow:inset 0 1px 3px rgba(0,0,0,.08);
}
.score-bar-fill{ height:100%; border-radius:999px; background:linear-gradient(90deg,var(--purple),var(--pink),var(--cyan)); }

/* Streamlit widgets */
.stButton > button{
    background:linear-gradient(135deg,#7C3AED 0%,#EC4899 55%,#06B6D4 100%) !important;
    color:white !important; border:0 !important;
    border-radius:18px !important; font-weight:900 !important;
    font-size:1rem !important; padding:.9rem 2rem !important;
    box-shadow:0 18px 40px rgba(124,58,237,.35) !important;
    transition:transform .15s ease, box-shadow .15s ease !important;
}
.stButton > button:hover{ transform:translateY(-2px); box-shadow:0 24px 55px rgba(236,72,153,.35) !important; }
.stDownloadButton > button{
    background:rgba(255,255,255,.9) !important; color:#111827 !important;
    border:1px solid rgba(139,92,246,.22) !important; border-radius:16px !important;
    font-weight:800 !important; box-shadow:0 12px 30px rgba(15,23,42,.08) !important;
}
.stAlert{ border-radius:18px !important; border:0 !important; box-shadow:0 12px 30px rgba(15,23,42,.08); }
hr{ border-color:rgba(148,163,184,.24) !important; margin:1.25rem 0 !important; }

.control-box .stSlider label,
.control-box .stTextArea label{ color:#C4B5FD !important; font-weight:800 !important; }
.control-box .stTextArea textarea{
    background:rgba(255,255,255,.08) !important;
    color:#fff !important;
    border:1px solid rgba(196,181,253,.30) !important;
    border-radius:18px !important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.08) !important;
}
.control-box .stTextArea textarea:focus{ border-color:#A78BFA !important; box-shadow:0 0 0 3px rgba(139,92,246,.20) !important; }

/* File uploader box */
[data-testid="stFileUploader"] section{
    border-radius:22px !important;
    border:1.5px dashed rgba(139,92,246,.35) !important;
    background:rgba(255,255,255,.72) !important;
    box-shadow:0 14px 35px rgba(79,70,229,.08);
}


/* Sidebar product panel */
section[data-testid="stSidebar"]{
    background:
        radial-gradient(circle at top left,rgba(139,92,246,.35),transparent 35%),
        linear-gradient(180deg,#0B1028,#17113D 52%,#0F172A) !important;
    border-right:1px solid rgba(255,255,255,.10);
}
section[data-testid="stSidebar"] > div{ padding:1.2rem 1rem 2rem !important; }
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span{ color:#F8FAFC !important; }
.sidebar-brand{
    display:flex; align-items:center; gap:12px;
    padding:1rem; border-radius:22px;
    background:rgba(255,255,255,.10);
    border:1px solid rgba(255,255,255,.16);
    box-shadow:0 18px 40px rgba(0,0,0,.20);
    margin-bottom:1.2rem;
}
.brand-icon{ width:44px; height:44px; border-radius:16px; display:grid; place-items:center; background:linear-gradient(135deg,#8B5CF6,#EC4899,#06B6D4); font-size:1.3rem; }
.brand-title{ font-family:'Space Grotesk',sans-serif; font-weight:900; color:#fff; font-size:1.05rem; }
.brand-sub{ color:#CBD5E1; font-size:.72rem; margin-top:2px; }
.dashboard-strip{
    display:flex; justify-content:space-between; align-items:center; gap:1rem;
    background:rgba(255,255,255,.82);
    border:1px solid rgba(139,92,246,.15);
    border-radius:22px;
    padding:1rem 1.2rem;
    box-shadow:0 18px 42px rgba(79,70,229,.10);
    margin:.2rem 0 1.2rem;
}
.dashboard-strip b{ color:#111827; }
.dashboard-strip span{ color:#6B7280; font-size:.9rem; }
[data-testid="stFileUploader"]{ margin-bottom:1rem; }

@media(max-width:900px){
    .metric-row{ grid-template-columns:repeat(2,1fr); }
    .candidate-card{ align-items:flex-start; flex-direction:column; }
    .control-box{ position:relative; }
}
@media(max-width:620px){ .metric-row{ grid-template-columns:1fr; } .hero{ padding:1.7rem; } }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🇮🇳 India Runs Hackathon 2026 · Track 01 · Data & AI</div>
    <h1>🤖 Intelligent Candidate Discovery</h1>
    <p>AI-powered recruiter that shortlists the best candidates for any job role in seconds using NLP & machine learning</p>
</div>
""", unsafe_allow_html=True)

# ── Premium Sidebar Controls ───────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🤖</div>
        <div>
            <div class="brand-title">AI Recruiter</div>
            <div class="brand-sub">Smart candidate ranking panel</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<h3>📋 Job Description</h3>', unsafe_allow_html=True)

    job_description = st.text_area(
        "Paste or type any job description:",
        value="""We are looking for an AI/ML Engineer with strong Python skills.
    Required: machine learning, deep learning, TensorFlow, scikit-learn,
    natural language processing, data analysis, model deployment.
    Experience with neural networks, transformers, and LLM is preferred.
    Minimum 3 years experience. Strong problem solving skills required.""",
        height=180,
        label_visibility="collapsed"
    )

    # Keywords
    if job_description:
        stop_words = {'and','or','the','a','an','is','are','we','for','with','in',
                      'of','to','be','experience','skills','required','preferred',
                      'minimum','years','strong','looking','problem','solving'}
        words = [w.strip('.,()') for w in job_description.lower().split()
                 if len(w) > 3 and w.lower().strip('.,()') not in stop_words]
        keywords = list(dict.fromkeys(words))[:10]
        tags_html = "".join([f'<span class="tag">{k}</span>' for k in keywords])
        st.markdown("**🔍 Keywords:**")
        st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h3>⚖️ Scoring Weights</h3>', unsafe_allow_html=True)
    skill_weight = st.slider("🎯 Skill Match", 0, 100, 70, step=5)
    exp_weight   = st.slider("📅 Experience",  0, 100, 20, step=5)
    act_weight   = st.slider("⚡ Activity",    0, 100, 10, step=5)
    total = skill_weight + exp_weight + act_weight

    if total != 100:
        st.error(f"⚠️ Weights = {total}% · Must be 100%")
    else:
        st.success("✅ Weights = 100%")
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.08);border-radius:8px;padding:8px;margin-top:4px'>
            <div style='display:flex;gap:4px;height:8px;border-radius:6px;overflow:hidden'>
                <div style='width:{skill_weight}%;background:#7c3aed'></div>
                <div style='width:{exp_weight}%;background:#06b6d4'></div>
                <div style='width:{act_weight}%;background:#10b981'></div>
            </div>
            <div style='display:flex;gap:8px;margin-top:5px;font-size:10px;color:#c4b5fd'>
                <span>■ Skill {skill_weight}%</span>
                <span>■ Exp {exp_weight}%</span>
                <span>■ Act {act_weight}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    top_n = st.slider("👥 Top N Candidates", 3, 20, 10)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Main Dashboard ─────────────────────────────────────────────
st.markdown("""
<div class="dashboard-strip">
    <div><b>Upload → Rank → Analyze → Export</b></div>
    <span>Designed for a clean hackathon demo with no empty layout gaps.</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">📂 Candidate Data</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload candidates CSV files (select multiple)",
    type=["csv"],
    accept_multiple_files=True
)

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
    st.info(f"ℹ️ Using default dataset — {len(df)} candidates ready.")

st.write("")
run = st.button("🚀 Run AI Ranking", use_container_width=True)

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
