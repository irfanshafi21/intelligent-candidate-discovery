import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Recruiter | India Runs Hackathon",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; }

/* ── Always show sidebar toggle button ── */
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: all !important;
}

/* ── Keep sidebar toggle always visible ── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    background: #7c3aed !important;
    border-radius: 0 8px 8px 0 !important;
    color: white !important;
    width: 24px !important;
    height: 48px !important;
    align-items: center !important;
    justify-content: center !important;
    top: 50% !important;
    box-shadow: 2px 2px 8px rgba(124,58,237,0.4) !important;
}
[data-testid="collapsedControl"]:hover {
    background: #6d28d9 !important;
    width: 28px !important;
}
[data-testid="collapsedControl"] svg {
    color: white !important;
    fill: white !important;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1a0533 0%, #2d1b69 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,0.3) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.25);
    border: 1px solid rgba(139,92,246,0.5);
    color: #c4b5fd;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero p {
    color: #a5b4fc;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}

/* ── Section headers ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e1b4b;
    margin: 1.5rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #f8f7ff 0%, #ede9fe 100%);
    border: 1px solid #c4b5fd;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .label {
    font-size: 11px;
    font-weight: 500;
    color: #7c3aed;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #1e1b4b;
    line-height: 1;
}
.metric-card .sub {
    font-size: 11px;
    color: #6d28d9;
    margin-top: 4px;
}

/* ── Candidate table rows ── */
.candidate-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    transition: box-shadow 0.2s;
}
.candidate-card:hover { box-shadow: 0 4px 12px rgba(109,40,217,0.1); }
.rank-badge {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    flex-shrink: 0;
}
.rank-1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #fff; }
.rank-2 { background: linear-gradient(135deg, #9ca3af, #6b7280); color: #fff; }
.rank-3 { background: linear-gradient(135deg, #cd7c2e, #b45309); color: #fff; }
.rank-other { background: #f3f4f6; color: #6b7280; }

/* ── Score bar ── */
.score-bar-bg {
    background: #f3f4f6;
    border-radius: 20px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #7c3aed, #a855f7);
}

/* ── Keyword tags ── */
.tag {
    display: inline-block;
    background: rgba(139,92,246,0.3);
    color: #e9d5ff !important;
    font-size: 11px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px;
    border: 1px solid rgba(139,92,246,0.5);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0533 0%, #1e1b4b 100%) !important;
}
section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
section[data-testid="stSidebar"] textarea {
    color: #ffffff !important;
    background-color: #2d1b69 !important;
    border: 1px solid rgba(139,92,246,0.6) !important;
    caret-color: #ffffff !important;
}
section[data-testid="stSidebar"] .stTextArea textarea {
    color: #ffffff !important;
    background-color: #2d1b69 !important;
}
section[data-testid="stSidebar"] textarea::placeholder { color: #a5b4fc !important; }
section[data-testid="stSidebar"] input { color: #ffffff !important; background-color: #2d1b69 !important; }
section[data-testid="stSidebar"] .stTextArea > div > div {
    background-color: #2d1b69 !important;
}
section[data-testid="stSidebar"] .stFileUploader {
    background-color: #2d1b69 !important;
    border: 1px solid rgba(139,92,246,0.5) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stFileUploader > div {
    background-color: #2d1b69 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
    background-color: #2d1b69 !important;
    border: 1px dashed rgba(139,92,246,0.6) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] * {
    color: #c4b5fd !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button *,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button span,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button p {
    background-color: #4c1d95 !important;
    background: #4c1d95 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] small {
    color: #4c1d95 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] {
    background-color: #2d1b69 !important;
}
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] > div {
    background-color: #2d1b69 !important;
}
section[data-testid="stSidebar"] .stFileUploader div div div div {
    background-color: #2d1b69 !important;
    color: #c4b5fd !important;
}
section[data-testid="stSidebar"] .stFileUploader small {
    color: #a78bfa !important;
}
section[data-testid="stSidebar"] .stFileUploader span {
    color: #c4b5fd !important;
}
section[data-testid="stSidebar"] .stFileUploader button[kind="secondary"] {
    background-color: rgba(124,58,237,0.4) !important;
    color: #e9d5ff !important;
    border-color: #7c3aed !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] span {
    color: #c4b5fd !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] small {
    color: #a78bfa !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div { background: #7c3aed !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
section[data-testid="stSidebar"] label { color: #c4b5fd !important; }

/* ── Run button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.5) !important;
}

/* ── Info / success boxes ── */
.stAlert { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: #e5e7eb !important; margin: 1.5rem 0 !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: #f3f4f6 !important;
    color: #1e1b4b !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* Hide entire white dropzone box */
section[data-testid="stSidebar"] div[data-testid="stFileUploadDropzone"] {
    display: none !important;
}
section[data-testid="stSidebar"] .stFileUploader > div {
    background: transparent !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)

# ── Hero Section ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🇮🇳 India Runs Hackathon 2026 · Track 01 · Data & AI</div>
    <h1>🤖 Intelligent Candidate Discovery</h1>
    <p>AI-powered recruiter that shortlists the best candidates for any job role in seconds using NLP & machine learning</p>
</div>
""", unsafe_allow_html=True)

# Keep sidebar always expanded
st.markdown("""
<script>
var sidebarToggle = window.parent.document.querySelector('[data-testid="collapsedControl"]');
if(sidebarToggle) sidebarToggle.style.display = "block";
</script>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## 📋 Job Description")
st.sidebar.markdown('<p style="color:#a78bfa;font-size:11px;margin-top:-10px;">💡 Press <b>S</b> key to open/close sidebar</p>', unsafe_allow_html=True)
st.sidebar.markdown("Paste, type or upload any job description")

job_description = st.sidebar.text_area(
    "📝 Type or paste job description here:",
    value="""We are looking for an AI/ML Engineer with strong Python skills.
Required: machine learning, deep learning, TensorFlow, scikit-learn,
natural language processing, data analysis, model deployment.
Experience with neural networks, transformers, and LLM is preferred.
Minimum 3 years experience. Strong problem solving skills required.""",
    height=200
)


# Keyword tags
if job_description:
    stop_words = {'and','or','the','a','an','is','are','we','for','with','in',
                  'of','to','be','experience','skills','required','preferred',
                  'minimum','years','strong','looking','problem','solving'}
    words = [w.strip('.,()') for w in job_description.lower().split()
             if len(w) > 3 and w.lower().strip('.,()') not in stop_words]
    keywords = list(dict.fromkeys(words))[:12]
    tags_html = "".join([f'<span class="tag">{k}</span>' for k in keywords])
    st.sidebar.markdown("**🔍 Detected keywords:**")
    st.sidebar.markdown(tags_html, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚖️ Scoring Weights")
skill_weight = st.sidebar.slider("🎯 Skill Match", 0, 100, 70, step=5)
exp_weight   = st.sidebar.slider("📅 Experience",  0, 100, 20, step=5)
act_weight   = st.sidebar.slider("⚡ Activity",    0, 100, 10, step=5)
total = skill_weight + exp_weight + act_weight

if total != 100:
    st.sidebar.error(f"⚠️ Weights = {total}% · Must be 100%")
else:
    st.sidebar.success("✅ Weights = 100%")
    # Visual weight breakdown
    st.sidebar.markdown(f"""
    <div style='background:rgba(255,255,255,0.08);border-radius:8px;padding:10px;margin-top:8px'>
        <div style='display:flex;gap:4px;height:10px;border-radius:6px;overflow:hidden'>
            <div style='width:{skill_weight}%;background:#7c3aed'></div>
            <div style='width:{exp_weight}%;background:#06b6d4'></div>
            <div style='width:{act_weight}%;background:#10b981'></div>
        </div>
        <div style='display:flex;gap:12px;margin-top:6px;font-size:11px'>
            <span style='color:#a78bfa'>■ Skill {skill_weight}%</span>
            <span style='color:#67e8f9'>■ Exp {exp_weight}%</span>
            <span style='color:#6ee7b7'>■ Activity {act_weight}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
top_n = st.sidebar.slider("👥 Show Top N Candidates", 3, 20, 10)

# ── Candidate Data ─────────────────────────────────────────────
st.markdown('<div class="section-title">📂 Candidate Data</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload candidates CSV files (you can select multiple)", type=["csv"], accept_multiple_files=True)

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
    dfs = []
    for file in uploaded:
        dfs.append(pd.read_csv(file))
    df_upload = pd.concat(dfs, ignore_index=True)
    if len(df_upload) < 5:
        st.warning(f"⚠️ Only {len(df_upload)} candidates found. Using default 50-candidate dataset.")
        df = get_default_data()
    else:
        df = df_upload
        st.success(f"✅ {len(df)} candidates loaded from {len(uploaded)} file(s)!")
else:
    df = get_default_data()

st.markdown("---")

# ── Run Button ────────────────────────────────────────────────
run = st.button("🚀 Run AI Ranking", use_container_width=True)

if run:
    if total != 100:
        st.error("❌ Scoring weights must add up to 100% before running.")
    elif not job_description.strip():
        st.error("❌ Please enter a job description.")
    else:
        with st.spinner("🤖 Analyzing candidates..."):
            vectorizer = TfidfVectorizer(stop_words='english')
            all_text = [job_description] + list(df['skills'])
            tfidf_matrix = vectorizer.fit_transform(all_text)
            similarity_scores = cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:]
            ).flatten()

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

        # ── Metric Cards ──────────────────────────────────────
        top1 = df_ranked.iloc[0]
        max_score = df_ranked['final_score'].max()
        avg_score = df_ranked['final_score'].mean()

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="label">👑 Top Candidate</div>
                <div class="value" style="font-size:1.2rem">{top1['name']}</div>
                <div class="sub">{top1.get('job_title','')}</div>
            </div>
            <div class="metric-card">
                <div class="label">🎯 Top Score</div>
                <div class="value">{max_score:.1f}</div>
                <div class="sub">out of 100</div>
            </div>
            <div class="metric-card">
                <div class="label">📊 Total Candidates</div>
                <div class="value">{len(df2)}</div>
                <div class="sub">profiles analyzed</div>
            </div>
            <div class="metric-card">
                <div class="label">📈 Avg Score</div>
                <div class="value">{avg_score:.1f}</div>
                <div class="sub">across all candidates</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Ranked Candidate Cards ────────────────────────────
        st.markdown(f'<div class="section-title">🏆 Top {top_n} Shortlisted Candidates</div>', unsafe_allow_html=True)

        for i, row in df_ranked.head(top_n).iterrows():
            rank = int(row['rank'])
            if rank == 1:
                badge_cls = "rank-1"
                rank_icon = "🥇"
            elif rank == 2:
                badge_cls = "rank-2"
                rank_icon = "🥈"
            elif rank == 3:
                badge_cls = "rank-3"
                rank_icon = "🥉"
            else:
                badge_cls = "rank-other"
                rank_icon = f"#{rank}"

            score = row['final_score']
            skill = row['skill_match_score']
            bar_width = min(int(score), 100)

            st.markdown(f"""
            <div class="candidate-card">
                <div class="rank-badge {badge_cls}">{rank_icon if rank <= 3 else rank}</div>
                <div style="flex:1">
                    <div style="font-weight:600;color:#1e1b4b;font-size:15px">{row['name']}</div>
                    <div style="color:#6b7280;font-size:12px;margin-top:2px">
                        {row.get('job_title','')} &nbsp;·&nbsp;
                        {row.get('experience_years','')} yrs &nbsp;·&nbsp;
                        {row.get('education','')}
                    </div>
                </div>
                <div style="min-width:140px">
                    <div style="font-size:11px;color:#7c3aed;margin-bottom:4px">Skill Match: {skill:.1f}%</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_width}%"></div>
                    </div>
                </div>
                <div style="text-align:right;min-width:70px">
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#7c3aed">{score:.1f}</div>
                    <div style="font-size:10px;color:#9ca3af">Final Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Visual Analysis</div>', unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)

        with ch1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_facecolor('#faf9ff')
            ax1.set_facecolor('#faf9ff')
            top10 = df_ranked.head(10)
            colors = ['#7c3aed' if i == 0 else '#a78bfa' if i < 3 else '#c4b5fd'
                      for i in range(len(top10))]
            bars = ax1.barh(top10['name'], top10['final_score'],
                            color=colors, height=0.6)
            ax1.set_xlabel('Final Score', color='#6b7280', fontsize=10)
            ax1.set_title('Top 10 Candidates', color='#1e1b4b',
                          fontsize=12, fontweight='bold', pad=12)
            ax1.invert_yaxis()
            ax1.set_xlim(0, 100)
            ax1.tick_params(colors='#6b7280', labelsize=9)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#e5e7eb')
            ax1.spines['bottom'].set_color('#e5e7eb')
            for bar, val in zip(bars, top10['final_score']):
                ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                         f'{val:.1f}', va='center', fontsize=8, color='#7c3aed')
            plt.tight_layout()
            st.pyplot(fig1)

        with ch2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#faf9ff')
            ax2.set_facecolor('#faf9ff')
            scatter = ax2.scatter(
                df2['skill_match_score'], df2['experience_years'],
                c=df2['final_score'], cmap='RdPu', s=80, alpha=0.85,
                edgecolors='white', linewidths=0.5
            )
            ax2.set_xlabel('Skill Match Score', color='#6b7280', fontsize=10)
            ax2.set_ylabel('Experience (years)', color='#6b7280', fontsize=10)
            ax2.set_title('Skill Match vs Experience', color='#1e1b4b',
                          fontsize=12, fontweight='bold', pad=12)
            ax2.tick_params(colors='#6b7280', labelsize=9)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#e5e7eb')
            ax2.spines['bottom'].set_color('#e5e7eb')
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Final Score', color='#6b7280', fontsize=9)
            cbar.ax.yaxis.set_tick_params(color='#6b7280', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig2)

        # ── Download ──────────────────────────────────────────
        st.markdown("---")
        out_cols = [c for c in ['rank','name','job_title','experience_years',
                                 'skill_match_score','activity_score',
                                 'final_score','education'] if c in df_ranked.columns]
        csv = df_ranked[out_cols].round(2).to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Full Ranked Results (CSV)",
            data=csv,
            file_name="ranked_candidates.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ── Footer ────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center;padding:2rem 0 1rem;color:#9ca3af;font-size:12px'>
            🤖 Intelligent Candidate Discovery System &nbsp;·&nbsp;
            India Runs Hackathon 2026 &nbsp;·&nbsp; Built by Irfan Shafi
        </div>
        """, unsafe_allow_html=True)
