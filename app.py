import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Recruiter | India Runs Hackathon",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Intelligent Candidate Discovery System")
st.caption("India Runs Hackathon · Track 01 · AI-powered candidate shortlisting")
st.divider()

# ── Sidebar: Job Description ──────────────────────────────────
st.sidebar.header("📋 Job Description")
job_description = st.sidebar.text_area(
    "Paste the job description here:",
    value="""We are looking for an AI/ML Engineer with strong Python skills.
Required: machine learning, deep learning, TensorFlow, scikit-learn,
natural language processing, data analysis, model deployment.
Experience with neural networks, transformers, and LLM is preferred.
Minimum 3 years experience. Strong problem solving skills required.""",
    height=200
)

st.sidebar.divider()
st.sidebar.header("⚖️ Scoring Weights")
skill_weight = st.sidebar.slider("Skill Match Weight", 0, 100, 60, step=5)
exp_weight   = st.sidebar.slider("Experience Weight",  0, 100, 20, step=5)
act_weight   = st.sidebar.slider("Activity Weight",    0, 100, 20, step=5)

total = skill_weight + exp_weight + act_weight
if total != 100:
    st.sidebar.warning(f"⚠️ Weights sum to {total}%. Should be 100%.")

st.sidebar.divider()
top_n = st.sidebar.slider("Show Top N Candidates", 3, 20, 10)

# ── Load / Upload Dataset ─────────────────────────────────────
st.subheader("📂 Candidate Data")
uploaded = st.file_uploader("Upload your candidates CSV (optional)", type=["csv"])

@st.cache_data
def get_default_data():
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
        'activity_score': np.random.randint(60, 100, 50),
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
    df = pd.read_csv(uploaded)
    st.success(f"✅ Loaded {len(df)} candidates from your file!")
else:
    df = get_default_data()
    st.info(f"ℹ️ Using default dataset — {len(df)} candidates loaded.")

# ── Run Ranking ───────────────────────────────────────────────
if st.button("🚀 Run AI Ranking", type="primary", use_container_width=True):
    if total != 100:
        st.error("Please make sure weights add up to 100% before running.")
    else:
        with st.spinner("Analyzing candidates..."):
            vectorizer = TfidfVectorizer(stop_words='english')
            all_text = [job_description] + list(df['skills'])
            tfidf_matrix = vectorizer.fit_transform(all_text)
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            df['skill_match_score'] = similarity_scores * 100
            df['exp_score'] = (df['experience_years'] / df['experience_years'].max()) * 100
            df['final_score'] = (
                df['skill_match_score'] * (skill_weight / 100) +
                df['exp_score']         * (exp_weight  / 100) +
                df['activity_score']    * (act_weight  / 100)
            )
            df['rank'] = df['final_score'].rank(ascending=False).astype(int)
            df_ranked = df.sort_values('rank').reset_index(drop=True)

        st.success("✅ Ranking complete!")

        # ── Metrics row ───────────────────────────────────────
        top1 = df_ranked.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👑 Top Candidate", top1['name'])
        col2.metric("🎯 Top Score",     f"{top1['final_score']:.1f}")
        col3.metric("📊 Candidates",    len(df))
        col4.metric("⚡ Shortlisted",   top_n)

        st.divider()

        # ── Top N table ───────────────────────────────────────
        st.subheader(f"🏆 Top {top_n} Candidates")
        display_cols = ['rank', 'name', 'job_title', 'experience_years',
                        'skill_match_score', 'activity_score', 'final_score', 'education']
        top_df = df_ranked[display_cols].head(top_n).copy()
        top_df['skill_match_score'] = top_df['skill_match_score'].round(1)
        top_df['final_score']       = top_df['final_score'].round(1)
        top_df.columns = ['Rank','Name','Job Title','Exp (yrs)',
                          'Skill Match %','Activity','Final Score','Education']
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        # ── Charts ────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Visual Analysis")
        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            top10 = df_ranked.head(10)
            ax1.barh(top10['name'], top10['final_score'], color='#7C3AED')
            ax1.set_xlabel('Final Score')
            ax1.set_title('Top 10 Candidates')
            ax1.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sc = ax2.scatter(df['skill_match_score'], df['exp_score'],
                             c=df['final_score'], cmap='viridis', s=80, alpha=0.8)
            ax2.set_xlabel('Skill Match Score')
            ax2.set_ylabel('Experience Score')
            ax2.set_title('Score Distribution')
            plt.colorbar(sc, ax=ax2, label='Final Score')
            plt.tight_layout()
            st.pyplot(fig2)

        # ── Download ──────────────────────────────────────────
        st.divider()
        csv = df_ranked[['rank','name','job_title','experience_years',
                          'skill_match_score','activity_score','final_score']]\
              .round(2).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Ranked Results CSV",
            data=csv,
            file_name="ranked_candidates.csv",
            mime="text/csv",
            use_container_width=True
        )
