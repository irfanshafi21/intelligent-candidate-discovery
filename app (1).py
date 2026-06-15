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
st.caption("India Runs Hackathon · Track 01 · AI-powered candidate shortlisting for any job role")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("📋 Job Description")
st.sidebar.markdown("Paste, type, or upload any job description")

# Upload JD as txt file
jd_file = st.sidebar.file_uploader("Upload Job Description (.txt)", type=["txt"])

if jd_file:
    job_description = jd_file.read().decode("utf-8")
    st.sidebar.success("✅ Job description uploaded!")
else:
    job_description = st.sidebar.text_area(
        "Or type/paste here:",
        value="""We are looking for an AI/ML Engineer with strong Python skills.
Required: machine learning, deep learning, TensorFlow, scikit-learn,
natural language processing, data analysis, model deployment.
Experience with neural networks, transformers, and LLM is preferred.
Minimum 3 years experience. Strong problem solving skills required.""",
        height=180
    )

# Show detected keywords
if job_description:
    stop_words = {'and', 'or', 'the', 'a', 'an', 'is', 'are', 'we', 'for',
                  'with', 'in', 'of', 'to', 'be', 'experience', 'skills',
                  'required', 'preferred', 'minimum', 'years', 'strong',
                  'looking', 'problem', 'solving'}
    words = [w.strip('.,') for w in job_description.lower().split()
             if len(w) > 3 and w.lower() not in stop_words]
    unique_keywords = list(dict.fromkeys(words))[:10]
    st.sidebar.markdown("**🔍 Detected keywords:**")
    st.sidebar.markdown(" · ".join([f"`{k}`" for k in unique_keywords]))

st.sidebar.divider()

# Weights
st.sidebar.header("⚖️ Scoring Weights")
skill_weight = st.sidebar.slider("Skill Match", 0, 100, 70, step=5)
exp_weight   = st.sidebar.slider("Experience",  0, 100, 20, step=5)
act_weight   = st.sidebar.slider("Activity",    0, 100, 10, step=5)
total = skill_weight + exp_weight + act_weight
if total != 100:
    st.sidebar.warning(f"⚠️ Weights = {total}%. Must be 100%.")
else:
    st.sidebar.success("✅ Weights = 100%")

st.sidebar.divider()
top_n = st.sidebar.slider("Show Top N Candidates", 3, 20, 10)

# ── Candidate Data ─────────────────────────────────────────────
st.subheader("📂 Candidate Data")
st.markdown("Upload your own CSV **or** use the default 50-candidate dataset.")

col_upload, col_template = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader("Upload Candidates CSV", type=["csv"])

with col_template:
    # Provide a template CSV for download
    template = pd.DataFrame({
        'name': ['John Doe', 'Jane Smith'],
        'skills': ['Python machine learning TensorFlow deep learning',
                   'JavaScript React Node.js HTML CSS'],
        'experience_years': [5, 3],
        'job_title': ['ML Engineer', 'Frontend Developer'],
        'activity_score': [85, 78],
        'education': ['M.Tech CS', 'B.Tech IT']
    })
    st.download_button(
        "📥 Download CSV Template",
        template.to_csv(index=False).encode('utf-8'),
        file_name="candidate_template.csv",
        mime="text/csv",
        use_container_width=True
    )

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
    df = pd.read_csv(uploaded)
    st.success(f"✅ Loaded {len(df)} candidates from your file!")
else:
    df = get_default_data()
    st.info(f"ℹ️ Using default dataset — {len(df)} candidates loaded.")

# ── Run Ranking ────────────────────────────────────────────────
st.divider()
run = st.button("🚀 Run AI Ranking", type="primary", use_container_width=True)

if run:
    if total != 100:
        st.error("❌ Weights must add up to 100% before running.")
    elif not job_description.strip():
        st.error("❌ Please enter a job description.")
    else:
        with st.spinner("Analyzing candidates for this role..."):

            # TF-IDF skill matching
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

        # Metrics
        top1 = df_ranked.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👑 Top Candidate", top1['name'])
        c2.metric("🎯 Top Score",     f"{top1['final_score']:.1f}")
        c3.metric("📊 Total Candidates", len(df2))
        c4.metric("⚡ Shortlisted",   top_n)

        st.divider()

        # Ranked table
        st.subheader(f"🏆 Top {top_n} Candidates for this Role")
        cols = ['rank', 'name', 'job_title', 'experience_years',
                'skill_match_score', 'activity_score', 'final_score', 'education']
        # Only show columns that exist
        cols = [c for c in cols if c in df_ranked.columns]
        top_df = df_ranked[cols].head(top_n).copy()
        top_df['skill_match_score'] = top_df['skill_match_score'].round(1)
        top_df['final_score']       = top_df['final_score'].round(1)
        rename = {
            'rank': 'Rank', 'name': 'Name', 'job_title': 'Job Title',
            'experience_years': 'Exp (yrs)', 'skill_match_score': 'Skill Match %',
            'activity_score': 'Activity', 'final_score': 'Final Score',
            'education': 'Education'
        }
        top_df.rename(columns=rename, inplace=True)
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        # Charts
        st.divider()
        st.subheader("📊 Visual Analysis")
        ch1, ch2 = st.columns(2)

        with ch1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            top10 = df_ranked.head(10)
            ax1.barh(top10['name'], top10['final_score'], color='#7C3AED')
            ax1.set_xlabel('Final Score')
            ax1.set_title(f'Top 10 Candidates')
            ax1.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig1)

        with ch2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sc = ax2.scatter(
                df2['skill_match_score'], df2['experience_years'],
                c=df2['final_score'], cmap='viridis', s=80, alpha=0.8
            )
            ax2.set_xlabel('Skill Match Score')
            ax2.set_ylabel('Experience (years)')
            ax2.set_title('Skill vs Experience')
            plt.colorbar(sc, ax=ax2, label='Final Score')
            plt.tight_layout()
            st.pyplot(fig2)

        # Download
        st.divider()
        out_cols = ['rank', 'name', 'job_title', 'experience_years',
                    'skill_match_score', 'activity_score', 'final_score']
        out_cols = [c for c in out_cols if c in df_ranked.columns]
        csv = df_ranked[out_cols].round(2).to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Ranked Results CSV",
            data=csv,
            file_name="ranked_candidates.csv",
            mime="text/csv",
            use_container_width=True
        )
