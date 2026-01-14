import streamlit as st
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ================== CONFIG ==================
import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


nltk.download('stopwords')

st.set_page_config(
    page_title="Intelligent Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

# ================== SKILLS DB ==================
SKILLS_DB = [
    "python","machine learning","deep learning","sql","pandas","numpy",
    "data analysis","data visualization","statistics","nlp",
    "tensorflow","pytorch","scikit learn","excel","power bi","tableau"
]

# ================== FUNCTIONS ==================
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]',' ',text)
    text = " ".join(w for w in text.split() if w not in stopwords.words('english'))
    return text

def extract_skills(text):
    return list({skill for skill in SKILLS_DB if skill in text})

def similarity_score(resume, jd):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform([resume, jd])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def detect_experience(text):
    if "intern" in text or "fresher" in text:
        return "Fresher"
    if "year" in text or "experience" in text:
        return "Experienced"
    return "Not Mentioned"

# -------- GENAI FUNCTIONS --------
def genai_resume_feedback(missing_skills, experience):
    prompt = f"""
You are an HR expert.
Give short, clear resume improvement feedback.

Missing skills: {missing_skills}
Experience level: {experience}

Give 3 bullet points only.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

def genai_candidate_summary(resume_text):
    prompt = f"""
Summarize this resume in 2 lines for a recruiter:

{resume_text[:1500]}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

# ================== UI ==================
st.markdown("""
<h1 style='text-align:center;'>🤖 Intelligent Resume Screening System</h1>
<p style='text-align:center; color:grey;'>AI + GenAI Powered ATS</p>
<hr>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Job Description")
    jd = st.text_area("Paste job description here", height=220)

with col2:
    st.subheader("📎 Upload Resume")
    resume_file = st.file_uploader("Upload resume (PDF only)", type=["pdf"])

threshold = st.slider("🎯 Shortlisting Threshold", 0.3, 0.8, 0.6)
analyze = st.button("🔍 Analyze Resume", use_container_width=True)

# ================== PROCESS ==================
if analyze:
    if resume_file and jd.strip():
        with st.spinner("🤖 AI + GenAI analyzing resume..."):
            resume_text = extract_text(resume_file)

            # 🔒 ATS safety check
            if len(resume_text.strip()) < 100:
                st.error(
                    "Resume text could not be extracted properly. "
                    "Please upload an ATS-friendly (text-based) PDF."
                )
                st.stop()

            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(jd)

            sim = similarity_score(resume_clean, jd_clean)

            resume_skills = extract_skills(resume_clean)
            jd_skills = extract_skills(jd_clean)

            matched = list(set(resume_skills) & set(jd_skills))
            missing = list(set(jd_skills) - set(resume_skills))

            skill_score = len(matched) / max(len(jd_skills), 1)
            final_score = (0.6 * sim) + (0.4 * skill_score)

            readiness = int(final_score * 100)
            skill_gap = 100 - readiness

            experience = detect_experience(resume_clean)

            if final_score >= threshold:
                status = "✅ Shortlisted"
                fit = "High Fit"
                color = "green"
            elif final_score >= threshold - 0.15:
                status = "⚠️ Needs Review"
                fit = "Medium Fit"
                color = "orange"
            else:
                status = "❌ Rejected"
                fit = "Low Fit"
                color = "red"

        # ================== OUTPUT ==================
        st.markdown("---")
        st.subheader("📊 ATS Result")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Final Score", round(final_score,2))
        c2.metric("Readiness", f"{readiness}%")
        c3.metric("Skill Gap", f"{skill_gap}%")
        c4.metric("Experience", experience)

        st.progress(min(final_score,1.0))
        st.markdown(
            f"<h4 style='color:{color}; text-align:center;'>Final Status: {status} | {fit}</h4>",
            unsafe_allow_html=True
        )

        st.markdown("### 🧠 Skill Analysis")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.success("Matched Skills")
            st.write(", ".join(matched) if matched else "None")
        with sc2:
            st.error("Missing Skills")
            st.write(", ".join(missing) if missing else "None")

        # ================== GENAI OUTPUT ==================
        st.markdown("### 🤖 GenAI Candidate Summary")
        st.info(genai_candidate_summary(resume_text))

        st.markdown("### 🚀 GenAI Resume Improvement Suggestions")
        st.warning(genai_resume_feedback(missing, experience))

    else:
        st.warning("Please upload resume and enter job description")

# ================== FOOTER ==================
st.markdown("""
<hr>
<p style='text-align:center; font-size:13px; color:grey;'>
Final AI + GenAI ATS Capstone Project | Python • NLP • LLM • Streamlit
</p>
""", unsafe_allow_html=True)



