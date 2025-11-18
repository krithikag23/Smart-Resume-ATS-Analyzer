import streamlit as st
from ats_analyzer import analyze_resume

st.set_page_config(page_title="Smart Resume ATS Analyzer", page_icon="📄")

st.title("📄 Smart Resume ATS Analyzer")
st.caption("ATS-friendly score based on text similarity + keyword coverage")

col1, col2 = st.columns(2)

with col1:
    jd_text = st.text_area("Job Description", height=250)

with col2:
    resume_text = st.text_area("Resume Text", height=250)

if st.button("Analyze"):
    result = analyze_resume(resume_text, jd_text)
    if not result:
        st.error("Add both Resume and Job Description!")
    else:
        st.metric("Overall Fit Score", f"{result['overall_fit']}/100")
        st.metric("Semantic Match", f"{result['semantic_similarity']}%")
        st.metric("Keyword Coverage", f"{result['keyword_coverage']}%")
        st.write("Matched Keywords:", ", ".join(result["matched_keywords"]))
        st.write("Missing Keywords:", ", ".join(result["missing_keywords"]))
        st.write("Sections Found:", ", ".join(result["sections_found"]))
