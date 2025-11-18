import streamlit as st
from ats_analyzer import analyze_resume

st.set_page_config(
    page_title="Smart Resume ATS Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart Resume ATS Analyzer")
st.caption("AI-powered matching between your resume and a job description using modern NLP embeddings.")

st.markdown("----")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧾 Job Description")
    jd_text = st.text_area(
        "Paste the job description here",
        height=280,
        placeholder="Paste the full job description from LinkedIn / Naukri / company site..."
    )

with col2:
    st.subheader("👤 Your Resume (Text)")
    resume_text = st.text_area(
        "Paste your resume content here",
        height=280,
        placeholder="Paste the text version of your resume (or export from PDF as text)."
    )

st.markdown("----")

if st.button("🔍 Analyze Match"):
    with st.spinner("Analyzing with AI..."):
        result = analyze_resume(resume_text, jd_text)

    fit = result["overall_fit"]
    sim = result["semantic_similarity"]
    cov = result["keyword_coverage"]
    length = result["length_words"]

    st.subheader("📊 ATS Match Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Fit Score", f"{fit} / 100")
    m2.metric("Semantic Match", f"{sim} %")
    m3.metric("Keyword Coverage", f"{cov} %")
    m4.metric("Resume Length (words)", str(length))

    st.progress(min(int(fit), 100))

    st.markdown("----")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("✅ Matched Keywords")
        if result["matched_keywords"]:
            st.write(", ".join(sorted(set(result["matched_keywords"]))))
        else:
            st.write("No strong keyword matches found yet.")

    with c2:
        st.subheader("⚠️ Missing / Weak Keywords")
        if result["missing_keywords"]:
            st.write(", ".join(sorted(set(result["missing_keywords"]))))
        else:
            st.write("Great! You are covering most high-impact keywords.")

    st.markdown("----")

    st.subheader("📚 Sections Detected in Your Resume")
    if result["sections_found"]:
        st.write(", ".join(sorted(set(result["sections_found"]))))
    else:
        st.write("No standard sections detected. Consider adding sections like 'Education', 'Experience', 'Projects', 'Skills'.")

    st.markdown("----")

    st.subheader("🧠 Detailed Suggestions")
    for s in result["suggestions"]:
        st.markdown(f"- {s}")

else:
    st.info("Paste your resume and job description, then click 'Analyze Match' to see the ATS insights.")
