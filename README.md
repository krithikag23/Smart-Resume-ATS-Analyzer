# 📄 Smart Resume ATS Analyzer

An AI-powered Resume Analyzer that evaluates how well a resume matches a job description — similar to how ATS (Applicant Tracking Systems) score resumes.

This tool calculates:
- Semantic similarity between resume and JD
- Keyword coverage based on required skills
- Resume structure (sections like Education, Projects, Skills, etc.)
- Overall ATS Fit Score (/100)

Built using lightweight Machine Learning models with **no heavy GPU dependencies** — works fully offline.

---

## 🚀 Features

✨ Text similarity scoring using TF-IDF  
✨ Detects matched & missing keywords from JD  
✨ Identifies resume structure (Standard ATS Sections)  
✨ Clear & actionable improvement suggestions  
✨ Fast + Local + Private (no data upload to servers)  
✨ Simple web UI using Streamlit  

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Web UI | Streamlit |
| NLP | Scikit-Learn TF-IDF |
| Data Handling | NumPy & Pandas |

No PyTorch, No spaCy, No HuggingFace → **Zero DLL issues on Windows** ✔️

---

## 📦 Installation

Clone the repository:

```sh
git clone https://github.com/<your-username>/Smart-Resume-ATS-Analyzer.git
cd Smart-Resume-ATS-Analyzer



