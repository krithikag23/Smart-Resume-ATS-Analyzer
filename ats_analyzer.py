import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SECTION_KEYWORDS = [
    "education",
    "experience",
    "projects",
    "skills",
    "certifications",
    "internship"
]

def clean_text(t):
    t = t or ""
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def extract_keywords(jd_text, top_n=20):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform([jd_text])
    scores = X.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    idx = scores.argsort()[::-1]
    return terms[idx][:top_n]

def detect_sections(resume_text):
    txt = resume_text.lower()
    return [sec for sec in SECTION_KEYWORDS if sec in txt]

def analyze_resume(resume_text, jd_text):
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)

    if not resume_text or not jd_text:
        return None

    # Vectorize both texts together
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])

    sim = cosine_similarity(vectors[0], vectors[1])[0][0]
    sim_score = (sim + 1) / 2  # normalize 0-1

    jd_keywords = extract_keywords(jd_text)
    resume_lower = resume_text.lower()

    matched = [kw for kw in jd_keywords if kw in resume_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_lower]

    keyword_coverage = len(matched) / max(len(jd_keywords), 1)

    sections = detect_sections(resume_text)
    section_score = len(sections) / len(SECTION_KEYWORDS)

    overall_fit = (0.65*sim_score + 0.25*keyword_coverage + 0.10*section_score) * 100

    return {
        "overall_fit": round(overall_fit, 1),
        "semantic_similarity": round(sim_score * 100, 1),
        "keyword_coverage": round(keyword_coverage * 100, 1),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "sections_found": sections,
        "length_words": len(resume_text.split())
    }
