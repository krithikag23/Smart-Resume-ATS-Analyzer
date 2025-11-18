import re
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_md")

SECTION_KEYWORDS = [
    "education",
    "experience",
    "projects",
    "skills",
    "certifications",
    "internship",
]


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_embedding(text: str):
    doc = nlp(clean_text(text))
    return doc.vector.reshape(1, -1)


def extract_keywords(jd_text: str, top_n: int = 25):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform([jd_text])
    idx = X.toarray()[0].argsort()[::-1]
    terms = vectorizer.get_feature_names_out()
    return terms[idx][:top_n]


def detect_sections(resume_text: str):
    found = [sec for sec in SECTION_KEYWORDS if sec in resume_text.lower()]
    return found


def analyze_resume(resume_text: str, jd_text: str):
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)

    if not resume_text or not jd_text:
        return {"error": "Missing text input"}

    resume_vec = get_embedding(resume_text)
    jd_vec = get_embedding(jd_text)

    sim = cosine_similarity(resume_vec, jd_vec)[0][0]
    sim_score = (sim + 1) / 2  

    jd_keywords = extract_keywords(jd_text)
    resume_lower = resume_text.lower()

    matched = [kw for kw in jd_keywords if kw in resume_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_lower]

    keyword_coverage = len(matched) / max(len(jd_keywords), 1)

    sections = detect_sections(resume_text)
    section_score = len(sections) / len(SECTION_KEYWORDS)

    overall_fit = (0.65 * sim_score + 0.25 * keyword_coverage + 0.10 * section_score) * 100
    overall_fit = round(overall_fit, 1)

    return {
        "overall_fit": overall_fit,
        "semantic_similarity": round(sim_score * 100, 1),
        "keyword_coverage": round(keyword_coverage * 100, 1),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "sections_found": sections,
        "length_words": len(resume_text.split()),
    }
