import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# ---------- Text Cleaning Helpers ----------

def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------- Model Loader (Sentence Embeddings) ----------

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        # Small, fast, widely used sentence-transformers model
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


# ---------- Keyword Extraction from Job Description ----------

def extract_keywords(jd_text: str, top_n: int = 20):
    jd_text = clean_text(jd_text)
    if not jd_text:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1000
    )
    X = vectorizer.fit_transform([jd_text])
    scores = X.toarray()[0]
    terms = np.array(vectorizer.get_feature_names_out())

    # Sort by tf-idf score
    idx = np.argsort(scores)[::-1]
    top_terms = terms[idx][:top_n]
    top_scores = scores[idx][:top_n]

    return list(zip(top_terms, top_scores))


# ---------- Section Detection ----------

SECTION_KEYWORDS = [
    "education",
    "experience",
    "projects",
    "skills",
    "certifications",
    "internship",
    "work experience",
]


def detect_sections(resume_text: str):
    text_lower = resume_text.lower()
    found = []
    for sec in SECTION_KEYWORDS:
        if sec in text_lower:
            found.append(sec)
    return found


# ---------- Main Analysis Function ----------

def analyze_resume(resume_text: str, jd_text: str):
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)

    if not resume_text or not jd_text:
        return {
            "overall_fit": 0,
            "semantic_similarity": 0,
            "keyword_coverage": 0,
            "missing_keywords": [],
            "matched_keywords": [],
            "sections_found": [],
            "length_words": len(resume_text.split()),
            "suggestions": ["Provide both resume text and job description."]
        }

    # 1. Semantic similarity using sentence embeddings
    model = get_embedding_model()
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_jd = model.encode(jd_text, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb_resume, emb_jd)[0][0])  # -1 to 1
    similarity_norm = (similarity + 1) / 2  # 0 to 1

    # 2. Keyword extraction from JD
    jd_keywords_scored = extract_keywords(jd_text, top_n=25)
    jd_keywords = [k for k, s in jd_keywords_scored]

    resume_lower = resume_text.lower()
    matched = []
    missing = []

    for kw in jd_keywords:
        # fuzzy-ish presence check
        if kw in resume_lower:
            matched.append(kw)
        else:
            missing.append(kw)

    if jd_keywords:
        keyword_coverage = len(matched) / len(jd_keywords)
    else:
        keyword_coverage = 0

    # 3. Section coverage
    sections_found = detect_sections(resume_text)

    # 4. Overall score (0–100)
    # Weight semantic similarity more than keywords
    overall_fit = (
        0.65 * similarity_norm +
        0.25 * keyword_coverage +
        0.10 * (len(sections_found) / max(len(SECTION_KEYWORDS), 1))
    ) * 100
    overall_fit = round(overall_fit, 1)

    # 5. Suggestions
    suggestions = []

    if similarity_norm < 0.6:
        suggestions.append(
            "Tailor your summary and experience to more closely match the role's responsibilities."
        )
    if keyword_coverage < 0.5:
        suggestions.append(
            "Add more role-specific keywords from the job description (skills, tools, responsibilities)."
        )
    if "projects" not in sections_found:
        suggestions.append("Add a 'Projects' section with concrete outcomes and technologies used.")
    if "skills" not in sections_found:
        suggestions.append("Add a dedicated 'Skills' section listing programming languages, tools and frameworks.")
    if len(resume_text.split()) < 200:
        suggestions.append("Your resume seems short. Add more detail to your experience and achievements.")
    if len(resume_text.split()) > 800:
        suggestions.append("Your resume is quite long. Consider trimming to 1–2 pages, focusing on impact.")

    if not suggestions:
        suggestions.append("Your resume is fairly well aligned. Fine-tune bullet points to make impact measurable.")

    result = {
        "overall_fit": overall_fit,
        "semantic_similarity": round(similarity_norm * 100, 1),
        "keyword_coverage": round(keyword_coverage * 100, 1),
        "missing_keywords": missing,
        "matched_keywords": matched,
        "sections_found": sections_found,
        "length_words": len(resume_text.split()),
        "suggestions": suggestions,
        "jd_keywords_scored": jd_keywords_scored,
    }
    return result
