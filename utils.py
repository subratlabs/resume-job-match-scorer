"""
utils.py
Core logic for the Resume Job Match Scorer: PDF extraction, text cleaning,
similarity scoring, and keyword comparison. Kept separate from the UI (app.py)
so the logic can be tested and reused independently of Streamlit.
"""

import re
from collections import Counter

import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ---- Constants ----
MIN_JOB_DESCRIPTION_LENGTH = 30      # characters
MIN_RESUME_TEXT_LENGTH = 50          # characters
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
LOW_MATCH_THRESHOLD = 40
GOOD_MATCH_THRESHOLD = 70
KEYWORD_POS_TAGS = ('NN', 'NNS', 'NNP', 'JJ')  # nouns and adjectives
TOP_N_KEYWORDS = 15

# Generic job-posting words that aren't real skills/tools, filtered out of keyword results
GENERIC_WORD_BLOCKLIST = {
    'strong', 'ideal', 'candidate', 'join', 'team', 'teams', 'role', 'ability',
    'experience', 'years', 'work', 'working', 'skills', 'knowledge', 'understanding',
    'requirements', 'responsibilities', 'looking', 'excellent', 'good', 'great',
    'proven', 'preferred', 'required', 'plus', 'must', 'including', 'related',
    'environment', 'opportunity', 'company', 'position', 'career', 'growth',
}

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded PDF file, page by page.
    Returns an empty string if the PDF is encrypted or unreadable.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        if pdf_reader.is_encrypted:
            return ""

        text = ""
        for page in pdf_reader.pages:
            text = text + (page.extract_text() or "")
        return text
    except Exception:
        return ""


def extract_text_from_docx(uploaded_file) -> str:
    """
    Extract all text from an uploaded DOCX (Word) file, paragraph by paragraph.
    Returns an empty string if the file is unreadable.
    """
    try:
        document = Document(uploaded_file)
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return text
    except Exception:
        return ""


def extract_resume_text(uploaded_file) -> str:
    """
    Extract text from an uploaded resume, detecting whether it's a PDF or DOCX
    based on the file name, and routing to the correct extraction function.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)   # ✅ correct
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)  # ✅ correct
    else:
        return ""


def clean_text(text: str) -> str:
    """Lowercase text and strip out punctuation/numbers/extra whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove common filler words (the, is, are, etc.) from text."""
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])


def calculate_similarity(resume_text: str, job_description: str):
    """
    Compute a TF-IDF + Cosine Similarity match score (0-100) between a resume
    and a job description. Returns the score plus the cleaned/processed text
    for both, so they can be reused for keyword extraction.
    """
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed


def extract_keywords(text: str) -> list:
    """Extract meaningful words (nouns/adjectives) from text using POS tagging,
    filtering out generic job-posting words that aren't real skills."""
    words = word_tokenize(text)
    tagged = pos_tag(words)
    return [
        word for word, tag in tagged
        if tag in KEYWORD_POS_TAGS and len(word) > 2 and word not in GENERIC_WORD_BLOCKLIST
    ]

def find_missing_keywords(resume_processed: str, job_processed: str, top_n: int = TOP_N_KEYWORDS):
    """
    Compare job description keywords against resume keywords.
    Returns (matched_keywords, missing_keywords), each ranked by how often
    the word appears in the job description.
    """
    job_keywords = extract_keywords(job_processed)
    resume_keywords = set(extract_keywords(resume_processed))
    job_keyword_counts = Counter(job_keywords)

    matched = []
    missing = []

    for word, count in job_keyword_counts.most_common():
        if word in resume_keywords:
            if word not in matched:
                matched.append(word)
        else:
            if word not in missing:
                missing.append(word)

    return matched[:top_n], missing[:top_n]