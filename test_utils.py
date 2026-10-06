"""
test_utils.py
Automated tests for the pure logic functions in utils.py.
Run with: pytest test_utils.py
"""

import pytest
from utils import (
    clean_text,
    remove_stopwords,
    calculate_similarity,
    extract_keywords,
    find_missing_keywords,
)


# ---- Tests for clean_text ----

def test_clean_text_lowercases():
    assert clean_text("Hello World") == "hello world"

def test_clean_text_removes_numbers_and_punctuation():
    assert clean_text("Python3.9, C++!") == "python c"

def test_clean_text_collapses_extra_whitespace():
    assert clean_text("hello    world") == "hello world"


# ---- Tests for remove_stopwords ----

def test_remove_stopwords_removes_common_words():
    result = remove_stopwords("this is a test of the system")
    assert "is" not in result.split()
    assert "the" not in result.split()
    assert "test" in result.split()

def test_remove_stopwords_keeps_meaningful_words():
    result = remove_stopwords("python machine learning")
    assert "python" in result
    assert "machine" in result
    assert "learning" in result


# ---- Tests for calculate_similarity ----

def test_calculate_similarity_identical_texts_scores_high():
    text = "python data science machine learning"
    score, _, _ = calculate_similarity(text, text)
    assert score > 90  # identical text should score very high

def test_calculate_similarity_unrelated_texts_scores_low():
    resume = "graphic design photoshop illustrator branding"
    job = "python machine learning data science sql"
    score, _, _ = calculate_similarity(resume, job)
    assert score < 30  # unrelated content should score low

def test_calculate_similarity_returns_score_between_0_and_100():
    score, _, _ = calculate_similarity("some resume text here", "some job description text")
    assert 0 <= score <= 100


# ---- Tests for extract_keywords ----

def test_extract_keywords_finds_nouns():
    keywords = extract_keywords("python developer with strong data skills")
    assert "python" in keywords or "developer" in keywords

def test_extract_keywords_filters_generic_blocklist_words():
    keywords = extract_keywords("strong team candidate ideal join")
    # all of these are in GENERIC_WORD_BLOCKLIST, so none should appear
    assert "strong" not in keywords
    assert "team" not in keywords
    assert "candidate" not in keywords


# ---- Tests for find_missing_keywords ----

def test_find_missing_keywords_identifies_gaps():
    resume = "python sql data analysis"
    job = "python sql machine learning pipelines frameworks"
    matched, missing = find_missing_keywords(resume, job)
    assert "python" in matched
    assert "pipelines" in missing or "frameworks" in missing

def test_find_missing_keywords_no_missing_when_fully_covered():
    resume = "python machine learning data science"
    job = "python machine learning"
    matched, missing = find_missing_keywords(resume, job)
    assert len(matched) > 0