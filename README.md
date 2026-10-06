# 📄 Resume Job Match Scorer

A Streamlit app that measures how well a resume matches a job description, using TF-IDF and Cosine Similarity — a classic NLP technique used by real-world ATS (Applicant Tracking System) tools.

## Features

- **Upload PDF or DOCX resumes** — supports both formats
- **Multi-resume comparison** — upload several resumes at once and rank them against one job description
- **Match score** — a 0-100% score showing how closely a resume's wording aligns with the job posting
- **Keyword analysis** — see which important job keywords are matched in the resume, and which are missing
- **Downloadable report** — preview and download a full text summary of the results
- **Automated tests** — core logic is covered by a pytest suite

## How it works

1. Upload one or more resumes (PDF or DOCX)
2. Paste the job description
3. Click **Analyze Match**
4. Review the match score, gauge chart, and keyword breakdown for each resume

The app uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical vectors, then calculates **Cosine Similarity** between the resume and job description vectors to produce a match score.

> **Note:** This score measures word/keyword overlap, not candidate quality. A strong candidate's resume may still score moderately if it uses different wording than the job posting — the keyword analysis is there to help bridge that gap.

## Tech stack

- [Streamlit](https://streamlit.io/) — web UI
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorization and cosine similarity
- [NLTK](https://www.nltk.org/) — text tokenization, stopword removal, POS tagging
- [PyPDF2](https://pypi.org/project/PyPDF2/) — PDF text extraction
- [python-docx](https://python-docx.readthedocs.io/) — DOCX text extraction
- [Plotly](https://plotly.com/python/) — gauge chart visualization

## Setup and installation

1. **Clone the repository**