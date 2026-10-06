"""
main.py
Streamlit UI for the Resume Job Match Scorer. All matching/scoring logic
lives in utils.py; this file only handles layout, inputs, and displaying results.
"""

import streamlit as st
import plotly.graph_objects as go
import nltk

from utils import (
    extract_resume_text,
    calculate_similarity,
    find_missing_keywords,
    MIN_JOB_DESCRIPTION_LENGTH,
    MIN_RESUME_TEXT_LENGTH,
    MAX_FILE_SIZE_BYTES,
    LOW_MATCH_THRESHOLD,
    GOOD_MATCH_THRESHOLD,
)

# Download nltk resources
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")

# ---- Page setup ----
st.set_page_config(page_title="Resume Job Match Scorer", page_icon="📄", layout="wide")

st.markdown("""

Upload your resume (PDF) and paste a job description to see how well they match!
This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job requirements.

""")

with st.sidebar:
    st.header("About")
    st.info("""

    This tool helps you:
    - Measure how your resume matches a job description
    - Identify important job keywords
    - Improve your resume based on missing terms

""")

    st.header("How It works")
    st.write("""
    1. Upload your resume (PDF)
    2. Upload the job description 
    3. Click **Analyze Match**
    4. Review Score & Suggestion

""")

def build_report_text(valid_results, job_description):
    """Build a plain-text summary of all analyzed resumes, for download."""
    lines = []
    lines.append("RESUME JOB MATCH SCORER — RESULTS REPORT")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Job Description:")
    lines.append(job_description.strip())
    lines.append("")
    lines.append("=" * 50)
    lines.append("")

    for rank, result in enumerate(valid_results, start=1):
        lines.append(f"#{rank} — {result['name']}")
        lines.append(f"Match Score: {result['score']:.2f}%")
        lines.append("")
        lines.append("Matched Keywords:")
        lines.append(", ".join(result['matched']) if result['matched'] else "None found")
        lines.append("")
        lines.append("Missing Keywords:")
        lines.append(", ".join(result['missing']) if result['missing'] else "None — great coverage!")
        lines.append("")
        lines.append("-" * 50)
        lines.append("")

    return "\n".join(lines)

def main():
    uploaded_files = st.file_uploader(
    "Upload your resume(s) (PDF or DOCX) — you can upload more than one to compare",
    type=['pdf', 'docx'],
    accept_multiple_files=True
)

    if uploaded_files:
        for f in uploaded_files:
            if f.size > MAX_FILE_SIZE_BYTES:
                st.error(f"'{f.name}' is too large. Please upload files smaller than 5MB.")
                return

    job_description = st.text_area("Paste the job description", height=200)

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.job_description = None

    if st.button("Analyze Match"):
        if not uploaded_files:
            st.warning("Please upload at least one resume")
            return
        if not job_description:
            st.warning("Please paste job description")
            return

        if len(job_description.strip()) < MIN_JOB_DESCRIPTION_LENGTH:
            st.warning("Job description seems too short. Please paste the full job description for accurate results.")
            return

        results = []

        with st.spinner("Analyzing your resume(s)...."):
            for uploaded_file in uploaded_files:
                resume_text = extract_resume_text(uploaded_file)

                if not resume_text or len(resume_text.strip()) < MIN_RESUME_TEXT_LENGTH:
                    results.append({
                        "name": uploaded_file.name,
                        "error": "Could not extract readable text from this file."
                    })
                    continue

                similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)
                matched_keywords, missing_keywords = find_missing_keywords(resume_processed, job_processed)

                results.append({
                    "name": uploaded_file.name,
                    "score": similarity_score,
                    "matched": matched_keywords,
                    "missing": missing_keywords,
                })

        # Sort successful results by score, highest first
        valid_results = sorted(
            [r for r in results if "score" in r],
            key=lambda r: r["score"],
            reverse=True
        )
        failed_results = [r for r in results if "error" in r]

        st.session_state.results = valid_results
        st.session_state.failed_results = failed_results
        st.session_state.job_description = job_description

    if st.session_state.results:
        valid_results = st.session_state.results
        failed_results = st.session_state.failed_results
        job_description_for_report = st.session_state.job_description

        st.subheader("Results")

        if len(valid_results) > 1:
            st.markdown("**Quick Summary**")
            summary_data = {
                "Rank": [f"#{i+1}" for i in range(len(valid_results))],
                "Resume": [r["name"] for r in valid_results],
                "Match Score": [f"{r['score']:.2f}%" for r in valid_results],
            }
            st.table(summary_data)

        col_caption, col_reset = st.columns([4, 1])
        with col_caption:
            if len(valid_results) > 1:
                st.caption(f"Ranked from best to lowest match, out of {len(valid_results)} resume(s) analyzed.")
        with col_reset:
            if st.button("🔄 Clear Results"):
                st.session_state.results = None
                st.session_state.failed_results = None
                st.session_state.job_description = None
                st.rerun()
                
        if valid_results:
            report_text = build_report_text(valid_results, job_description_for_report)

            with st.expander("👁️ Preview Report (click to view before downloading)"):
                st.text(report_text)

            st.download_button(
                label="📥 Download Results Report",
                data=report_text,
                file_name="resume_match_report.txt",
                mime="text/plain",
            )

        for rank, result in enumerate(valid_results, start=1):
            score = result["score"]
            colours = ['#ff4b4b', '#ffa726', '#0f9d58']
            colour_index = min(int(score // 33), 2)

            label = f"#{rank} — {result['name']}" if len(valid_results) > 1 else result['name']
            with st.expander(label, expanded=(rank == 1)):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': colours[colour_index]},
                        'steps': [
                            {'range': [0, LOW_MATCH_THRESHOLD], 'color': '#ffe0e0'},
                            {'range': [LOW_MATCH_THRESHOLD, GOOD_MATCH_THRESHOLD], 'color': '#fff3d9'},
                            {'range': [GOOD_MATCH_THRESHOLD, 100], 'color': '#e2f7e9'},
                        ],
                    }
                ))
                fig.update_layout(height=250, margin=dict(t=20, b=10, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_chart_{rank}")

                if score < LOW_MATCH_THRESHOLD:
                    st.warning("Low Match, consider tailoring your resume more closely.")
                elif score < GOOD_MATCH_THRESHOLD:
                    st.info("Good Match. Your resume aligns fairly well.")
                else:
                    st.success("Excellent Match! Your resume strongly aligns.")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Matched Keywords**")
                    st.write(", ".join(result["matched"]) if result["matched"] else "No strong keyword matches found.")
                with col2:
                    st.markdown("**❌ Missing Keywords**")
                    if result["missing"]:
                        st.write(", ".join(result["missing"]))
                    else:
                        st.write("Great! No major missing keywords.")

        for result in failed_results:
            st.error(f"'{result['name']}': {result['error']}")

    st.divider()
    st.caption("Built with Streamlit, scikit-learn, and NLTK")

if __name__ == "__main__":
    main()