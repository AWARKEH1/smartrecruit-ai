import json
import re
import PyPDF2
import streamlit as st

from agno.agent import Agent
from agno.models.openai import OpenAIChat


st.set_page_config(
    page_title="SmartRecruit AI",
    page_icon="🤖",
    layout="wide"
)


def init_session_state():
    defaults = {
        "openai_api_key": "",
        "resume_text": "",
        "analysis_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def extract_text_from_pdf(pdf_file) -> str:
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

        return text.strip()

    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def clean_json_response(response_text: str) -> str:
    response_text = response_text.strip()
    response_text = re.sub(r"```json", "", response_text)
    response_text = re.sub(r"```", "", response_text)
    return response_text.strip()


def create_matching_agent() -> Agent:
    return Agent(
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=st.session_state.openai_api_key,
        ),
        description="You are an AI recruiter and career advisor specialized in data science, machine learning, NLP and AI engineering.",
        instructions=[
            "Analyze the CV against the job description.",
            "Extract matching skills and missing skills.",
            "Evaluate technical fit, project fit, experience fit and language fit.",
            "Give concrete resume improvement recommendations.",
            "Return only valid JSON.",
        ],
        markdown=False,
    )


def analyze_cv_job_match(resume_text: str, job_description: str, agent: Agent) -> dict:
    prompt = f"""
You are SmartRecruit AI.

Analyze the following CV against the job description.

CV:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return ONLY a valid JSON object with this structure:

{{
  "match_score": 0,
  "profile_summary": "short summary of the candidate profile",
  "job_summary": "short summary of the job",
  "matching_skills": ["skill 1", "skill 2"],
  "missing_skills": ["skill 1", "skill 2"],
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "resume_improvements": ["improvement 1", "improvement 2"],
  "recommended_keywords": ["keyword 1", "keyword 2"],
  "cover_letter": "short customized cover letter in professional French",
  "final_recommendation": "clear recommendation"
}}

Rules:
- match_score must be between 0 and 100.
- Be honest but constructive.
- Focus on AI Engineer, Data Scientist, NLP Engineer and Machine Learning roles.
- The cover letter must be in French.
- Do not include markdown.
- Do not include explanations outside JSON.
"""

    response = agent.run(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    content = clean_json_response(content)

    return json.loads(content)


def display_analysis(result: dict):
    score = result.get("match_score", 0)

    st.subheader("🎯 Matching Score")
    st.progress(score / 100)
    st.metric("CV / Job Match", f"{score}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Matching Skills")
        for skill in result.get("matching_skills", []):
            st.write(f"- {skill}")

    with col2:
        st.subheader("⚠️ Missing Skills")
        for skill in result.get("missing_skills", []):
            st.write(f"- {skill}")

    st.subheader("👤 Profile Summary")
    st.write(result.get("profile_summary", ""))

    st.subheader("💼 Job Summary")
    st.write(result.get("job_summary", ""))

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("💪 Strengths")
        for item in result.get("strengths", []):
            st.write(f"- {item}")

    with col4:
        st.subheader("🔧 Weaknesses")
        for item in result.get("weaknesses", []):
            st.write(f"- {item}")

    st.subheader("🚀 Resume Improvement Suggestions")
    for item in result.get("resume_improvements", []):
        st.write(f"- {item}")

    st.subheader("🔑 Recommended Keywords")
    st.write(", ".join(result.get("recommended_keywords", [])))

    st.subheader("✉️ Generated Cover Letter")
    st.text_area(
        "Cover Letter",
        value=result.get("cover_letter", ""),
        height=250,
    )

    st.subheader("📌 Final Recommendation")
    st.info(result.get("final_recommendation", ""))


def main():
    init_session_state()

    st.title("🤖 SmartRecruit AI")
    st.write("AI-powered CV and job offer matching assistant for Data Science and AI roles.")

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
        )

        if api_key:
            st.session_state.openai_api_key = api_key

        st.markdown("---")
        st.write("Built with Python, Streamlit, PyPDF2 and LLMs.")

    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    st.header("1. Upload your CV")
    resume_file = st.file_uploader("Upload your CV as PDF", type=["pdf"])

    if resume_file:
        with st.spinner("Extracting text from CV..."):
            st.session_state.resume_text = extract_text_from_pdf(resume_file)

        if st.session_state.resume_text:
            st.success("CV processed successfully.")

            with st.expander("Preview extracted CV text"):
                st.text_area(
                    "Extracted CV text",
                    st.session_state.resume_text,
                    height=250,
                )

    st.header("2. Paste the job description")
    job_description = st.text_area(
        "Paste the job offer here",
        height=300,
        placeholder="Paste the full job description here...",
    )

    if st.button("Analyze Match"):
        if not st.session_state.resume_text:
            st.error("Please upload a CV first.")
            return

        if not job_description.strip():
            st.error("Please paste a job description first.")
            return

        with st.spinner("Analyzing CV and job description..."):
            try:
                agent = create_matching_agent()
                result = analyze_cv_job_match(
                    st.session_state.resume_text,
                    job_description,
                    agent,
                )
                st.session_state.analysis_result = result
                st.success("Analysis completed.")

            except json.JSONDecodeError:
                st.error("The AI response was not valid JSON. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if st.session_state.analysis_result:
        st.header("3. Results")
        display_analysis(st.session_state.analysis_result)


if __name__ == "__main__":
    main()
