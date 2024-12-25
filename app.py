from dotenv import load_dotenv
import os
import base64
import streamlit as st
import io
from PIL import Image
import pdf2image
import google.generativeai as genai
from crewai import Task
from crewai import Agent, Crew, Process
from crewai_tools import SerperDevTool
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from tools import tool
from agents import job_requirements_researcher, resume_swot_analyser
from tasks import research, resume_swot_analysis
from utils import *  # Assuming you have utility functions

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_gemini_response(input,pdf_cotent,prompt):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content([input,pdf_content[0],prompt])
    return response.text

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
sec_key = "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"  # Replace with your actual API token

# # Initialize HuggingFaceEndpoint with model parameters correctly assigned
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=sec_key,  # Use the correct token parameter
    temperature=0.7,                   # Pass temperature as a direct parameter
    
)



# Initialize the job requirement researcher and resume SWOT analyzer agents
job_requirements_researcher = Agent(
    role='Market Research Analyst',
    goal='Provide up-to-date market analysis of industry job requirements of the domain specified',
    backstory='An expert analyst with a keen eye for market trends.',
    tools=[tool],
    verbose=True,
    llm=llm,
    max_iters=1
)

resume_swot_analyser = Agent(
    role='Resume SWOT Analyser',
    goal='Perform a SWOT Analysis on the Resume based on the industry Job Requirements report from job_requirements_researcher and provide a json report.',
    backstory='An expert in hiring so has a great idea on resumes',
    verbose=True,
    llm=llm,
    max_iters=1,
    allow_delegation=True
)

# Initialize the Crew AI process and task definitions
research = Task(
    description='Research market job requirements based on a search query',
    expected_output='A report on what skills are required and real-time projects that can enhance chances of landing a job',
    agent=job_requirements_researcher,
    tools=[tool]
)

resume_swot_analysis = Task(
    description='Perform SWOT analysis on the resume based on job requirements research and provide match percentage',
    expected_output='A detailed SWOT analysis and resume match percentage with improvement suggestions',
    agent=resume_swot_analyser,
    tools=[tool],
    output_file='resume_review.md'
)

# Initialize the Crew
crew = Crew(
    agents=[job_requirements_researcher, resume_swot_analyser],
    tasks=[research, resume_swot_analysis],
    verbose=1,
    process=Process.sequential
)

# Streamlit UI setup
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS TRACKING SYSTEM")
input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

# Handling PDF conversion
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Handling response generation using Google Gemini
def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text

# Streamlit buttons and logic for invoking the Crew AI agents
submit1 = st.button("Tell Me About the Resume")  
submit2 = st.button("How Can I Improve my Skills") 
submit3 = st.button("Percentage match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description.
Please share your professional evaluation on whether the candidate's profile aligns with the role.
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality,
your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First, the output should come as percentage, followed by missing keywords, and final thoughts.
"""

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload the resume")

elif submit3:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload the resume")

# Start Crew AI tasks when needed
if submit2:
    if uploaded_file is not None:
        st.write("Starting Crew AI analysis...")
        result = crew.kickoff()  # Executes the Crew AI tasks
        st.subheader("Crew AI Result")
        st.write(result)
