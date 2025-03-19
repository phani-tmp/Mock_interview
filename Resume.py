
import pdfplumber
import docx

from fuzzywuzzy import fuzz
import json

import requests


GROQ_API_KEY = "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"  # Replace with your actual Groq API key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF resume."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n" if page.extract_text() else ""
    return text

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX resume."""
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])


def fetch_job_description(role, company):
    """Generate a job description using Groq API."""
    prompt = f"""
    Generate a detailed job description for the role of {role} at {company}.
    Include key responsibilities, required skills, and qualifications.
    Return the job description as a plain text string.
    """
    response = query_groq_api(prompt)
    return response.strip() if response else "Job description not available."


def query_groq_api(prompt):
    """Query the Groq API with a given prompt and return the response."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-2.5-32b",  # Replace with a valid Groq model
        "messages": [{"role": "user", "content": prompt}],  # Use "user" role for user prompts
        "max_tokens": 500
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad status codes (4xx, 5xx)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print(f"Error querying Groq API: {e}")
        return None

def extract_skills_from_resume(resume_text):
    """Extract skills from the resume text using Groq API."""
    prompt = f"""
    Please extract strictly the key technical skills from the following resume text.
    Only return the skills in a list format, like this: ["skill1", "skill2", "skill3", ...].
    The format must not include any other information. 
    Resume Text: {resume_text}
    Return only a list of skills in JSON format.
    """
    response = query_groq_api(prompt)
    try:
        skills = json.loads(response)
        # print("Extracted Skills from Resume:", skills)  # Add print for debugging
        return skills
    except json.JSONDecodeError:
        print("Error in parsing skills from resume.")
        return []


def extract_skills_from_jd(jd_text):
    """Extract skills from the job description text using Groq API."""
    prompt = f"""
    Please extract strictly the key technical skills from the following job description text.
    Only return the skills in a list format, like this: ["skill1", "skill2", "skill3", ...].
    The format must not include any other information. 
    jd_text: {jd_text}
    
    Return only a list of skills in JSON format.
    """
    response = query_groq_api(prompt)
    try:
        jd_skills = json.loads(response)
        # Handle case where the skills are inside a dictionary
        if isinstance(jd_skills, dict) and 'skills' in jd_skills:
            return jd_skills['skills']  # Extract skills list from dictionary
        return jd_skills  # Return as-is if it's already a list of skills
    except json.JSONDecodeError:
        return []

def compare_skills(resume_skills, jd_skills):
    """Compare extracted resume skills with job description skills."""
    
    # Check if the input is already a list of strings or a list of dictionaries
    if isinstance(resume_skills, list) and isinstance(resume_skills[0], str):
        resume_skill_list = resume_skills  # If it's already a list of skills (strings)
    elif isinstance(resume_skills, list) and isinstance(resume_skills[0], dict):
        resume_skill_list = []
        for item in resume_skills:
            resume_skill_list.extend(item.get('skills', []))  # Extract skills from each category
    else:
        raise ValueError("Invalid format for resume_skills.")

    if isinstance(jd_skills, list) and isinstance(jd_skills[0], str):
        jd_skill_list = jd_skills  # If it's already a list of skills (strings)
    elif isinstance(jd_skills, list) and isinstance(jd_skills[0], dict):
        jd_skill_list = []
        for item in jd_skills:
            jd_skill_list.extend(item.get('skills', []))  # Extract skills from each category
    else:
        raise ValueError("Invalid format for jd_skills.")
    
    # Convert to sets
    resume_skill_set = set(resume_skill_list)
    jd_skill_set = set(jd_skill_list)
    
    # Calculate similarity and missing skills
    similarity = fuzz.partial_ratio(" ".join(resume_skill_set), " ".join(jd_skill_set))
    missing_skills = list(jd_skill_set - resume_skill_set)
    
    return similarity, missing_skills



def generate_interview_level(similarity_score):
    """Generate interview level based on skill match similarity score."""
    if similarity_score > 80:
        level = "Advanced"
    elif similarity_score > 50:
        level = "Intermediate"
    else:
        level = "Beginner"
    return level

def analyze_resume_and_jd(resume_text, jd_text):
    """Analyze the resume and job description, and return missing skills and interview level."""
    # Extract skills from both resume and job description
    resume_skills = extract_skills_from_resume(resume_text)
    # print(resume_skills)
    jd_skills = extract_skills_from_jd(jd_text)
    
    # print(jd_skills)
    # Compare the skills
    similarity_score, missing_skills = compare_skills(resume_skills, jd_skills)
    
    # Generate interview level
    interview_level = generate_interview_level(similarity_score)
    
    return {
        "similarity_score": similarity_score,
        "missing_skills": missing_skills,
        "interview_level": interview_level
    }

# Example usage
if __name__ == "__main__":
    resume_path = "Resume_.pdf"  # Change to actual resume path
    role = "Machine Learning Intern"
    company = "Google"
    
    # Extract resume text
    if resume_path.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_path)
    elif resume_path.endswith(".docx"):
        resume_text = extract_text_from_docx(resume_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")
    
    
    
    # Fetch and process job description
    job_description = fetch_job_description(role, company)
    
    
    result = analyze_resume_and_jd(resume_text, job_description)
    print("Similarity Score:", result["similarity_score"])
    print("Missing Skills:", result["missing_skills"])
    print("Interview Level:", result["interview_level"])