# local_model.py
import io
import os
import re
import traceback
from typing import Tuple, Dict

import fitz  # PyMuPDF
import docx  # python-docx

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------
# Pre-load all heavy libraries and models at startup.
# --------------------------
print("Starting up: Loading transformer models...")
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch

# Load models into memory once when the module is imported
_sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
_bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_bert_model = BertModel.from_pretrained("bert-base-uncased")
_bert_model.eval()
print("Transformer models loaded successfully.")

# --------------------------
# Built-in stopwords
# --------------------------
EN_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "resume", "job", "description", "work", "experience", "skill", "skills", "applicant", "application"
}

# --------------------------
# Job Suggestions Database
# --------------------------
JOB_SUGGESTIONS_DB = {
    "Data Scientist": {"python", "sql", "machine", "learning", "tensorflow", "pytorch", "analysis"},
    "Data Analyst": {"sql", "python", "excel", "tableau", "analysis", "statistics"},
    "Backend Developer": {"python", "java", "sql", "docker", "aws", "api", "git"},
    "Frontend Developer": {"react", "javascript", "html", "css", "git", "ui", "ux"},
    "Full-Stack Developer": {"python", "javascript", "react", "sql", "docker", "git"},
    "Machine Learning Engineer": {"python", "tensorflow", "pytorch", "machine", "learning", "docker", "cloud"},
    "Project Manager": {"agile", "scrum", "project", "management", "jira"}
}

# --------------------------
# Utilities: text extraction
# --------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [p.get_text("text") for p in doc]
        doc.close()
        return "\n".join(p for p in pages if p)
    except Exception as e:
        return f"[Error reading PDF: {e}]"


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    try:
        docx_io = io.BytesIO(docx_bytes)
        doc = docx.Document(docx_io)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    except Exception as e:
        return f"[Error reading DOCX: {e}]"


def extract_text_from_fileobj(file_obj) -> Tuple[str, str]:
    fname = "uploaded_file"
    try:
        fname = os.path.basename(file_obj.name)
        with open(file_obj.name, "rb") as f:
            raw_bytes = f.read()
        ext = fname.split('.')[-1].lower()
        if ext == "pdf":
            return (extract_text_from_pdf_bytes(raw_bytes), fname)
        elif ext == "docx":
            return (extract_text_from_docx_bytes(raw_bytes), fname)
        else:
            return (raw_bytes.decode("utf-8", errors="ignore"), fname)
    except Exception as exc:
        return (f"[Error reading uploaded file: {exc}\n{traceback.format_exc()}]", fname)

# --------------------------
# Text preprocessing
# --------------------------
def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    words = t.split()
    if remove_stopwords:
        words = [w for w in words if w not in EN_STOPWORDS]
    return " ".join(words)

# --------------------------
# Embedding helpers
# --------------------------
def get_sentence_embedding(text: str, mode: str = "sbert") -> np.ndarray:
    if mode == "sbert":
        return _sentence_transformer.encode([text], show_progress_bar=False)
    elif mode == "bert":
        tokens = _bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            out = _bert_model(**tokens)
            cls = out.last_hidden_state[:, 0, :].numpy()
        return cls
    else:
        raise ValueError("Unsupported mode")

def calculate_similarity(resume_text: str, job_text: str, mode: str = "sbert") -> float:
    r_emb = get_sentence_embedding(resume_text, mode=mode)
    j_emb = get_sentence_embedding(job_text, mode=mode)
    sim = cosine_similarity(r_emb, j_emb)[0][0]
    return float(np.round(sim * 100, 2))

# --------------------------
# Keyword analysis
# --------------------------
DEFAULT_KEYWORDS = {
    "skills": {"python", "nlp", "java", "sql", "tensorflow", "pytorch", "docker", "git", "react", "cloud", "aws",
               "azure"},
    "concepts": {"machine", "learning", "data", "analysis", "nlp", "vision", "agile", "scrum"},
    "roles": {"software", "engineer", "developer", "manager", "scientist", "analyst", "architect"},
}

def analyze_resume_keywords(resume_text: str, job_description: str):
    clean_resume = preprocess_text(resume_text)
    clean_job = preprocess_text(job_description)
    resume_words = set(clean_resume.split())
    job_words = set(clean_job.split())
    missing = {}
    for cat, kws in DEFAULT_KEYWORDS.items():
        missing_from_cat = [kw for kw in kws if kw in job_words and kw not in resume_words]
        if missing_from_cat:
            missing[cat] = sorted(missing_from_cat)
    low_resume = (resume_text or "").lower()
    sections_present = {
        "skills": "skills" in low_resume,
        "experience": "experience" in low_resume or "employment" in low_resume,
        "summary": "summary" in low_resume or "objective" in low_resume,
    }
    suggestions = []
    if any(missing.values()):
        for cat, kws in missing.items():
            for kw in kws:
                if cat == "skills":
                    suggestions.append(
                        f"Add keyword '{kw}' to your Skills section." if sections_present["skills"]
                        else f"Consider creating a Skills section to include '{kw}'."
                    )
                elif cat == "concepts":
                    suggestions.append(
                        f"Try to demonstrate your knowledge of '{kw}' in your Experience or Projects section."
                    )
                elif cat == "roles":
                    suggestions.append(f"Align your Summary/Objective to mention the title '{kw}'.")
    else:
        suggestions.append("Great job! Your resume contains many of the keywords found in the job description.")
    return missing, "\n".join(f"- {s}" for s in suggestions)

# --------------------------
# Project Section Analysis
# --------------------------
def extract_projects_section(resume_text: str) -> str:
    project_headings = ["projects", "personal projects", "academic projects", "portfolio"]
    end_headings = [
        "skills", "technical skills", "experience", "work experience",
        "education", "awards", "certifications", "languages", "references"
    ]
    lines = resume_text.split('\n')
    start_index = -1
    end_index = len(lines)

    # Find start
    for i, line in enumerate(lines):
        cleaned_line = line.strip().lower()
        if cleaned_line in project_headings:
            start_index = i
            break
    if start_index == -1:
        return "Could not automatically identify a 'Projects' section in this resume."

    # Find end (FIX: use lines[i], not stale 'line')
    for i in range(start_index + 1, len(lines)):
        cleaned_line = lines[i].strip().lower()
        if len(cleaned_line.split()) < 4 and cleaned_line in end_headings:
            end_index = i
            break

    project_section_lines = lines[start_index:end_index]
    return "\n".join(project_section_lines)

def analyze_projects_fit(projects_text: str, job_description_text: str, mode: str) -> str:
    if projects_text.startswith("Could not"):
        return "Cannot analyze project fit as no projects section was found."

    cleaned_projects = preprocess_text(projects_text)
    cleaned_job = preprocess_text(job_description_text)

    if not cleaned_projects:
        return "Projects section is empty or contains no relevant text to analyze."

    project_fit_score = calculate_similarity(cleaned_projects, cleaned_job, mode=mode)

    if project_fit_score >= 75:
        verdict = f"<p style='color:green;'>✅ <b>Highly Relevant ({project_fit_score:.2f}%)</b>: The projects listed are an excellent match for this job's requirements.</p>"
    elif project_fit_score >= 50:
        verdict = f"<p style='color:limegreen;'>👍 <b>Relevant ({project_fit_score:.2f}%)</b>: The projects show relevant skills and experience for this role.</p>"
    else:
        verdict = f"<p style='color:orange;'>⚠️ <b>Moderately Relevant ({project_fit_score:.2f}%)</b>: The projects may not directly align with the key requirements. Consider highlighting different aspects of your work.</p>"

    return verdict

# --------------------------
# Formatting and Suggestion Functions
# --------------------------
def format_missing_keywords(missing: Dict) -> str:
    if not any(missing.values()):
        return "✅ No critical keywords seem to be missing. Great job!"
    output = "### 🔑 Keywords Missing From Your Resume\n"
    for category, keywords in missing.items():
        if keywords:
            output += f"**Missing {category.capitalize()}:** {', '.join(keywords)}\n"
    return output

def suggest_jobs(resume_text: str) -> str:
    resume_words = set(preprocess_text(resume_text).split())
    suggestions = []
    for job_title, required_skills in JOB_SUGGESTIONS_DB.items():
        matched_skills = resume_words.intersection(required_skills)
        if len(matched_skills) >= 3:
            suggestions.append(job_title)
    if not suggestions:
        return "Could not determine strong job matches from the resume. Try adding more specific skills and technologies."
    output = "### 🚀 Job Titles You May Be a Good Fit For\n"
    for job in suggestions:
        output += f"- {job}\n"
    return output

def extract_top_keywords(text: str, top_n: int = 15) -> str:
    if not text.strip():
        return "Not enough text provided."
    try:
        vectorizer = TfidfVectorizer(stop_words=list(EN_STOPWORDS))
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = np.array(vectorizer.get_feature_names_out())
        scores = tfidf_matrix.toarray().flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        top_keywords = feature_names[top_indices]
        return ", ".join(top_keywords)
    except ValueError:
        return "Could not extract keywords (text may be too short)."
