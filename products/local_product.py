import os
import re
import numpy as np

# Local embeddings via sentence-transformers
from sentence_transformers import SentenceTransformer

# -------- App config (local) --------
EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL_LOCAL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_LEN           = int(os.getenv("MAX_TEXT_LEN", "20000"))

# Load local model once (cached for the process)
# First run will download weights; subsequent runs reuse the cache.
_model = SentenceTransformer(EMBED_MODEL_LOCAL)

# -------- Utilities --------
def _trim(s: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= MAX_LEN else s[:MAX_LEN]

def _tok_words(s: str) -> set:
    return set(re.findall(r"[a-zA-Z]+", (s or "").lower()))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _embed_local(text: str) -> np.ndarray:
    text = _trim(text)
    # encode returns a 1D vector for a single string when convert_to_numpy=True
    vec = _model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
    if vec.ndim != 1:
        vec = np.asarray(vec).reshape(-1)
    return vec.astype(np.float32)

# -------- Resume file readers --------
def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf(path: str) -> str:
    from pypdf import PdfReader
    text = []
    for p in PdfReader(path).pages:
        text.append(p.extract_text() or "")
    return "\n".join(text)

def _read_docx(path: str) -> str:
    import docx  # package: python-docx
    d = docx.Document(path)
    return "\n".join(par.text for par in d.paragraphs)

def read_resume_file(file_obj) -> str:
    """Gradio File -> extract text from PDF/DOCX/TXT."""
    if not file_obj:
        return ""
    path = getattr(file_obj, "name", "") or ""
    name = (getattr(file_obj, "orig_name", "") or path).lower()
    if not path:
        return ""
    if name.endswith(".pdf"):
        return _read_pdf(path)
    if name.endswith(".docx"):
        return _read_docx(path)
    if name.endswith(".txt"):
        return _read_txt(path)
    # best-effort fallback
    return _read_txt(path)

# -------- Core scoring --------
def compare_resume_to_jd(resume_text: str, jd_text: str) -> str:
    resume_text = _trim(resume_text)
    jd_text     = _trim(jd_text)
    if not resume_text or not jd_text:
        raise ValueError("Please upload a resume file and provide a job description.")

    r_vec = _embed_local(resume_text)
    j_vec = _embed_local(jd_text)
    score = _cosine(r_vec, j_vec) * 100.0

    r, j = _tok_words(resume_text), _tok_words(jd_text)
    matched = sorted(list(r & j))[:50]
    missing = sorted(list(j - r))[:50]

    tier = "Excellent" if score >= 80 else "Good" if score >= 65 else "Fair" if score >= 50 else "Low"

    lines = [
        f"Match score: {score:.2f}% ({tier})",
        f"Local model: {EMBED_MODEL_LOCAL}",
        "",
        "Top matched terms:",
        ", ".join(matched) if matched else "-",
        "",
        "Top missing terms:",
        ", ".join(missing) if missing else "-",
    ]
    return "\n".join(lines)

# -------- Gradio callback used by app.py --------
def run_local_product(resume_file, jd_text) -> str:
    try:
        resume_txt = read_resume_file(resume_file)
        return compare_resume_to_jd(resume_txt, jd_text)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
