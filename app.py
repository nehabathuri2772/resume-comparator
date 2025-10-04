# FastAPI + Gradio (mounted). Resume is uploaded as a file (PDF/DOCX/TXT).

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import gradio as gr
import io
import os
from typing import Tuple

# --- Lightweight CPU-only text extraction ---
from pypdf import PdfReader
from docx import Document

# ----------------------------
# Helpers: text extraction
# ----------------------------
ALLOWED_EXTS = {".pdf", ".docx", ".txt"}

def _ext_from_name(name: str) -> str:
    _, ext = os.path.splitext((name or "").lower())
    return ext

def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """
    Extract text from bytes for .pdf, .docx, .txt.
    Falls back to utf-8 decoding for unknown text-ish files.
    """
    ext = _ext_from_name(filename)

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts).strip()

    if ext == ".docx":
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    if ext == ".txt":
        return data.decode("utf-8", errors="ignore").strip()

    # Fallback: try text
    try:
        return data.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""

def extract_text_from_path(path: str) -> Tuple[str, str]:
    """
    Read file from disk (gradio File returns a filepath) and extract text.
    Returns (text, message).
    """
    if not path or not os.path.exists(path):
        return "", "No file found."
    name = os.path.basename(path)
    ext = _ext_from_name(name)
    if ext not in ALLOWED_EXTS:
        return "", f"Unsupported file type: {ext or 'unknown'}"

    with open(path, "rb") as f:
        data = f.read()
    text = extract_text_from_bytes(name, data)
    if not text:
        return "", "Could not extract any text from the file."
    return text, f"Extracted text from {name}."

# ----------------------------
# Scoring logic (placeholder)
# ----------------------------
def compare_resumes(resume_text: str, jd_text: str) -> dict:
    """
    Replace this stub with your full logic (embeddings, sections, etc.).
    For now we compute a simple overlap score of word sets.
    """
    import re
    tokenize = lambda s: set(re.findall(r"[a-zA-Z]+", (s or "").lower()))
    r = tokenize(resume_text)
    j = tokenize(jd_text)
    score = 0.0 if not j else len(r & j) / len(j)

    return {
        "match_score": round(float(score) * 100, 2),
        "matched_terms": sorted(list(r & j))[:50],
        "missing_terms": sorted(list(j - r))[:50],
        "notes": "Demo score = |resume∩JD| / |JD|. Swap with your real logic."
    }

# ----------------------------
# FastAPI app (global `api`)
# ----------------------------
api = FastAPI(title="Resume Comparator API", version="1.1")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/healthz")
def healthz():
    return {"status": "ok"}

# ---- JSON text API
class ComparePayload(BaseModel):
    resume_text: str
    job_description: str

@api.post("/api/analyze")
def analyze(payload: ComparePayload):
    return compare_resumes(payload.resume_text, payload.job_description)

# ---- Multipart API: file upload + jd_text
@api.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...), jd_text: str = Form(...)):
    data = await file.read()
    resume_text = extract_text_from_bytes(file.filename, data)
    result = compare_resumes(resume_text, jd_text)
    return {"filename": file.filename, "result": result}

# ----------------------------
# Gradio UI
# ----------------------------
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
)  # <- no .set(...)

CSS = """
/* simple rounded cards & spacing */
.card { border: 1px solid #e9ecef; border-radius: 14px; padding: 14px; background: #fff; }
"""

def gradio_compare_with_file(resume_file_path, jd_text):
    """
    resume_file_path: filepath (Gradio's default for gr.File with type='filepath')
    """
    text, msg = extract_text_from_path(resume_file_path) if resume_file_path else ("", "No file.")
    if not text:
        return f"{msg}\n\nPlease upload a PDF/DOCX/TXT with readable text."

    result = compare_resumes(text, jd_text)
    pretty = [
        f"Match score: {result['match_score']}%",
        "",
        "Top matched terms:",
        ", ".join(result["matched_terms"]) or "-",
        "",
        "Top missing terms:",
        ", ".join(result["missing_terms"]) or "-",
        "",
        result["notes"],
    ]
    return "\n".join(pretty)

with gr.Blocks(theme=THEME, css=CSS) as demo:
    gr.Markdown(
        "# 📄 Resume Comparator (File Upload)\n"
        "Upload your **resume file** (PDF/DOCX/TXT) and paste the **job description**."
    )
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["card"]):
                resume_file = gr.File(
                    label="Resume file (.pdf, .docx, .txt)",
                    file_types=[".pdf", ".docx", ".txt"],
                    type="filepath",
                    height=150,
                )
                jd_in = gr.Textbox(label="Job description", lines=12, placeholder="Paste JD text here")
                with gr.Row():
                    btn = gr.Button("Analyze", variant="primary")
                    clr = gr.Button("Clear")

        with gr.Column(scale=7):
            with gr.Group(elem_classes=["card"]):
                out = gr.Textbox(label="Result", lines=16)

    btn.click(gradio_compare_with_file, inputs=[resume_file, jd_in], outputs=[out])
    clr.click(lambda: (None, "", ""), outputs=[resume_file, jd_in, out])

    gr.Markdown(
        "### API\n"
        "- Text JSON: `POST /api/analyze` with `{resume_text, job_description}`\n"
        "- File JSON: `POST /api/analyze-file` (multipart) with `file` and `jd_text`\n"
        "- Health: `GET /healthz`"
    )

# 1) Mount Gradio at a path WITH trailing slash
app = gr.mount_gradio_app(api, demo, path="/gradio/")

# 2) Redirect roots to the canonical UI path
@api.get("/")
def root():
    return RedirectResponse(url="/gradio/", status_code=307)

@api.get("/gradio")
def gradio_no_slash():
    return RedirectResponse(url="/gradio/", status_code=307)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
