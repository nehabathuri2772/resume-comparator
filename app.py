import os, re, time, random
from functools import lru_cache
from typing import Tuple
from html import escape

import numpy as np
import gradio as gr
from huggingface_hub import InferenceClient

# Single-port assets
os.environ["GRADIO_USE_MULTIPLE_PORTS"] = "0"
os.environ.setdefault("GRADIO_USE_CDN", "0")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

# --- Config ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN")                       # set in Space → Settings → Variables
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "20000"))
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", "40"))

hf = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN, timeout=REQ_TIMEOUT)

# --- Helpers ---
def _trim(s: str, n: int = MAX_TEXT_LEN) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

def _tok_words(s: str) -> set:
    return set(re.findall(r"[a-zA-Z]+", (s or "").lower()))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _backoff(a: int):  # tiny retry
    time.sleep(min(1.5 ** a, 8.0) + random.random())

@lru_cache(maxsize=256)
def _embed_cached(text: str) -> Tuple[bool, object]:
    try:
        feats = hf.feature_extraction(text)  # timeout is set on the client
        arr = np.array(feats, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        if arr.ndim != 1:
            return False, "Unexpected embedding shape from the Inference API."
        return True, arr
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _embed(text: str, retries: int = 2) -> np.ndarray:
    text = _trim(text)
    if not text:
        raise ValueError("Empty text to embed.")
    last = None
    for a in range(1, retries + 2):
        ok, val = _embed_cached(text)
        if ok:
            return val
        last = val
        _backoff(a)
    raise RuntimeError(last or "Unknown Inference API error.")

# --- Read resume files ---
def _read_txt(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf(p: str) -> str:
    from pypdf import PdfReader
    text = []
    for page in PdfReader(p).pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def _read_docx(p: str) -> str:
    import docx
    d = docx.Document(p)
    return "\n".join(par.text for par in d.paragraphs)

def read_resume_file(file_obj) -> str:
    if not file_obj:
        return ""
    path = getattr(file_obj, "name", "") or ""
    name = (getattr(file_obj, "orig_name", "") or path).lower()
    if not path:
        return ""
    try:
        if name.endswith(".pdf"):  return _read_pdf(path)
        if name.endswith(".docx"): return _read_docx(path)
        if name.endswith(".txt"):  return _read_txt(path)
        return _read_txt(path)  # best effort
    except Exception as e:
        raise RuntimeError(f"Resume file parse failed: {e}")

# --- Core ---
def compare_resumes_text_only(resume_text: str, jd_text: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set (Space → Settings → Variables).")

    resume_text = _trim(resume_text)
    jd_text = _trim(jd_text)
    if not resume_text or not jd_text:
        raise ValueError("Please provide both resume content and job description.")

    r_vec = _embed(resume_text)
    j_vec = _embed(jd_text)
    score = _cosine(r_vec, j_vec) * 100.0

    r, j = _tok_words(resume_text), _tok_words(jd_text)
    matched = sorted(list(r & j))[:50]
    missing = sorted(list(j - r))[:50]

    tier = "Excellent" if score >= 80 else "Good" if score >= 65 else "Fair" if score >= 50 else "Low"

    # Plain text output
    lines = [
        f"Match score: {score:.2f}% ({tier})",
        f"Model: {EMBED_MODEL}",
        "",
        "Top matched terms:",
        ", ".join(matched) if matched else "-",
        "",
        "Top missing terms:",
        ", ".join(missing) if missing else "-",
    ]
    return "\n".join(lines)

# --- Gradio callback ---
def run(resume_file, resume_text, jd_text):
    try:
        file_text = read_resume_file(resume_file) if resume_file else ""
        effective_resume = file_text or resume_text or ""
        return compare_resumes_text_only(effective_resume, jd_text)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

# --- UI (text only) ---
with gr.Blocks() as demo:
    gr.Markdown("# Resume Comparator (text output)")
    with gr.Row():
        with gr.Column():
            resume_file = gr.File(label="Upload resume (PDF/DOCX/TXT)",
                                  file_types=[".pdf",".docx",".txt"], file_count="single")
        jd_text = gr.Textbox(label="Job description (text)", lines=14)

    with gr.Row():
        go = gr.Button("Analyze", variant="primary")
        clear = gr.Button("Clear")

    result = gr.Textbox(label="Result", lines=16)

    go.click(run, inputs=[resume_file, resume_text, jd_text], outputs=[result])

    def _clear():
        return None, "", "", ""
    clear.click(_clear, outputs=[resume_file, resume_text, jd_text, result])

    gr.Markdown(f"Embeddings via HF Inference API • Model: `{EMBED_MODEL}`")

# Launch (Gradio SDK)
demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
