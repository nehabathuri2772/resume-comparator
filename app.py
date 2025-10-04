import os, re
import numpy as np
import gradio as gr
from huggingface_hub import InferenceClient

# -------- Space/Gradio config (single port; no CDN) --------
os.environ["GRADIO_USE_MULTIPLE_PORTS"] = "0"
os.environ.setdefault("GRADIO_USE_CDN", "0")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

# -------- App config --------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN    = os.getenv("HF_TOKEN")                      # set in Space → Settings → Variables
MAX_LEN     = int(os.getenv("MAX_TEXT_LEN", "20000"))
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", "40"))

# HF Inference API client (timeout belongs on the client)
hf = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN, timeout=REQ_TIMEOUT)

# -------- Utilities --------
def _trim(s: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= MAX_LEN else s[:MAX_LEN]

def _tok_words(s: str) -> set:
    return set(re.findall(r"[a-zA-Z]+", (s or "").lower()))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _embed(text: str) -> np.ndarray:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set (add it in Space → Settings → Variables).")
    feats = hf.feature_extraction(_trim(text))           # no timeout kwarg here
    arr = np.array(feats, dtype=np.float32)
    if arr.ndim == 2:                                    # token-level → mean pool
        arr = arr.mean(axis=0)
    if arr.ndim != 1:
        raise RuntimeError("Unexpected embedding shape from the Inference API.")
    return arr

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
    import docx
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
    # best-effort fallback as text
    return _read_txt(path)

# -------- Core scoring --------
def compare_resume_to_jd(resume_text: str, jd_text: str) -> str:
    resume_text = _trim(resume_text)
    jd_text     = _trim(jd_text)
    if not resume_text or not jd_text:
        raise ValueError("Please upload a resume file and provide a job description.")

    r_vec = _embed(resume_text)
    j_vec = _embed(jd_text)
    score = _cosine(r_vec, j_vec) * 100.0

    r, j = _tok_words(resume_text), _tok_words(jd_text)
    matched = sorted(list(r & j))[:50]
    missing = sorted(list(j - r))[:50]

    tier = "Excellent" if score >= 80 else "Good" if score >= 65 else "Fair" if score >= 50 else "Low"

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

# -------- Gradio handlers --------
def run(resume_file, jd_text):
    try:
        resume_txt = read_resume_file(resume_file)
        return compare_resume_to_jd(resume_txt, jd_text)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

# -------- UI (file + JD only; text output) --------
with gr.Blocks() as demo:
    gr.Markdown("# Resume Comparator")

    with gr.Row():
        resume_file = gr.File(
            label="Upload resume (PDF/DOCX/TXT)",
            file_types=[".pdf", ".docx", ".txt"],
            file_count="single",
        )
        jd_text = gr.Textbox(
            label="Job description (text)",
            lines=16,
            placeholder="Paste the job description here"
        )

    with gr.Row():
        go = gr.Button("Analyze", variant="primary")
        clear = gr.Button("Clear")

    result = gr.Textbox(label="Result", lines=18)

    go.click(run, inputs=[resume_file, jd_text], outputs=[result])

    def _clear():
        return None, ""
    clear.click(_clear, outputs=[resume_file, jd_text])
    
    gr.Markdown(f"Embeddings via HF Inference API • Model: `{EMBED_MODEL}`")

# Launch (Gradio SDK)
demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
