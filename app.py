# app.py — Gradio Space that uses HF Inference API for embeddings
# Inputs: resume file (PDF/DOCX/TXT) or pasted resume text, plus JD text
# Output: match score + matched/missing terms (also available via REST: POST /api/predict)

import os, re, json, time, random
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import gradio as gr
from huggingface_hub import InferenceClient

# Single-port assets (prevents "7861" CSS/JS failures)
os.environ["GRADIO_USE_MULTIPLE_PORTS"] = "0"
os.environ.setdefault("GRADIO_USE_CDN", "0")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

# ---------------- Config ----------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN")  # set in Space → Settings → Variables
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "20000"))
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", "40"))

# Build the Inference API client (timeout belongs here, not on method calls)
hf = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN, timeout=REQ_TIMEOUT)

# -------------- Utilities ---------------
def _trim(s: str, n: int = MAX_TEXT_LEN) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

def _tok_words(s: str) -> set:
    # alpha tokens; simple + robust for overlap
    return set(re.findall(r"[a-zA-Z]+", (s or "").lower()))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _backoff(attempt: int):
    time.sleep(min(1.5 ** attempt, 8.0) + random.random())

@lru_cache(maxsize=256)
def _embed_cached(text: str) -> Tuple[bool, object]:
    try:
        feats = hf.feature_extraction(text)  # no timeout kwarg here
        arr = np.array(feats, dtype=np.float32)
        if arr.ndim == 2:  # token-level → mean pool
            arr = arr.mean(axis=0)
        if arr.ndim != 1:
            return False, "Unexpected embedding shape from the Inference API."
        return True, arr
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _embed(text: str, retries: int = 2) -> np.ndarray:
    text = _trim(text)
    if not text:
        raise ValueError("Empty text given to embed()")
    last = None
    for a in range(1, retries + 2):
        ok, val = _embed_cached(text)
        if ok:
            return val  # np.ndarray
        last = val
        _backoff(a)
    raise RuntimeError(last or "Unknown Inference API error.")

# ---------- Resume file parsing ----------
def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        text = []
        r = PdfReader(path)
        for page in r.pages:
            txt = page.extract_text() or ""
            text.append(txt)
        return "\n".join(text)
    except Exception as e:
        raise RuntimeError(f"PDF parse failed: {e}")

def _read_docx(path: str) -> str:
    try:
        import docx  # python-docx
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        raise RuntimeError(f"DOCX parse failed: {e}")

def read_resume_file(file_obj) -> str:
    """Gradio File returns a TempFile object with .name (path) and .orig_name."""
    if not file_obj:
        return ""
    path = getattr(file_obj, "name", None) or ""
    name = (getattr(file_obj, "orig_name", "") or path).lower()
    if not path:
        return ""

    if name.endswith(".pdf"):
        return _read_pdf(path)
    if name.endswith(".docx"):
        return _read_docx(path)
    if name.endswith(".txt"):
        return _read_txt(path)
    # Fallback: try reading as text
    try:
        return _read_txt(path)
    except Exception:
        raise RuntimeError("Unsupported file type. Please upload PDF/DOCX/TXT.")

# -------------- Core logic --------------
def compare_resumes(resume_text: str, jd_text: str) -> dict:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Add it in Space → Settings → Variables → HF_TOKEN.")

    resume_text = _trim(resume_text)
    jd_text = _trim(jd_text)
    if not resume_text or not jd_text:
        raise ValueError("Please provide both resume content and job description.")

    r_vec = _embed(resume_text)
    j_vec = _embed(jd_text)
    score = _cosine(r_vec, j_vec) * 100.0

    r, j = _tok_words(resume_text), _tok_words(jd_text)
    matched = sorted(list(r & j))[:100]
    missing = sorted(list(j - r))[:100]

    return {
        "model": EMBED_MODEL,
        "match_score": round(score, 2),
        "matched_terms": matched,
        "missing_terms": missing,
        "resume_len": len(resume_text),
        "jd_len": len(jd_text),
        "notes": "Embeddings via Hugging Face Inference API.",
    }

# -------------- Gradio handler --------------
def run_analysis(resume_file, resume_text, jd_text):
    try:
        # Prefer file → otherwise use pasted resume_text
        file_text = read_resume_file(resume_file) if resume_file else ""
        effective_resume = file_text or resume_text or ""
        result = compare_resumes(effective_resume, jd_text)

        # Pretty outputs
        score = result["match_score"]
        tier = "Excellent" if score >= 80 else "Good" if score >= 65 else "Fair" if score >= 50 else "Low"
        score_md = (
            f"### ✅ Match Score\n**{score:.2f}%**  &nbsp; <span class='badge'>{tier}</span>\n"
            f"_Model: `{result['model']}` • Resume chars: {result['resume_len']} • JD chars: {result['jd_len']}_"
        )
        chips_html = (
            "<div class='chip-title'>Top matched terms</div><div class='chip-wrap'>"
            + "".join(f"<span class='chip'>{gr.utils.sanitize_html(x)}</span>" for x in result["matched_terms"][:40])
            + "</div><br/><div class='chip-title'>Top missing terms</div><div class='chip-wrap'>"
            + "".join(f"<span class='chip'>{gr.utils.sanitize_html(x)}</span>" for x in result["missing_terms"][:40])
            + "</div>"
        )
        notes_md = f"> {result['notes']}"
        return score_md, chips_html, notes_md, json.dumps(result, indent=2), ""
    except Exception as e:
        # Show concise error
        return "", "", "", json.dumps({"error": str(e)}, indent=2), f"❌ {type(e).__name__}: {e}"

# ------------------- UI -------------------
with gr.Blocks(css="""
.badge{padding:4px 10px;border-radius:9999px;background:var(--secondary-100);font-size:.85rem}
.chip-title{font-weight:600;margin:12px 0 6px}
.chip-wrap{display:flex;gap:8px;flex-wrap:wrap;max-width:1100px}
.chip{padding:6px 10px;border-radius:999px;border:1px solid var(--border-color-primary);font-size:.9rem}
.err{color:#b44;font-weight:600}
""") as demo:
    gr.Markdown("# Resume Comparator")

    with gr.Row():
        with gr.Column():
            resume_file = gr.File(label="Upload resume (PDF/DOCX/TXT)", file_types=[".pdf",".docx",".txt"], file_count="single")
            resume_text = gr.Textbox(label="OR paste resume text", lines=10, placeholder="Leave empty if you uploaded a file.")
        jd_text = gr.Textbox(label="Job description (text)", lines=14)

    with gr.Row():
        analyze_btn = gr.Button("Analyze", variant="primary")
        clear_btn = gr.Button("Clear")

    score_md = gr.Markdown("")
    chips_html = gr.HTML("")
    notes_md = gr.Markdown("")
    with gr.Accordion("Raw JSON", open=False):
        json_out = gr.Code(label="Result JSON", language="json", value="")
    error_md = gr.Markdown("", elem_classes=["err"])

    analyze_btn.click(
        run_analysis,
        inputs=[resume_file, resume_text, jd_text],
        outputs=[score_md, chips_html, notes_md, json_out, error_md],
    )

    def _clear():
        return None, "", "", "", "", ""
    clear_btn.click(_clear, outputs=[resume_file, resume_text, jd_text, score_md, json_out, error_md])

    gr.Markdown(f"Embeddings: `{EMBED_MODEL}` via HF Inference API")

# Spaces (Gradio SDK) needs launch here
demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
