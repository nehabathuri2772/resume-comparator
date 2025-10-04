# app.py — Gradio-only Space using HF Inference API for embeddings
# Works with Space SDK = Gradio. Provides UI + POST /api/predict.

import os
import re
import time
import json
import random
from functools import lru_cache

import numpy as np
import gradio as gr
from huggingface_hub import InferenceClient, InferenceTimeoutError

# ---- Single-port assets (avoid 7861 / CSS 404s) ----
os.environ["GRADIO_USE_MULTIPLE_PORTS"] = "0"
os.environ.setdefault("GRADIO_USE_CDN", "0")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

# ---- Config ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN")  # set this in Space → Settings → Variables
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "20000"))
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", "40"))

hf = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN)

# ---- Helpers ----
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _tok_words(s: str) -> set:
    return set(re.findall(r"[a-zA-Z]+", (s or "").lower()))

def _trim(s: str, n: int = MAX_TEXT_LEN) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

def _jx(attempt: int):
    # jittered exponential backoff
    delay = min(1.5 ** attempt, 8.0) + random.random()
    time.sleep(delay)

@lru_cache(maxsize=256)
def _embed_cached(text: str):
    # returns (ok, vec_or_err)
    try:
        feats = hf.feature_extraction(text, timeout=REQ_TIMEOUT)  # list[list[float]] or list[float]
    except InferenceTimeoutError:
        return False, f"Inference API timed out after {REQ_TIMEOUT}s."
    except Exception as e:
        return False, f"Inference API error: {type(e).__name__}: {e}"

    arr = np.array(feats, dtype=np.float32)
    if arr.ndim == 2:  # token-level → mean pool
        arr = arr.mean(axis=0)
    if arr.ndim != 1:
        return False, "Unexpected embedding shape from the Inference API."
    return True, arr

def _embed_with_retry(text: str, retries: int = 3) -> np.ndarray:
    text = _trim(text)
    last_err = None
    for attempt in range(1, retries + 1):
        ok, val = _embed_cached(text)
        if ok:
            return val
        last_err = val
        _jx(attempt)
    raise RuntimeError(last_err or "Unknown Inference API error.")

def compare_resumes(resume_text: str, jd_text: str) -> dict:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Add it in Space → Settings → Variables → HF_TOKEN.")

    resume_text = _trim(resume_text)
    jd_text = _trim(jd_text)
    if not resume_text or not jd_text:
        raise ValueError("Both resume text and job description are required.")

    r_vec = _embed_with_retry(resume_text)
    j_vec = _embed_with_retry(jd_text)
    score = _cosine(r_vec, j_vec) * 100.0

    r, j = _tok_words(resume_text), _tok_words(jd_text)
    matched = sorted(list(r & j))
    missing = sorted(list(j - r))

    return {
        "model": EMBED_MODEL,
        "match_score": round(score, 2),
        "matched_terms": matched[:100],
        "missing_terms": missing[:100],
        "resume_len": len(resume_text),
        "jd_len": len(jd_text),
        "notes": "Embeddings computed via Hugging Face Inference API (remote).",
    }

# ---- Gradio functions ----
def _chips(items, title):
    if not items:
        return f"<div class='chip-title'>{title}</div><div>-</div>"
    chips = "".join(f"<span class='chip'>{gr.utils.sanitize_html(x)}</span>" for x in items)
    return f"<div class='chip-title'>{title}</div><div class='chip-wrap'>{chips}</div>"

def gradio_compare(resume_text, jd_text, progress=gr.Progress(track_tqdm=False)):
    progress(0, desc="Embedding")
    try:
        result = compare_resumes(resume_text, jd_text)
    except Exception as e:
        err = f"❌ {e}"
        return "", "", err, gr.update(visible=True, value=json.dumps({"error": str(e)}, indent=2))

    progress(0.8, desc="Rendering")
    score = result["match_score"]
    tier = ("Excellent" if score >= 80 else "Good" if score >= 65 else "Fair" if score >= 50 else "Low")

    score_md = f"""
### ✅ Match Score
**{score:.2f}%**  &nbsp; <span class="badge">{tier}</span>  
_Model: `{result['model']}` • Resume chars: {result['resume_len']} • JD chars: {result['jd_len']}_
"""
    chips_html = f"""
<div class="chips">
  {_chips(result["matched_terms"][:40], "Top matched terms")}
  {_chips(result["missing_terms"][:40], "Top missing terms")}
</div>
"""
    notes_md = f"> {result['notes']}"
    json_blob = json.dumps(result, indent=2)
    progress(1.0, desc="Done")

    return score_md, chips_html, notes_md, gr.update(visible=True, value=json_blob)

# ---- Example texts ----
EXAMPLE_RESUME = """Experienced ETL Developer skilled in SQL, Python, and data modeling. Built cloud pipelines on AWS (S3, Glue, Redshift).
Designed CDC patterns, unit tests, and CI/CD with GitHub Actions. Collaborated with stakeholders to define SLAs and
data quality metrics using Great Expectations. Mentored juniors and documented playbooks.
"""

EXAMPLE_JD = """Seeking a Senior ETL Developer to design and maintain cloud-based data pipelines. Must have strong SQL, data modeling,
and experience with AWS (S3, Glue, Redshift) or Azure equivalents. Bonus: Python, CI/CD, testing frameworks, and
observability (Great Expectations). Collaborate cross-functionally and uphold data SLAs.
"""

# ---- Gradio UI ----
with gr.Blocks(css="""
.badge { padding: 4px 10px; border-radius: 9999px; background: var(--secondary-100); font-size: 0.85rem; }
.chips { display:flex; gap: 28px; flex-wrap: wrap; }
.chip-title { font-weight: 600; margin-bottom: 6px; }
.chip-wrap { display:flex; gap: 8px; flex-wrap: wrap; max-width: 1100px; }
.chip { padding: 6px 10px; border-radius: 999px; border: 1px solid var(--border-color-primary); font-size: 0.9rem; }
.warn { color: #a66; font-weight: 600; }
.footer { opacity: 0.8; }
""") as demo:
    gr.Markdown(
        """
# Resume Comparator
Paste your resume and a job description to get a quick overlap score powered by the **Hugging Face Inference API**.
        """
    )
    if not HF_TOKEN:
        gr.Markdown("**⚠️ Warning:** `HF_TOKEN` is not set. Set it in Space → **Settings → Variables** to enable scoring.", elem_classes=["warn"])

    with gr.Row():
        resume_in = gr.Textbox(label="Resume text", lines=16, value=EXAMPLE_RESUME)
        jd_in = gr.Textbox(label="Job description", lines=16, value=EXAMPLE_JD)

    with gr.Row():
        analyze_btn = gr.Button("Analyze", variant="primary")
        clear_btn = gr.Button("Clear")

    score_card = gr.Markdown("")
    chips_html = gr.HTML("")
    notes_md = gr.Markdown("")
    with gr.Accordion("Raw JSON", open=False):
        json_out = gr.Code(label="Result JSON", language="json", value="", visible=False)

    analyze_btn.click(
        gradio_compare,
        inputs=[resume_in, jd_in],
        outputs=[score_card, chips_html, notes_md, json_out],
    )

    def _clear():
        return "", "", "", gr.update(visible=False, value="")
    clear_btn.click(_clear, outputs=[score_card, chips_html, notes_md, json_out])

    with gr.Accordion("API (auto-exposed by Gradio)", open=False):
        gr.Markdown(
            "")
**Endpoint (Gradio):**
