# app.py
# FastAPI + Gradio on one port (HF Spaces friendly). Fancy UI, same /api/analyze.

import os
import re
import time
import json
import math
import random
from functools import lru_cache

import numpy as np
import gradio as gr
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient, InferenceTimeoutError

# ----------------------------
# Force single-port, local assets (prevents 7861 / CSS 404s)
# ----------------------------
os.environ["GRADIO_USE_MULTIPLE_PORTS"] = "0"
os.environ.setdefault("GRADIO_USE_CDN", "0")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_SERVER_PORT", os.getenv("PORT", "7860"))

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN")  # Set in HF Space Settings → Variables
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "20000"))  # hard cap to be nice to API
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", "40"))     # seconds

if not HF_TOKEN:
    # We won't crash; the UI will show a warning and API will raise a 401-ish message if called.
    print("⚠️  HF_TOKEN is not set. Set it in your Space (Settings → Variables) for remote embeddings.")

hf = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN)

# ----------------------------
# FastAPI app (global `app`)
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
    return {"status": "ok", "model": EMBED_MODEL}

class ComparePayload(BaseModel):
    resume_text: str
    job_description: str

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _tokenize_words(s: str) -> set:
    # Simple alphas-only tokens; tweak if you want to preserve hyphens, etc.
    return set(re.findall(r"[a-zA-Z]+", s.lower()))

def _trim(s: str, max_len: int = MAX_TEXT_LEN) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= max_len else s[:max_len]

def _backoff_sleep(attempt: int):
    # jittered exponential backoff (attempt starts at 1)
    base = min(1.5 ** attempt, 8.0)
    time.sleep(base + random.random())

@lru_cache(maxsize=256)
def _embed_text_cached(text: str) -> tuple:
    """
    Cached embedding; returns (ok, vector or error_message).
    We cache by exact text string; if you want to loosen, hash/normalize.
    """
    try:
        feats = hf.feature_extraction(text, timeout=REQ_TIMEOUT)  # list[list[float]] or list[float]
    except InferenceTimeoutError:
        return (False, f"Inference API timed out after {REQ_TIMEOUT}s.")
    except Exception as e:
        return (False, f"Inference API error: {type(e).__name__}: {e}")

    arr = np.array(feats, dtype=np.float32)
    if arr.ndim == 2:  # token embeddings → mean-pool
        arr = arr.mean(axis=0)
    if arr.ndim != 1:
        return (False, "Unexpected embedding shape from the Inference API.")
    return (True, arr)

def _embed_remote_with_retry(text: str, retries: int = 3):
    text = _trim(text)
    last_err = None
    for attempt in range(1, retries + 1):
        ok, val = _embed_text_cached(text)
        if ok:
            return val
        last_err = val
        _backoff_sleep(attempt)
    raise RuntimeError(last_err or "Unknown embedding error.")

def compare_resumes(resume_text: str, jd_text: str) -> dict:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not configured. Set it in your Space (Settings → Variables) and restart."
        )

    resume_text = _trim(resume_text)
    jd_text = _trim(jd_text)

    if not resume_text or not jd_text:
        raise ValueError("Both resume text and job description are required.")

    r_vec = _embed_remote_with_retry(resume_text)
    j_vec = _embed_remote_with_retry(jd_text)
    score = _cosine(r_vec, j_vec) * 100.0

    r, j = _tokenize_words(resume_text), _tokenize_words(jd_text)
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

@api.post("/api/analyze")
def analyze(payload: ComparePayload):
    try:
        result = compare_resumes(payload.resume_text, payload.job_description)
        return result
    except Exception as e:
        return {"error": str(e)}

# ----------------------------
# Gradio UI
# ----------------------------
def _chips(items, title):
    if not items:
        return f"<div class='chip-title'>{title}</div><div>-</div>"
    chips = "".join(f"<span class='chip'>{gr.utils.sanitize_html(x)}</span>" for x in items)
    return f"<div class='chip-title'>{title}</div><div class='chip-wrap'>{chips}</div>"

def gradio_compare(resume_text, jd_text, progress=gr.Progress(track_tqdm=False)):
    progress(0, desc="Embedding resume")
    try:
        result = compare_resumes(resume_text, jd_text)
    except Exception as e:
        return (
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=f"❌ {e}"),
            gr.update(visible=True, value=json.dumps({"error": str(e)}, indent=2)),
        )

    progress(0.8, desc="Rendering results")

    score = result["match_score"]
    # Simple letter-ish tier, adjustable thresholds
    tier = (
        "Excellent" if score >= 80
        else "Good" if score >= 65
        else "Fair" if score >= 50
        else "Low"
    )

    card_md = f"""
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
    return (
        card_md,
        chips_html,
        notes_md,
        gr.update(visible=True, value=json_blob),
    )

# Nice defaults/example texts
EXAMPLE_RESUME = """\
Experienced ETL Developer skilled in SQL, Python, and data modeling. Built cloud pipelines on AWS (S3, Glue, Redshift).
Designed CDC patterns, unit tests, and CI/CD with GitHub Actions. Collaborated with stakeholders to define SLAs and
data quality metrics using Great Expectations. Mentored juniors and documented playbooks.
"""

EXAMPLE_JD = """\
Seeking a Senior ETL Developer to design and maintain cloud-based data pipelines. Must have strong SQL, data modeling,
and experience with AWS (S3, Glue, Redshift) or Azure equivalents. Bonus: Python, CI/CD, testing frameworks, and
observability (Great Expectations). Collaborate cross-functionally and uphold data SLAs.
"""

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

- **Private by default**: your texts are only embedded via the configured model.
- **Tip**: Higher score means closer overall similarity; still scan the **missing terms** to tailor your resume.
        """
    )

    if not HF_TOKEN:
        gr.Markdown("**⚠️ Warning:** `HF_TOKEN` is not set. Set it in Space → **Settings → Variables** to enable scoring.", elem_classes=["warn"])

    with gr.Tabs():
        with gr.TabItem("Analyze"):
            with gr.Row():
                resume_in = gr.Textbox(
                    label="Resume text",
                    lines=16,
                    placeholder="Paste resume text here",
                    value=EXAMPLE_RESUME
                )
                jd_in = gr.Textbox(
                    label="Job description",
                    lines=16,
                    placeholder="Paste job description here",
                    value=EXAMPLE_JD
                )
            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.Button("Clear")

            score_card = gr.Markdown(label="Score", value="")
            chips = gr.HTML("")
            notes = gr.Markdown("")
            with gr.Accordion("Raw JSON", open=False):
                json_out = gr.Code(label="Result JSON", language="json", value="", visible=False)

            analyze_btn.click(
                gradio_compare,
                inputs=[resume_in, jd_in],
                outputs=[score_card, chips, notes, json_out]
            )

            def _clear():
                return "", "", "", gr.update(visible=False, value="")
            clear_btn.click(
                _clear,
                outputs=[score_card, chips, notes, json_out]
            )

        with gr.TabItem("API"):
            gr.Markdown(
                """
**Endpoint**

