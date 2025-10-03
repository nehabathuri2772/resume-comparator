import os
import io
import re
import tempfile
from typing import List, Tuple, Dict, Any

import numpy as np
import gradio as gr

# ---------- Text + file utils ----------
from pypdf import PdfReader
try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Embedding stacks
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

# ---------- Embedders (lazy / cached) ----------
_SBERT_MODEL_NAME = os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_BERT_MODEL_NAME  = os.environ.get("BERT_MODEL",  "bert-base-uncased")

_Sbert: SentenceTransformer | None = None
_BertTok: AutoTokenizer | None = None
_Bert: AutoModel | None = None


def _load_sbert() -> SentenceTransformer:
    global _Sbert
    if _Sbert is None:
        _Sbert = SentenceTransformer(_SBERT_MODEL_NAME)
    return _Sbert


def _load_bert():
    global _BertTok, _Bert
    if _BertTok is None or _Bert is None:
        _BertTok = AutoTokenizer.from_pretrained(_BERT_MODEL_NAME)
        _Bert    = AutoModel.from_pretrained(_BERT_MODEL_NAME)
        _Bert.eval()
    return _BertTok, _Bert


# ---------- Core helpers ----------
def extract_text_from_fileobj(fileobj) -> str:
    """
    Accepts an object with `.name` (path on disk) or a (name, bytes) tuple.
    Returns extracted text for PDF/DOCX/TXT.
    """
    # Support (filename, bytes) tuples (used by API path)
    if isinstance(fileobj, tuple) and len(fileobj) == 2 and isinstance(fileobj[1], (bytes, bytearray)):
        filename, data = fileobj
        suffix = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".txt") as tmp:
            tmp.write(data)
            tmp.flush()
            path = tmp.name
    else:
        # Gradio File component gives an object with .name path
        path = getattr(fileobj, "name", None)
        if not path:
            return ""

    name = path.lower()
    if name.endswith(".pdf"):
        text_parts = []
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for p in reader.pages:
                try:
                    text_parts.append(p.extract_text() or "")
                except Exception:
                    text_parts.append("")
        return "\n".join(text_parts).strip()

    if (name.endswith(".docx") or name.endswith(".doc")) and HAS_DOCX:
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs).strip()

    # Fallback: treat as text
    with open(path, "rb") as f:
        try:
            return f.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""


def preprocess_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def embed_sbert(texts: List[str]) -> np.ndarray:
    model = _load_sbert()
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs)


@torch.no_grad()
def embed_bert_mean(texts: List[str]) -> np.ndarray:
    tok, mdl = _load_bert()
    # mean-pool last hidden state
    outs: List[np.ndarray] = []
    for t in texts:
        enc = tok(
            t if t else "",
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        out = mdl(**enc)
        last_hidden = out.last_hidden_state  # [1, 512, hidden]
        mask = enc["attention_mask"]         # [1, 512]
        mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean   = (summed / counts).squeeze(0).cpu().numpy()
        outs.append(mean / (np.linalg.norm(mean) + 1e-9))
    return np.vstack(outs)


def embed_texts(texts: List[str], mode: str = "sbert") -> np.ndarray:
    mode = (mode or "sbert").lower()
    if mode == "bert":
        return embed_bert_mean(texts)
    # default
    return embed_sbert(texts)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def top_keywords(text: str, n: int = 20) -> List[str]:
    if not text:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=4096)
    X = vec.fit_transform([text])
    inds = np.argsort(-X.toarray()[0])[:n]
    inv = {v: k for k, v in vec.vocabulary_.items()}
    return [inv[i] for i in inds if i in inv]


def simple_highlights(resume_text: str, job_text: str, top_k: int = 8, mode: str = "sbert") -> List[str]:
    sentences = [s.strip() for s in re.split(r"[.\n]", job_text) if s.strip()]
    if not sentences:
        return []
    rvec = embed_texts([resume_text], mode=mode)[0]
    svecs = embed_texts(sentences, mode=mode)
    sims = cosine_similarity([rvec], svecs)[0]
    idxs = np.argsort(-sims)[:top_k]
    return [f"{sentences[i]}  (sim={sims[i]:.2f})" for i in idxs]


# ---------- Analysis ----------
def analyze_resumes(fileobjs: List[object], job_description: str, mode: str = "sbert"):
    jd = preprocess_text(job_description)
    if not jd:
        raise ValueError("Job description is empty.")

    jd_vec = embed_texts([jd], mode=mode)[0]
    jd_kw  = top_keywords(jd, n=25)

    results = []
    best = None

    for f in fileobjs:
        text = preprocess_text(extract_text_from_fileobj(f))
        if not text:
            results.append({
                "filename": getattr(f, "name", (f[0] if isinstance(f, tuple) else "unknown")),
                "error": "No text extracted"
            })
            continue

        rvec  = embed_texts([text], mode=mode)[0]
        score = cosine(rvec, jd_vec)  # 0..1
        score_pct = round(max(0.0, min(1.0, score)) * 100, 2)

        r_kw = set(top_keywords(text, n=25))
        jd_k = set(jd_kw)
        missing = sorted(list(jd_k - r_kw))[:25]
        overlap = sorted(list(jd_k & r_kw))[:25]

        highlights = simple_highlights(text, jd, top_k=8, mode=mode)

        item = {
            "filename": os.path.basename(getattr(f, "name", (f[0] if isinstance(f, tuple) else "resume"))),
            "chars": len(text),
            "similarity": score_pct,
            "overlap_keywords": overlap,
            "missing_keywords": missing,
            "highlights": highlights,
        }
        results.append(item)
        if (best is None) or (item["similarity"] > best["similarity"]):
            best = item

    verdict_md = ""
    missing_md = ""
    if best:
        verdict_md = f"### Best Match: **{best['filename']}**  \n**Similarity:** {best['similarity']}%"
    # basic missing keywords markdown (for the best one)
    if best and best.get("missing_keywords"):
        missing_md = "### Keywords Missing From Your Resume\n- " + "\n- ".join(best["missing_keywords"])
    else:
        missing_md = "### No obvious missing keywords detected."

    suggestions_text = (
        "- Make sure the top skills from the JD appear in your resume summary.\n"
        "- Add concrete metrics (%, $, time saved) for your relevant projects.\n"
        "- Mirror phrasing from the JD where truthful (helps ATS)."
    )

    # Backward-compatible tuple style (as referenced earlier)
    best_fname = best["filename"] if best else None
    projects_section = ""  # placeholder
    project_fit_verdict = f"Resolved with mode: **{mode}**."

    resume_keywords_text = ", ".join(sorted(set().union(*[set(r.get("overlap_keywords", [])) for r in results])))
    jd_keywords_text = ", ".join(jd_kw)

    return (
        best["similarity"] if best else 0.0,           # score (best)
        verdict_md,                                     # verdict markdown
        missing_md,                                     # missing keywords markdown
        suggestions_text,                               # improvement suggestions
        "",                                             # job_suggestions (unused here)
        projects_section,                               # projects section
        project_fit_verdict,                            # project fit verdict markdown
        resume_keywords_text,                           # resume top keywords (string)
        jd_keywords_text,                               # jd top keywords (string)
        best_fname,                                     # best filename
    )


# ---------- Gradio UI ----------
def build_ui():
    with gr.Blocks(title="Resume Comparator") as demo:
        gr.Markdown("## Resume Comparator — UI + API\nUpload one or more resumes and paste a JD. Use the API at **/api/analyze**.")

        with gr.Row():
            mode = gr.Radio(choices=["sbert", "bert"], value="sbert", label="Embedding mode")
        jd   = gr.Textbox(label="Job Description", lines=10, placeholder="Paste JD here...")
        files = gr.Files(label="Resumes (PDF/DOCX/TXT)", file_count="multiple",
                         file_types=[".pdf", ".docx", ".doc", ".txt"])

        with gr.Row():
            verdict = gr.Markdown(label="Verdict")
            missing = gr.Markdown(label="Missing keywords")
        with gr.Row():
            tips    = gr.Markdown(label="Suggestions")
            summary = gr.JSON(label="Per-resume summary (JSON)")

        def _run(mode_val, jd_text, files_list):
            if not jd_text:
                raise gr.Error("Please paste a job description.")
            fileobjs = files_list or []
            res = analyze_resumes(fileobjs, jd_text, mode=mode_val)
            score, verdict_md, missing_md, suggestions_text, _, _, project_fit, resume_kw, jd_kw, best_name = res

            per_resume = []
            # re-run lightweight loop to show per-resume JSON
            for f in fileobjs:
                fname = os.path.basename(getattr(f, "name", "resume"))
                txt = preprocess_text(extract_text_from_fileobj(f))
                if not txt:
                    per_resume.append({"filename": fname, "error": "No text"})
                    continue
                sim = cosine(embed_texts([txt], mode=mode_val)[0], embed_texts([jd_text], mode=mode_val)[0])
                per_resume.append({"filename": fname, "similarity": round(float(sim)*100, 2)})

            verdict_out = verdict_md + f"\n\n{project_fit}"
            return verdict_out, missing_md, suggestions_text, per_resume

        gr.Button("Analyze").click(
            _run, inputs=[mode, jd, files],
            outputs=[verdict, missing, tips, summary]
        )

    return demo


# ---------- FASTAPI: JSON API + mount UI ----------
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

api = FastAPI(title="Resume Comparator API", version="1.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def _uploadfiles_to_temp(files: List[UploadFile]) -> List[object]:
    """Convert FastAPI UploadFile list → objects with .name on disk."""
    out = []
    for uf in files:
        data = uf.file.read()
        suffix = os.path.splitext(uf.filename or "file")[1] or ".txt"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        class _Shim: pass
        s = _Shim(); s.name = tmp.name
        out.append(s)
    return out

@api.post("/api/analyze")
def analyze_endpoint(
    job_description: str = Form(...),
    mode: str = Form("sbert"),
    files: List[UploadFile] = File(...),
):
    fileobjs = _uploadfiles_to_temp(files)
    try:
        (
            score, verdict, missing_formatted, suggestions_text,
            job_suggestions, projects_section, project_fit_verdict,
            resume_keywords_text, jd_keywords_text, best_fname
        ) = analyze_resumes(fileobjs, job_description, mode)
    finally:
        for fo in fileobjs:
            try:
                os.unlink(fo.name)
            except Exception:
                pass

    return {
        "best_match_filename": best_fname,
        "similarity_score": score,
        "verdict_markdown": verdict,
        "missing_keywords_markdown": missing_formatted,
        "improvement_suggestions": suggestions_text,
        "project_fit_verdict_markdown": project_fit_verdict,
        "projects_section_text": projects_section,
        "resume_top_keywords": resume_keywords_text,
        "job_top_keywords": jd_keywords_text,
        "mode_used": mode,
    }

@api.get("/api/health")
def health():
    return {"status": "ok"}

# Mount the Gradio UI at root
demo = build_ui()
app = gr.mount_gradio_app(api, demo, path="/")
