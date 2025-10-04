# app.py
# Minimal, HF-Spaces-friendly app: FastAPI + Gradio mounted on "/"
# Keep this file named app.py and make sure a global variable `app` exists.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr

# ----------------------------
# 1) FastAPI app (global `app`)
# ----------------------------
api = FastAPI(title="Resume Comparator API", version="1.0")

# (Optional) CORS if you want to call from browsers elsewhere
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

# ----------------------------
# 2) Your API contract
# ----------------------------
class ComparePayload(BaseModel):
    resume_text: str
    job_description: str

def compare_resumes(resume_text: str, jd_text: str) -> dict:
    """
    Replace this stub with your real logic. Keep it pure and fast for CPU Spaces.
    Return a JSON-serializable dict.
    """
    # --- BEGIN: place your real logic here --------------------
    # For now we just compute a very simple overlap score.
    import re
    tokenize = lambda s: set(re.findall(r"[a-zA-Z]+", s.lower()))
    r = tokenize(resume_text)
    j = tokenize(jd_text)
    if not j:
        score = 0.0
    else:
        score = len(r & j) / len(j)

    return {
        "match_score": round(float(score) * 100, 2),
        "matched_terms": sorted(list(r & j))[:50],
        "missing_terms": sorted(list(j - r))[:50],
        "notes": "Replace this with your actual embedding / section-based logic."
    }
    # --- END: place your real logic here ----------------------

@api.post("/api/analyze")
def analyze(payload: ComparePayload):
    result = compare_resumes(payload.resume_text, payload.job_description)
    return result

# ----------------------------
# 3) Gradio UI (mounted on "/")
# ----------------------------
def gradio_compare(resume_text, jd_text):
    result = compare_resumes(resume_text, jd_text)
    pretty = [
        f"Match score: {result['match_score']}%",
        "",
        "Top matched terms:",
        ", ".join(result["matched_terms"]) or "-",
        "",
        "Top missing terms:",
        ", ".join(result["missing_terms"]) or "-",
        "",
        result["notes"]
    ]
    return "\n".join(pretty)

with gr.Blocks() as demo:
    gr.Markdown("# Resume Comparator (demo)\nPaste your resume and JD to see a quick overlap score.")
    with gr.Row():
        resume_in = gr.Textbox(label="Resume text", lines=14, placeholder="Paste resume text here")
        jd_in = gr.Textbox(label="Job description", lines=14, placeholder="Paste job description here")
    out = gr.Textbox(label="Result", lines=14)
    btn = gr.Button("Analyze")
    btn.click(gradio_compare, inputs=[resume_in, jd_in], outputs=[out])
    gr.Markdown(
        "API: `POST /api/analyze` with JSON `{resume_text, job_description}`.\n"
        "Health: `GET /healthz`."
    )

# Mount Gradio onto FastAPI at "/"
app = gr.mount_gradio_app(api, demo, path="/")

# Optional: local run (unused on Spaces, fine to leave)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
