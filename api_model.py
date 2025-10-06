# API_model.py
import os
import numpy as np
from huggingface_hub import InferenceClient

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN    = os.getenv("HF_TOKEN")                      # set in env/Space Variables
MAX_LEN     = int(os.getenv("MAX_TEXT_LEN", "20000"))
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", "40"))

_hf = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN, timeout=REQ_TIMEOUT)

def _trim(s: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= MAX_LEN else s[:MAX_LEN]

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _embed_api(text: str) -> np.ndarray:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set (add it in environment or Space → Settings → Variables).")
    feats = _hf.feature_extraction(_trim(text))  # may raise on auth/model/timeouts
    arr = np.array(feats, dtype=np.float32)
    if arr.ndim == 2:  # token-level → mean pool
        arr = arr.mean(axis=0)
    if arr.ndim != 1:
        raise RuntimeError(f"Unexpected embedding shape from the Inference API: {arr.shape}")
    return arr

def calculate_similarity_api(text_a: str, text_b: str) -> float:
    a_vec = _embed_api(text_a)
    b_vec = _embed_api(text_b)
    score = _cosine(a_vec, b_vec) * 100.0
    return float(np.round(score, 2))

def check_api_health() -> tuple[bool, str]:
    """Quick probe to fail fast with a helpful message in the UI."""
    try:
        _ = _embed_api("healthcheck")
        return True, f"OK (model={EMBED_MODEL})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
