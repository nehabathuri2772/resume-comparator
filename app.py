import os
import gradio as gr
import requests
from products.API_product import run_api_product   # <-- implement
from products.local_product import run_local_product  # <-- implement

# --------- Settings ----------
PORT = int(os.getenv("PORT", "8015"))     # systemd will pass PORT=8015
APP_TITLE = "Resume Comparator — Unified"
# -----------------------------

# Optional: provide a small adapter for the API-based call if your product expects HTTP
def api_runner(resume_file, jd_text):
    """
    Example adapter: if your API product is an HTTP endpoint, call it here.
    Otherwise, call your Python logic directly in products/api_product.py
    """
    # Example direct Python call (preferred, if available):
    return run_api_product(resume_file, jd_text)

    # Example HTTP call instead (uncomment if your API is remote):
    # url = os.getenv("API_URL", "https://example.com/compare")
    # token = os.getenv("API_TOKEN", "")
    # headers = {"Authorization": f"Bearer {token}"} if token else {}
    # files = {"resume": open(resume_file, "rb")}
    # data = {"jd": jd_text}
    # r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    # r.raise_for_status()
    # return r.json().get("result", r.text)

def local_runner(resume_file, jd_text):
    return run_local_product(resume_file, jd_text)

with gr.Blocks(title=APP_TITLE, fill_height=True) as demo:
    gr.Markdown(f"# {APP_TITLE}")

    with gr.Tabs():
        with gr.Tab("API Product"):
            with gr.Row():
                resume_api = gr.File(label="Upload resume (PDF/DOCX/TXT)", file_types=[".pdf", ".docx", ".txt"])
                jd_api = gr.Textbox(label="Job description", lines=16, placeholder="Paste the job description here")
            with gr.Row():
                run_api_btn = gr.Button("Run API Comparison", variant="primary")
                clear_api_btn = gr.Button("Clear")
            result_api = gr.Textbox(label="Result", lines=18)

            run_api_btn.click(api_runner, inputs=[resume_api, jd_api], outputs=result_api)
            clear_api_btn.click(lambda: (None, ""), outputs=[resume_api, jd_api])

        with gr.Tab("Local Product"):
            with gr.Row():
                resume_loc = gr.File(label="Upload resume (PDF/DOCX/TXT)", file_types=[".pdf", ".docx", ".txt"])
                jd_loc = gr.Textbox(label="Job description", lines=16, placeholder="Paste the job description here")
            with gr.Row():
                run_loc_btn = gr.Button("Run Local Comparison", variant="primary")
                clear_loc_btn = gr.Button("Clear")
            result_loc = gr.Textbox(label="Result", lines=18)

            run_loc_btn.click(local_runner, inputs=[resume_loc, jd_loc], outputs=result_loc)
            clear_loc_btn.click(lambda: (None, ""), outputs=[resume_loc, jd_loc])

    gr.Markdown("**One server, one port (8015)** • Choose a tab to run API or Local.")

# Important: one process, one port
if __name__ == "__main__":
    demo.queue()  # good for concurrency
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)
