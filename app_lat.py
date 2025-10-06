import os
import gradio as gr

# One-port friendly settings (good for systemd & your VM)
os.environ.setdefault("GRADIO_USE_MULTIPLE_PORTS", "0")
os.environ.setdefault("GRADIO_USE_CDN", "0")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

# Import the two product entrypoints (these must exist)
from products.api_product import run_api_product      # <- note products.
from products.local_product import run_local_product  # <- note products.

APP_TITLE = "Resume Comparator — Unified (API + Local)"
PORT = int(os.getenv("PORT", "8015"))

def _clear():
    # Return values must match the outputs list order
    return None, ""  # clears file, clears textbox

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")

    with gr.Tabs():
        # ------------- Tab 1: API (HF Inference) -------------
        with gr.TabItem("API (HF Inference)"):
            with gr.Row():
                api_resume = gr.File(
                    label="Upload resume (PDF/DOCX/TXT)",
                    file_types=[".pdf", ".docx", ".txt"],
                    file_count="single",
                )
                api_jd = gr.Textbox(
                    label="Job description (text)",
                    lines=16,
                    placeholder="Paste the job description here"
                )

            with gr.Row():
                api_go = gr.Button("Analyze via API", variant="primary")
                api_clear = gr.Button("Clear")

            api_result = gr.Textbox(label="Result", lines=18)
            api_go.click(run_api_product, inputs=[api_resume, api_jd], outputs=[api_result])
            api_clear.click(_clear, outputs=[api_resume, api_jd])

        # ------------- Tab 2: Local (on-VM) -------------
        with gr.TabItem("Local (On-VM)"):
            with gr.Row():
                loc_resume = gr.File(
                    label="Upload resume (PDF/DOCX/TXT)",
                    file_types=[".pdf", ".docx", ".txt"],
                    file_count="single",
                )
                loc_jd = gr.Textbox(
                    label="Job description (text)",
                    lines=16,
                    placeholder="Paste the job description here"
                )

            with gr.Row():
                loc_go = gr.Button("Analyze locally", variant="primary")
                loc_clear = gr.Button("Clear")

            loc_result = gr.Textbox(label="Result", lines=18)
            loc_go.click(run_local_product, inputs=[loc_resume, loc_jd], outputs=[loc_result])
            loc_clear.click(_clear, outputs=[loc_resume, loc_jd])

    gr.Markdown(
        "Tip: the **API** tab calls Hugging Face Inference API (needs `HF_TOKEN`). "
        "The **Local** tab runs entirely on the VM."
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT, share=False)
