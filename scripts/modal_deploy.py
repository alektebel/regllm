"""
RegLLM — Modal deployment
Qwen2.5-7B-Instruct + LoRA adapter served via FastAPI + Gradio
GPU: A10G (24GB VRAM) — free tier includes $30/month credits
"""

import modal
import os

# ── Image ────────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "transformers>=4.45.0",
        "peft>=0.12.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.0",
        "gradio>=4.44.0",
        "huggingface_hub>=0.24.0",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App("regllm", image=image)

# Cache model weights in a Modal Volume so they're not re-downloaded every cold start
volume = modal.Volume.from_name("regllm-weights", create_if_missing=True)
VOLUME_PATH = "/vol/models"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "kabesaml/regllm-qwen25-7b-banking-lora"

SYSTEM_PROMPT = (
    "Eres un asistente experto en regulación bancaria y el sector bancario español. "
    "Responde con datos precisos y cita la normativa cuando sea posible. "
    "Si no conoces la respuesta con certeza, dilo claramente."
)

# ── Download weights (run once) ───────────────────────────────────────────────

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    secrets=[],
)
def download_weights():
    from huggingface_hub import snapshot_download
    import os

    hf_token = os.environ.get("HF_TOKEN")

    base_path = f"{VOLUME_PATH}/base"
    adapter_path = f"{VOLUME_PATH}/adapter"

    if not os.path.exists(f"{base_path}/config.json"):
        print(f"Downloading base model: {BASE_MODEL}")
        snapshot_download(
            BASE_MODEL,
            local_dir=base_path,
            token=hf_token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        volume.commit()
        print("Base model downloaded.")
    else:
        print("Base model already cached.")

    if not os.path.exists(f"{adapter_path}/adapter_config.json"):
        print(f"Downloading adapter: {ADAPTER_REPO}")
        snapshot_download(
            ADAPTER_REPO,
            local_dir=adapter_path,
            token=hf_token,
        )
        volume.commit()
        print("Adapter downloaded.")
    else:
        print("Adapter already cached.")


# ── Inference class ───────────────────────────────────────────────────────────

@app.cls(
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=600,
    scaledown_window=300,  # keep warm for 5 min after last request
)
class RegLLM:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_path = f"{VOLUME_PATH}/base"
        adapter_path = f"{VOLUME_PATH}/adapter"

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=True
        )

        print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print("Applying LoRA adapter...")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        print("Model ready.")

    @modal.method()
    def generate(self, messages: list, max_new_tokens: int = 512, temperature: float = 0.3):
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Gradio web UI ─────────────────────────────────────────────────────────────

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="GET")
def ui():
    """Redirect to the Gradio app."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/gradio")


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def gradio_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app
    from fastapi import FastAPI

    fastapi_app = FastAPI()

    def chat(message, history, max_tokens, temperature):
        llm = RegLLM()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, assistant_msg in history:
            msgs.append({"role": "user", "content": user_msg})
            if assistant_msg:
                msgs.append({"role": "assistant", "content": assistant_msg})
        msgs.append({"role": "user", "content": message})

        response = llm.generate.remote(msgs, max_tokens, temperature)
        history = history + [(message, response)]
        return "", history

    with gr.Blocks(title="RegLLM", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align:center;margin-bottom:1rem">
            <h1>🏦 RegLLM</h1>
            <p>Banking regulation assistant · EBA · CRR/CRD · Basel III/IV · Credit Risk</p>
            <p style="font-size:0.85em;color:#666">Qwen2.5-7B-Instruct + LoRA fine-tuned on Spanish banking regulation</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=520)
                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="Ask about CRR, EBA guidelines, Basel, PD/LGD/EAD...",
                        show_label=False, scale=4, lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary")

                gr.Examples(
                    examples=[
                        "What does CRR Article 92 say about capital requirements?",
                        "Explain ICAAP according to EBA guidelines",
                        "What is LGD and how is it estimated under IRB?",
                        "¿Qué define el default según el artículo 178 del CRR?",
                        "Diferencias entre Pilar 1 y Pilar 2 en Basilea",
                    ],
                    inputs=txt,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                max_tokens = gr.Slider(128, 2048, value=512, step=128, label="Max tokens")
                temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
                gr.Markdown("""
                ### Model
                - **Base**: Qwen2.5-7B-Instruct
                - **Adapter**: LoRA (r=32, α=64)
                - **Topics**: EBA, CRR/CRD, Basel, IRB, IFRS 9

                ### Links
                - [Adapter weights](https://huggingface.co/kabesaml/regllm-qwen25-7b-banking-lora)
                """)

        send_btn.click(fn=chat, inputs=[txt, chatbot, max_tokens, temperature], outputs=[txt, chatbot])
        txt.submit(fn=chat, inputs=[txt, chatbot, max_tokens, temperature], outputs=[txt, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, txt])

    return mount_gradio_app(app=fastapi_app, blocks=demo, path="/")


# ── CLI: deploy and download weights ─────────────────────────────────────────

if __name__ == "__main__":
    with app.run():
        download_weights.remote()
