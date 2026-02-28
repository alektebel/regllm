import os
import modal
from pathlib import Path

# Local paths
local_dir = Path(__file__).parent.parent
root_path = local_dir / "models/finetuned"
adapter_paths = sorted(
    [
        p
        for p in root_path.glob("run_*/final_model")
        if (p / "adapter_model.safetensors").exists()
    ],
    key=lambda p: p.stat().st_mtime,
)
if not adapter_paths:
    adapter_paths = sorted(
        [
            p
            for p in root_path.glob("run_*/checkpoint-*")
            if (p / "adapter_model.safetensors").exists()
        ],
        key=lambda p: p.stat().st_mtime,
    )

if not adapter_paths:
    raise FileNotFoundError("No trained adapter found. Run training first.")

LATEST_ADAPTER = adapter_paths[-1]
print(f"Using adapter: {LATEST_ADAPTER.relative_to(local_dir)}")

# Define the Modal App
# In Modal 1.x, we use Image.add_local_dir to include local files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "gradio",
        "chromadb",
        "sentence-transformers",
        "rank-bm25",
        "langdetect",
        "pydantic",
        "python-dotenv",
        "hf_transfer",
        "scikit-learn",
    )
    .add_local_dir(local_dir / "src", remote_path="/root/src")
    .add_local_dir(local_dir / "vector_db", remote_path="/root/vector_db")
    .add_local_dir(LATEST_ADAPTER, remote_path="/root/adapter")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("regllm-app", image=image)


@app.cls(
    gpu="A10G",
    container_idle_timeout=300,
)
class RegLLM:
    @modal.enter()
    def setup(self):
        import sys

        sys.path.append("/root")

        from src.rag_system import RegulatoryRAGSystem
        from src.citation_rag import CitationRAG
        from src.chat_engine import ChatEngine
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print("Initializing RAG systems...")
        self.rag_system = RegulatoryRAGSystem(
            persist_directory="/root/vector_db/chroma_db"
        )

        try:
            self.citation_rag = CitationRAG(chroma_path="/root/vector_db/chroma_db")
        except Exception as e:
            print(f"Citation RAG unavailable: {e}")
            self.citation_rag = None

        print("Loading Model...")
        BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/root/adapter", trust_remote_code=True
        )

        # QLoRA loading (4-bit) to save VRAM on A10G
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, "/root/adapter")
        self.model.eval()

        self.engine = ChatEngine(
            self.rag_system, citation_rag=self.citation_rag, db_source="modal"
        )
        print("Ready.")

    @modal.method()
    def generate(self, question, history, n_sources=5, hybrid=True):
        import torch

        # Topic guard
        if not self.engine.check_topic(question):
            from src.chat_engine import REJECTION_CARD

            return REJECTION_CARD

        # Build context and messages
        context, _ = self.engine.build_context(
            question, n_sources=n_sources, hybrid=hybrid
        )
        messages = self.engine.build_messages(question, context, history)

        # Inference
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_response = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        # Parse and enrich
        parsed = self.engine.parse_response(raw_response)
        parsed = self.engine.enrich_references(parsed, question)

        return self.engine.render_message(parsed)


@app.function(allow_concurrent_inputs=10)
@modal.asgi_app()
def ui():
    import gradio as gr
    from gradio.routes import mount_gradio_app
    from fastapi import FastAPI
    from src.chat_engine import CSS, EXAMPLES

    web_app = FastAPI()
    llm = RegLLM()

    def predict(message, history):
        formatted_history = []
        for h in history:
            if h[0]:
                formatted_history.append({"role": "user", "content": h[0]})
            if h[1]:
                formatted_history.append({"role": "assistant", "content": h[1]})

        response = llm.generate.remote(message, formatted_history)
        history.append((message, response))
        return "", history

    with gr.Blocks(title="RegLLM — Modal", css=CSS) as demo:
        gr.HTML(f"""
        <div class="regllm-header">
            <div style="flex:1">
                <h1>RegLLM — Modal</h1>
                <p class="subtitle">Asistente de regulación bancaria · EBA · CRR/CRD · Basilea III/IV</p>
            </div>
            <span class="badge">A10G GPU</span>
        </div>
        """)

        chatbot = gr.Chatbot(height=550)
        with gr.Row():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Escribe tu pregunta sobre regulación bancaria...",
                scale=4,
            )
            btn = gr.Button("Enviar", variant="primary", scale=1)

        gr.Examples(examples=EXAMPLES, inputs=txt)

        txt.submit(predict, [txt, chatbot], [txt, chatbot])
        btn.click(predict, [txt, chatbot], [txt, chatbot])

    return mount_gradio_app(app=web_app, blocks=demo, path="/")
