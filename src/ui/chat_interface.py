"""
Interactive chat interface for the finetuned banking regulation model.
Supports both CLI and web-based interfaces.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegulationChatbot:
    """Chatbot for banking regulation queries."""

    def __init__(self, model_path: str, base_model: str = 'microsoft/phi-2'):
        """
        Initialize chatbot.

        Args:
            model_path: Path to finetuned model
            base_model: Base model name
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Using device: {self.device}")

        self.load_model()

    def load_model(self):
        """Load the finetuned model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()

            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_response(self, question: str, max_length: int = 300,
                         temperature: float = 0.7) -> str:
        """
        Generate a response to a question.

        Args:
            question: User question
            max_length: Maximum response length
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        # Format prompt
        system_prompt = """Eres un experto en regulación bancaria española. Responde preguntas sobre regulación bancaria, especialmente sobre parámetros de riesgo de crédito. Siempre cita las fuentes de tu información. Si no estás seguro, di "No tengo información suficiente"."""

        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].strip()
        else:
            response = full_response

        return response

    def chat_cli(self):
        """Run an interactive CLI chat."""
        print("=" * 60)
        print("Spanish Banking Regulation Chatbot")
        print("=" * 60)
        print("Ask questions about Spanish banking regulations.")
        print("Type 'exit' or 'quit' to end the conversation.\n")

        conversation_history = []

        while True:
            # Get user input
            question = input("\nYou: ").strip()

            if question.lower() in ['exit', 'quit', 'salir']:
                print("Goodbye!")
                break

            if not question:
                continue

            # Generate response
            print("\nAssistant: ", end="", flush=True)
            response = self.generate_response(question)
            print(response)

            # Store in history
            conversation_history.append({
                'question': question,
                'response': response
            })

    def create_gradio_interface(self):
        """Create a Gradio web interface."""

        def chat_fn(message, history):
            """Chat function for Gradio."""
            response = self.generate_response(message)
            return response

        # Create interface
        interface = gr.ChatInterface(
            fn=chat_fn,
            title="🏦 Spanish Banking Regulation Chatbot",
            description="""
            Ask questions about Spanish banking regulations, credit risk parameters,
            and compliance requirements. The model has been trained on official
            documents from Bank of Spain, ECB, BOE, CNMV, and Basel Committee.

            **Examples:**
            - ¿Qué es la probabilidad de default (PD)?
            - ¿Qué regulación se aplica al cálculo de capital para riesgo de crédito?
            - Explica el método IRB para carteras retail
            - What are the requirements for calculating LGD?
            """,
            examples=[
                "¿Qué es la probabilidad de default (PD)?",
                "¿Qué regulación se aplica al cálculo de capital para riesgo de crédito?",
                "Explica el método IRB para carteras retail",
                "¿Cuáles son los requisitos para el cálculo de LGD?",
                "What is credit risk in Spanish banking regulation?",
            ],
            #theme=gr.themes.Soft(),
            #retry_btn=None,
            #undo_btn="Delete Previous",
            #clear_btn="Clear",
        )

        return interface


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Banking Regulation Chatbot')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to finetuned model')
    parser.add_argument('--base-model', type=str, default='microsoft/phi-2',
                       help='Base model name')
    parser.add_argument('--interface', type=str, default='web',
                       choices=['web', 'cli'],
                       help='Interface type (web or cli)')
    parser.add_argument('--share', action='store_true',
                       help='Create public link for Gradio interface')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port for web interface')

    args = parser.parse_args()

    # Initialize chatbot
    chatbot = RegulationChatbot(
        model_path=args.model_path,
        base_model=args.base_model
    )

    # Launch interface
    if args.interface == 'cli':
        chatbot.chat_cli()
    else:
        interface = chatbot.create_gradio_interface()
        interface.launch(
            share=args.share,
            server_port=args.port
        )


if __name__ == "__main__":
    main()
