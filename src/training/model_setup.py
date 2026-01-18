"""
Model setup for Ollama 3B or similar small language models.
Downloads and prepares the model for finetuning.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from pathlib import Path
import sys

# Add project root to path for config import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSetup:
    """Setup and configure the base model for finetuning."""

    # Available models for finetuning
    AVAILABLE_MODELS = {
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',          # 7B params - Best quality
        'phi-3-mini': 'microsoft/Phi-3-mini-4k-instruct',  # 3.8B params
        'stablelm-3b': 'stabilityai/stablelm-3b-4e1t',     # 3B params
        'phi-2': 'microsoft/phi-2',                         # 2.7B params - Fast, good balance
        'gemma-2b': 'google/gemma-2b-it',                  # 2B params
        'qwen-1.8b': 'Qwen/Qwen2-1.8B-Instruct',           # 1.8B params - Fastest
    }

    def __init__(self, model_name: str = None, output_dir: str = 'models'):
        """
        Initialize model setup.

        Args:
            model_name: Name of the model (key from AVAILABLE_MODELS). If None, uses config.py
            output_dir: Directory to save the model
        """
        # Use config value if not specified
        self.model_name = model_name if model_name is not None else MODEL['base_model']
        self.model_path = self.AVAILABLE_MODELS.get(self.model_name)

        if not self.model_path:
            logger.warning(f"Model {model_name} not found. Available models: {list(self.AVAILABLE_MODELS.keys())}")
            logger.info("Defaulting to phi-2")
            self.model_path = self.AVAILABLE_MODELS['phi-2']

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        if self.device == "cpu":
            logger.warning("GPU not detected. Training will be slow. Consider using Google Colab or a GPU instance.")

    def load_model_and_tokenizer(self, use_4bit: bool = True):
        """
        Load model and tokenizer with quantization for memory efficiency.

        Args:
            use_4bit: Use 4-bit quantization to reduce memory usage

        Returns:
            model, tokenizer
        """
        logger.info(f"Loading model: {self.model_path}")

        # Quantization config for 4GB RAM constraint
        if use_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")
        else:
            bnb_config = None
            logger.info("Loading model without quantization")

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='right'
            )

            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            logger.info("Tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            logger.info("\nYou may need to accept the model's license agreement on Hugging Face.")
            logger.info(f"Visit: https://huggingface.co/{self.model_path}")
            logger.info("\nAlternatively, use huggingface-cli login to authenticate:")
            logger.info("pip install huggingface_hub")
            logger.info("huggingface-cli login")
            raise

        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config if use_4bit else None,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            logger.info("Model loaded successfully")
            logger.info(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")

            # Print memory usage
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory allocated: {allocated:.2f} GB")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("\nTroubleshooting:")
            logger.info("1. Check if you have accepted the model license")
            logger.info("2. Authenticate with: huggingface-cli login")
            logger.info("3. Ensure you have sufficient memory")
            logger.info(f"4. Try a smaller model from: {list(self.AVAILABLE_MODELS.keys())}")
            raise

        return model, tokenizer

    def setup_lora(self, model, r: int = None, lora_alpha: int = None,
                   target_modules: list = None, lora_dropout: float = None):
        """
        Setup LoRA (Low-Rank Adaptation) for efficient finetuning.

        Args:
            model: Base model
            r: LoRA rank (default from config.py)
            lora_alpha: LoRA alpha (default from config.py)
            target_modules: Modules to apply LoRA to (default from config.py)
            lora_dropout: Dropout rate (default from config.py)

        Returns:
            PEFT model with LoRA
        """
        # Use config values if not specified
        lora_config_values = MODEL['lora']
        r = r if r is not None else lora_config_values['r']
        lora_alpha = lora_alpha if lora_alpha is not None else lora_config_values['lora_alpha']
        lora_dropout = lora_dropout if lora_dropout is not None else lora_config_values['lora_dropout']
        target_modules = target_modules if target_modules is not None else lora_config_values['target_modules']

        logger.info(f"Setting up LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Target modules: {target_modules}")

        # LoRA configuration
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Add LoRA adapters
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return model

    def test_model(self, model, tokenizer, prompt: str = "¿Qué es el riesgo de crédito?"):
        """Test the model with a sample prompt."""
        logger.info(f"\nTesting model with prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Response: {response}\n")

        return response


def main():
    """Main function to setup and test the model."""
    logger.info("=== Model Setup ===\n")

    # List available models
    logger.info("Available models:")
    for name, path in ModelSetup.AVAILABLE_MODELS.items():
        logger.info(f"  - {name}: {path}")

    logger.info(f"\nUsing model from config.py: {MODEL['base_model']}")
    logger.info(f"LoRA config: r={MODEL['lora']['r']}, alpha={MODEL['lora']['lora_alpha']}")

    logger.info("\nIMPORTANT: Some models may require authentication.")
    logger.info("If you encounter permission errors:")
    logger.info("1. Visit the model page on Hugging Face and accept the license")
    logger.info("2. Run: huggingface-cli login")
    logger.info("3. Enter your Hugging Face token\n")

    # Setup model using config.py values (model_name=None uses config default)
    setup = ModelSetup()

    try:
        # Load model and tokenizer
        model, tokenizer = setup.load_model_and_tokenizer(use_4bit=True)

        # Setup LoRA for efficient finetuning
        model = setup.setup_lora(model)

        # Test the model
        setup.test_model(model, tokenizer)

        logger.info("Model setup complete! Ready for finetuning.")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        logger.info("\nPlease follow the instructions above to resolve the issue.")


if __name__ == "__main__":
    main()
