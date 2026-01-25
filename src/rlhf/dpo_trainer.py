#!/usr/bin/env python3
"""
DPO Trainer for RegLLM

Implements Direct Preference Optimization training using the TRL library.
Works with 4-bit quantized models using QLoRA.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    # Model
    model_path: str = "models/finetuned/run_20260118_220204/final_model"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # DPO hyperparameters
    beta: float = 0.1  # KL penalty coefficient
    learning_rate: float = 5e-7
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 100
    warmup_ratio: float = 0.1

    # LoRA config (for additional fine-tuning)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Sequence lengths
    max_length: int = 1024
    max_prompt_length: int = 512

    # Output
    output_dir: str = "models/dpo"

    # Training settings
    fp16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 50


class RegulationDPOTrainer:
    """DPO Trainer for banking regulation model."""

    def __init__(self, config: Optional[DPOConfig] = None):
        self.config = config or DPOConfig()
        self.model = None
        self.tokenizer = None
        self.ref_model = None

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer with 4-bit quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading model from {self.config.model_path}")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with LoRA adapters
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load existing LoRA adapters if available
        adapter_path = Path(self.config.model_path)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            from peft import PeftModel
            logger.info(f"Loading LoRA adapters from {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
                is_trainable=True,
            )
        else:
            logger.info("No existing adapters found, will create new LoRA layers")

        logger.info("Model loaded successfully")

    def _prepare_peft_model(self):
        """Prepare model with PEFT/LoRA for training."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Check if model already has PEFT
        if hasattr(self.model, 'peft_config'):
            logger.info("Model already has PEFT configuration")
            return

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"LoRA applied. Trainable params: {self.model.print_trainable_parameters()}")

    def train(
        self,
        train_dataset,
        eval_dataset=None,
    ):
        """Run DPO training."""
        from trl import DPOTrainer, DPOConfig as TRLDPOConfig

        # Load model if not already loaded
        if self.model is None:
            self._load_model_and_tokenizer()

        # Prepare PEFT
        self._prepare_peft_model()

        # DPO training arguments
        training_args = TRLDPOConfig(
            output_dir=self.config.output_dir,
            beta=self.config.beta,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_steps=self.config.max_steps,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            remove_unused_columns=False,
            report_to="none",  # Disable wandb etc.
        )

        # Create trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Use implicit reference model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting DPO training...")
        trainer.train()

        # Save final model
        output_path = Path(self.config.output_dir) / "final_model"
        trainer.save_model(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

        logger.info(f"Training complete. Model saved to {output_path}")
        return output_path

    def train_from_feedback(self, feedback_file: Optional[Path] = None):
        """Train from collected feedback data."""
        from .preference_dataset import PreferenceDataset

        # Load preference dataset
        dataset = PreferenceDataset()

        if feedback_file:
            count = dataset.load_from_feedback(feedback_file)
        else:
            count = dataset.load_from_file()

        if count == 0:
            logger.error("No preference data found. Collect feedback first!")
            return None

        logger.info(f"Loaded {count} preference pairs")

        # Get stats
        stats = dataset.get_stats()
        logger.info(f"Dataset stats: {stats}")

        # Split into train/eval
        train_data, eval_data = dataset.split_train_eval(eval_ratio=0.1)

        logger.info(f"Training on {len(train_data)} pairs, evaluating on {len(eval_data)}")

        # Run training
        return self.train(train_data, eval_data)


def main():
    """Main function for DPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="DPO Training for RegLLM")
    parser.add_argument("--feedback-file", type=str, help="Path to feedback JSONL file")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--output-dir", type=str, default="models/dpo", help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("RegLLM - DPO Training")
    print("=" * 60)

    # Create config
    config = DPOConfig(
        beta=args.beta,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )

    # Create trainer
    trainer = RegulationDPOTrainer(config)

    # Train from feedback
    feedback_file = Path(args.feedback_file) if args.feedback_file else None
    output_path = trainer.train_from_feedback(feedback_file)

    if output_path:
        print()
        print("=" * 60)
        print(f"DPO training complete!")
        print(f"Model saved to: {output_path}")
        print("=" * 60)
    else:
        print()
        print("Training failed - no preference data available.")
        print("Collect user feedback using the Gradio UI first.")


if __name__ == "__main__":
    main()
