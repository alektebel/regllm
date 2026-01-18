"""
Training script for finetuning on banking regulation data.
Supports both small subset overfitting test and full dataset training.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List
import argparse
import sys

# Add project root to path for config import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import MODEL, TRAINING, DATA_PROCESSING
from .model_setup import ModelSetup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegulationDataset(Dataset):
    """Dataset for banking regulation QA pairs."""

    def __init__(self, data_file: Path, tokenizer, max_length: int = None):
        """
        Initialize dataset.

        Args:
            data_file: Path to JSON file with training data
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length (default from config.py)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else DATA_PROCESSING['max_seq_length']

        # Load data - support both .json and .jsonl formats
        logger.info(f"Loading data from {data_file}")

        if str(data_file).endswith('.jsonl'):
            self.data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a training example."""
        item = self.data[idx]

        # Format the conversation using the model's chat template
        messages = item['messages']
        text = self._format_messages(messages)

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Create labels (same as input_ids for causal LM)
        labels = encodings['input_ids'].clone()

        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

    def _format_messages(self, messages: List[Dict]) -> str:
        """
        Format messages using the tokenizer's chat template.
        This ensures proper formatting for Qwen2.5 (ChatML), Llama, etc.
        """
        # Use tokenizer's built-in chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return text
            except Exception as e:
                logger.warning(f"apply_chat_template failed: {e}, using fallback format")

        # Fallback: manual ChatML format (compatible with Qwen2.5)
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        return formatted


class TrainingMonitor:
    """Monitor and visualize training progress."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.eval_losses = []
        self.epochs = []

    def log_metrics(self, epoch: int, train_loss: float, eval_loss: float = None):
        """Log training metrics."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)

        if eval_loss is not None:
            self.eval_losses.append(eval_loss)

        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}" +
                   (f", Eval Loss = {eval_loss:.4f}" if eval_loss else ""))

    def plot_losses(self, filename: str = "training_progress.png"):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))

        plt.plot(self.epochs, self.train_losses, label='Training Loss', marker='o')

        if self.eval_losses:
            plt.plot(self.epochs, self.eval_losses, label='Validation Loss', marker='s')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training plot to {output_path}")
        plt.close()

    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save metrics to JSON file."""
        metrics = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses
        }

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")


class RegulationTrainer:
    """Trainer for banking regulation model."""

    def __init__(self, model_name: str = None,
                 data_dir: str = 'data/processed',
                 output_dir: str = 'models/finetuned',
                 use_small_subset: bool = False):
        """
        Initialize trainer.

        Args:
            model_name: Name of base model to use (default from config.py)
            data_dir: Directory with processed data
            output_dir: Directory to save finetuned model
            use_small_subset: Whether to use small subset for overfitting test
        """
        # Use config value if not specified
        self.model_name = model_name if model_name is not None else MODEL['base_model']
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.use_small_subset = use_small_subset

        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize monitoring
        self.monitor = TrainingMonitor(Path('logs'))

        # Setup model (will use config.py values)
        self.setup = ModelSetup(model_name=self.model_name)
        self.model = None
        self.tokenizer = None

        logger.info(f"Trainer initialized with model: {self.model_name}")
        logger.info(f"LoRA config from config.py: r={MODEL['lora']['r']}, alpha={MODEL['lora']['lora_alpha']}")

    def load_model(self):
        """Load and prepare model for training."""
        logger.info("Loading model...")

        # Load base model and tokenizer
        use_4bit = MODEL.get('use_4bit', True)
        self.model, self.tokenizer = self.setup.load_model_and_tokenizer(use_4bit=use_4bit)

        # Setup LoRA using config.py values (setup_lora now uses config defaults)
        self.model = self.setup.setup_lora(self.model)

        logger.info("Model loaded and configured")

    def load_datasets(self):
        """Load training and validation datasets."""
        # Select training file
        if self.use_small_subset:
            train_file = self.data_dir / "train_data_small.json"
            logger.info("Using SMALL SUBSET for overfitting test")
        else:
            train_file = self.data_dir / "train_data.json"
            logger.info("Using FULL DATASET for training")

        val_file = self.data_dir / "val_data.json"

        # Create datasets
        train_dataset = RegulationDataset(train_file, self.tokenizer)
        val_dataset = RegulationDataset(val_file, self.tokenizer)

        return train_dataset, val_dataset

    def train(self, num_epochs: int = None, batch_size: int = None,
             learning_rate: float = None, gradient_accumulation_steps: int = None):
        """
        Train the model.

        Args:
            num_epochs: Number of training epochs (default from config.py)
            batch_size: Batch size per device (default from config.py)
            learning_rate: Learning rate (default from config.py)
            gradient_accumulation_steps: Steps to accumulate gradients (default from config.py)
        """
        # Get training config based on subset mode
        train_config = TRAINING['small_subset'] if self.use_small_subset else TRAINING['full_training']

        # Use config values if not specified
        num_epochs = num_epochs if num_epochs is not None else train_config['epochs']
        batch_size = batch_size if batch_size is not None else train_config['batch_size']
        learning_rate = learning_rate if learning_rate is not None else train_config['learning_rate']
        gradient_accumulation_steps = gradient_accumulation_steps if gradient_accumulation_steps is not None else train_config['gradient_accumulation_steps']

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Load datasets
        train_dataset, val_dataset = self.load_datasets()

        # Training arguments using config values
        training_args = TrainingArguments(
            output_dir=str(self.run_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=TRAINING.get('weight_decay', 0.01),
            warmup_steps=TRAINING.get('warmup_steps', 100),
            max_grad_norm=TRAINING.get('max_grad_norm', 1.0),
            logging_steps=TRAINING.get('logging_steps', 10),
            eval_strategy=TRAINING.get('evaluation_strategy', 'epoch'),
            save_strategy=TRAINING.get('save_strategy', 'epoch'),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            report_to="none",  # Disable wandb, tensorboard, etc.
            save_total_limit=TRAINING.get('save_total_limit', 2),
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

        train_result = trainer.train()

        # Save model
        logger.info("Saving model...")
        trainer.save_model(str(self.run_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.run_dir / "final_model"))

        # Log metrics
        metrics = train_result.metrics
        logger.info(f"Training complete! Metrics: {metrics}")

        # Plot training progress
        self._extract_and_plot_history(trainer)

        return trainer

    def _extract_and_plot_history(self, trainer):
        """Extract training history and plot."""
        log_history = trainer.state.log_history

        train_losses = []
        eval_losses = []
        epochs = []

        for log in log_history:
            if 'loss' in log:
                train_losses.append(log['loss'])
            if 'eval_loss' in log:
                eval_losses.append(log['eval_loss'])
                epochs.append(log['epoch'])

        # Plot
        if train_losses:
            plt.figure(figsize=(12, 5))

            # Training loss
            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)

            # Validation loss
            if eval_losses:
                plt.subplot(1, 2, 2)
                plt.plot(epochs, eval_losses, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Validation Loss')
                plt.grid(True, alpha=0.3)

            output_path = Path('logs') / f"training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training plot to {output_path}")
            plt.close()

    def test_model(self, prompt: str = "¿Qué es la probabilidad de default (PD) en la regulación bancaria española?"):
        """Test the finetuned model."""
        logger.info(f"\nTesting model with prompt: {prompt}")

        # Format prompt using chat template
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to ChatML format
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nResponse:\n{response}\n")

        return response


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train banking regulation model')
    parser.add_argument('--model', type=str, default=None,
                       help=f"Model name to use (default from config.py: {MODEL['base_model']})")
    parser.add_argument('--small-subset', action='store_true',
                       help='Use small subset for overfitting test')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default from config.py)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default from config.py)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default from config.py)')

    args = parser.parse_args()

    logger.info("=== Banking Regulation Model Training ===\n")
    logger.info(f"Config: model={MODEL['base_model']}, LoRA r={MODEL['lora']['r']}, alpha={MODEL['lora']['lora_alpha']}")

    # Create trainer (uses config.py defaults if args are None)
    trainer = RegulationTrainer(
        model_name=args.model,
        use_small_subset=args.small_subset
    )

    # Train (uses config.py defaults if args are None)
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Test the model
    test_prompts = [
        "¿Qué es la probabilidad de default (PD)?",
        "¿Qué regulación se aplica al cálculo de capital para riesgo de crédito?",
        "Explica el método IRB para carteras retail"
    ]

    logger.info("\n=== Testing Finetuned Model ===\n")
    for prompt in test_prompts:
        trainer.test_model(prompt)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
