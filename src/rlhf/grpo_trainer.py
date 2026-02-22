#!/usr/bin/env python3
"""
GRPO Trainer for RegLLM

Implements Group Relative Policy Optimization (DeepSeek-R1 / RLVR pattern)
using TRL's GRPOTrainer with deterministic test-based rewards.

For each prompt, generates G completions, scores them against ground truth
using keyword/source/format rewards, and updates the policy via clipped
surrogate loss with group-relative advantages.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig

from config import GRPO, INFERENCE, MODEL
from src.rlhf.grpo_rewards import TestRewardComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegLLMGRPOTrainer:
    """GRPO trainer for banking regulation QA using test-as-reward."""

    def __init__(
        self,
        output_dir: str = "models/grpo",
        num_generations: int = None,
        max_completion_length: int = None,
        learning_rate: float = None,
        num_train_epochs: int = None,
        per_device_train_batch_size: int = None,
        gradient_accumulation_steps: int = None,
        beta: float = None,
        lora_r: int = None,
        lora_alpha: int = None,
        ground_truth_path: str = "data/test_ground_truth.json",
    ):
        self.output_dir = output_dir
        self.num_generations = num_generations or GRPO['group_size']
        self.max_completion_length = max_completion_length or GRPO['max_completion_length']
        self.learning_rate = learning_rate or GRPO['learning_rate']
        self.num_train_epochs = num_train_epochs or GRPO['epochs']
        self.per_device_train_batch_size = per_device_train_batch_size or GRPO['batch_size']
        self.gradient_accumulation_steps = gradient_accumulation_steps or GRPO['gradient_accumulation_steps']
        self.beta = beta or GRPO['beta']
        self.lora_r = lora_r or GRPO['lora']['r']
        self.lora_alpha = lora_alpha or GRPO['lora']['lora_alpha']
        self.ground_truth_path = PROJECT_ROOT / ground_truth_path

        self.reward_weights = GRPO['reward_weights']
        self.reward_computer = TestRewardComputer()

        # Resolve base model name (inline to avoid import chain issues)
        _AVAILABLE_MODELS = {
            'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
            'phi-3-mini': 'microsoft/Phi-3-mini-4k-instruct',
            'stablelm-3b': 'stabilityai/stablelm-3b-4e1t',
            'phi-2': 'microsoft/phi-2',
            'gemma-2b': 'google/gemma-2b-it',
            'qwen-1.8b': 'Qwen/Qwen2-1.8B-Instruct',
        }
        self.base_model = _AVAILABLE_MODELS.get(
            MODEL['base_model'], MODEL['base_model']
        )

    def _build_dataset(self) -> Dataset:
        """
        Load ground truth and convert to HF Dataset.

        The dataset has a 'prompt' column (formatted with system prompt)
        plus metadata columns passed through as kwargs to the reward function.
        """
        with open(self.ground_truth_path) as f:
            data = json.load(f)

        entries = data["entries"]
        system_prompt = INFERENCE['system_prompt']

        records = []
        for entry in entries:
            # Format prompt as chat messages for Qwen
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": entry["pregunta"]},
            ]

            records.append({
                "prompt": prompt,
                "datos_clave": json.dumps(entry.get("datos_clave", [])),
                "fuentes_esperadas": json.dumps(entry.get("fuentes_esperadas", [])),
                "entry_id": entry["id"],
            })

        dataset = Dataset.from_list(records)
        logger.info(f"Built dataset with {len(dataset)} entries")
        return dataset

    def _reward_func(self, prompts, completions, **kwargs):
        """
        Reward function passed to GRPOTrainer.

        Computes combined reward for each (prompt, completion) pair using
        keyword overlap, source matching, and format checks.

        Args:
            prompts: List of prompt strings.
            completions: List of completion strings.
            **kwargs: Dataset metadata columns (datos_clave, fuentes_esperadas, etc.)

        Returns:
            List of float rewards.
        """
        datos_clave_list = kwargs.get("datos_clave", [])
        fuentes_list = kwargs.get("fuentes_esperadas", [])

        rewards = []
        for i, completion in enumerate(completions):
            # Parse JSON metadata for this sample
            try:
                datos = json.loads(datos_clave_list[i]) if isinstance(datos_clave_list[i], str) else datos_clave_list[i]
            except (json.JSONDecodeError, IndexError, TypeError):
                datos = []

            try:
                fuentes = json.loads(fuentes_list[i]) if isinstance(fuentes_list[i], str) else fuentes_list[i]
            except (json.JSONDecodeError, IndexError, TypeError):
                fuentes = []

            entry = {"datos_clave": datos, "fuentes_esperadas": fuentes}
            reward = self.reward_computer.combined_reward(
                completion, entry, weights=self.reward_weights
            )
            rewards.append(reward)

        return rewards

    def train(self):
        """Run GRPO training."""
        from trl import GRPOTrainer, GRPOConfig

        logger.info("=" * 60)
        logger.info("RegLLM - GRPO Training")
        logger.info("=" * 60)
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"Num generations per prompt: {self.num_generations}")
        logger.info(f"Max completion length: {self.max_completion_length}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Epochs: {self.num_train_epochs}")
        logger.info(f"Beta (KL penalty): {self.beta}")
        logger.info(f"LoRA r={self.lora_r}, alpha={self.lora_alpha}")
        logger.info(f"Reward weights: {self.reward_weights}")

        # Build dataset
        dataset = self._build_dataset()

        # LoRA config
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=GRPO['lora']['lora_dropout'],
            target_modules=GRPO['lora']['target_modules'],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # GRPO training config
        training_args = GRPOConfig(
            output_dir=self.output_dir,
            num_generations=self.num_generations,
            max_completion_length=self.max_completion_length,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            beta=self.beta,
            loss_type="grpo",
            scale_rewards="group",
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            fp16=not (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
            temperature=0.7,
        )

        # Model kwargs for 4-bit quantization
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if MODEL['use_4bit']:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        training_args.model_init_kwargs = model_kwargs

        # Create trainer
        trainer = GRPOTrainer(
            model=self.base_model,
            reward_funcs=self._reward_func,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
        )

        logger.info("Starting GRPO training...")
        trainer.train()

        # Save final model
        output_path = Path(self.output_dir) / "final_model"
        trainer.save_model(str(output_path))
        if trainer.processing_class is not None:
            trainer.processing_class.save_pretrained(str(output_path))

        logger.info(f"Training complete. Model saved to {output_path}")
        return output_path

    def evaluate(self, model_path: Optional[str] = None):
        """
        Evaluate a trained model by computing average reward on the ground truth set.

        Args:
            model_path: Path to the trained model. If None, uses output_dir/final_model.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path = model_path or str(Path(self.output_dir) / "final_model")
        logger.info(f"Evaluating model from {model_path}")

        # Load model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load ground truth
        with open(self.ground_truth_path) as f:
            data = json.load(f)

        system_prompt = INFERENCE['system_prompt']
        entries = data["entries"]

        total_reward = 0.0
        total_keyword = 0.0
        total_source = 0.0
        total_format = 0.0

        for entry in entries:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": entry["pregunta"]},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_completion_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

            kw = self.reward_computer.keyword_reward(completion, entry.get("datos_clave", []))
            src = self.reward_computer.source_reward(completion, entry.get("fuentes_esperadas", []))
            fmt = self.reward_computer.format_reward(completion)
            combined = self.reward_computer.combined_reward(completion, entry, self.reward_weights)

            total_keyword += kw
            total_source += src
            total_format += fmt
            total_reward += combined

            logger.info(
                f"[{entry['id']}] kw={kw:.2f} src={src:.1f} fmt={fmt:.1f} "
                f"combined={combined:.2f} | {completion[:80]}..."
            )

        n = len(entries)
        logger.info("=" * 60)
        logger.info(f"Evaluation Results ({n} entries)")
        logger.info(f"  Avg keyword reward:  {total_keyword / n:.3f}")
        logger.info(f"  Avg source reward:   {total_source / n:.3f}")
        logger.info(f"  Avg format reward:   {total_format / n:.3f}")
        logger.info(f"  Avg combined reward: {total_reward / n:.3f}")
        logger.info("=" * 60)

        return {
            "keyword": total_keyword / n,
            "source": total_source / n,
            "format": total_format / n,
            "combined": total_reward / n,
        }


def main():
    """Main entry point for GRPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Training for RegLLM")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--group-size", type=int, default=None, help="Completions per prompt")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--beta", type=float, default=None, help="KL penalty coefficient")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--output-dir", type=str, default="models/grpo", help="Output directory")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--model-path", type=str, default=None, help="Model path for eval-only")
    args = parser.parse_args()

    trainer = RegLLMGRPOTrainer(
        output_dir=args.output_dir,
        num_generations=args.group_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        beta=args.beta,
    )

    if args.eval_only:
        trainer.evaluate(model_path=args.model_path)
    else:
        output_path = trainer.train()
        print()
        print("=" * 60)
        print(f"GRPO training complete!")
        print(f"Model saved to: {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
