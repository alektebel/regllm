#!/usr/bin/env python3
"""
Preference Dataset for DPO Training

Formats preference pairs into datasets compatible with TRL's DPOTrainer.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class PreferencePair:
    """A single preference pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    metadata: Optional[Dict] = None


class PreferenceDataset:
    """Manages preference data for DPO training."""

    SYSTEM_PROMPT = """Eres un experto en regulación bancaria española y europea. Tu tarea es responder preguntas sobre normativa prudencial, especialmente sobre parámetros de riesgo de crédito (PD, LGD, EAD), Basilea III, CRR y directrices de la EBA.

REGLAS IMPORTANTES:
- Responde SIEMPRE en español
- Cita la fuente específica de tu información
- Si no tienes información suficiente, indícalo claramente
- Sé preciso y técnico en tus explicaciones"""

    def __init__(self, pairs_file: Optional[Path] = None):
        self.pairs_file = pairs_file or Path("data/preferences/dpo_pairs.json")
        self.pairs: List[PreferencePair] = []

    def load_from_file(self, path: Optional[Path] = None) -> int:
        """Load preference pairs from JSON file."""
        path = path or self.pairs_file

        if not path.exists():
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.pairs = []
        for item in data:
            self.pairs.append(PreferencePair(
                prompt=item["prompt"],
                chosen=item["chosen"],
                rejected=item["rejected"],
                metadata=item.get("metadata"),
            ))

        return len(self.pairs)

    def load_from_feedback(self, feedback_file: Path) -> int:
        """Load preference pairs from raw feedback JSONL file."""
        from .feedback_collector import FeedbackCollector

        collector = FeedbackCollector(feedback_file=feedback_file)
        pairs_data = collector.create_preference_pairs()

        self.pairs = []
        for item in pairs_data:
            self.pairs.append(PreferencePair(
                prompt=item["prompt"],
                chosen=item["chosen"],
                rejected=item["rejected"],
            ))

        return len(self.pairs)

    def add_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        metadata: Optional[Dict] = None,
    ):
        """Add a preference pair manually."""
        self.pairs.append(PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata,
        ))

    def _format_prompt(self, user_query: str) -> str:
        """Format the prompt with system message for training."""
        # Format in ChatML style
        return f"""<|im_start|>system
{self.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""

    def _format_response(self, response: str) -> str:
        """Format the response for training."""
        return f"{response}<|im_end|>"

    def to_dpo_format(self) -> List[Dict]:
        """Convert to format expected by TRL DPOTrainer."""
        formatted = []

        for pair in self.pairs:
            formatted.append({
                "prompt": self._format_prompt(pair.prompt),
                "chosen": self._format_response(pair.chosen),
                "rejected": self._format_response(pair.rejected),
            })

        return formatted

    def to_hf_dataset(self) -> "Dataset":
        """Convert to HuggingFace Dataset for DPOTrainer."""
        if not HAS_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")

        formatted = self.to_dpo_format()
        return Dataset.from_list(formatted)

    def save(self, path: Optional[Path] = None):
        """Save preference pairs to JSON file."""
        path = path or self.pairs_file
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for pair in self.pairs:
            item = {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
            }
            if pair.metadata:
                item["metadata"] = pair.metadata
            data.append(item)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def split_train_eval(
        self, eval_ratio: float = 0.1
    ) -> tuple:
        """Split dataset into training and evaluation sets."""
        if not HAS_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")

        dataset = self.to_hf_dataset()
        split = dataset.train_test_split(test_size=eval_ratio)

        return split["train"], split["test"]

    def get_stats(self) -> Dict:
        """Get statistics about the preference dataset."""
        if not self.pairs:
            return {"total_pairs": 0}

        chosen_lengths = [len(p.chosen) for p in self.pairs]
        rejected_lengths = [len(p.rejected) for p in self.pairs]
        prompt_lengths = [len(p.prompt) for p in self.pairs]

        return {
            "total_pairs": len(self.pairs),
            "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
            "avg_chosen_length": sum(chosen_lengths) / len(chosen_lengths),
            "avg_rejected_length": sum(rejected_lengths) / len(rejected_lengths),
            "unique_prompts": len(set(p.prompt for p in self.pairs)),
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)


def main():
    """Demo of preference dataset."""
    dataset = PreferenceDataset()

    # Add some demo pairs
    dataset.add_pair(
        prompt="¿Qué es la probabilidad de impago?",
        chosen="La probabilidad de impago (PD) es la probabilidad de que un deudor no cumpla con sus obligaciones de pago en un horizonte temporal de 12 meses. Según el artículo 4 del CRR, el impago se considera cuando el deudor tiene obligaciones vencidas por más de 90 días o cuando la entidad considera improbable que pague.",
        rejected="La PD es un número que los bancos usan.",
    )

    dataset.add_pair(
        prompt="¿Cuál es el floor de PD bajo IRB?",
        chosen="Según Basilea III final y el CRR3, el floor de PD para exposiciones corporativas es del 0.05% (5 puntos básicos). Este floor asegura que ninguna exposición tenga una PD estimada inferior a este umbral, proporcionando un nivel mínimo de conservadurismo.",
        rejected="No estoy seguro del floor de PD.",
    )

    # Get stats
    stats = dataset.get_stats()
    print("Preference Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save
    dataset.save()
    print(f"\nSaved to: {dataset.pairs_file}")


if __name__ == "__main__":
    main()
