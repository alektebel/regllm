#!/usr/bin/env python3
"""
Feedback Collector for RLHF

Collects and stores user feedback (thumbs up/down) for DPO training.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict
import threading


@dataclass
class FeedbackEntry:
    """Represents a single feedback entry."""
    id: str
    timestamp: str
    query: str
    response: str
    feedback: Literal["positive", "negative"]
    model_name: str
    session_id: Optional[str] = None
    context: Optional[str] = None
    metadata: Optional[Dict] = None


class FeedbackCollector:
    """Collects and manages user feedback for RLHF training."""

    DEFAULT_FEEDBACK_FILE = Path("data/preferences/feedback.jsonl")

    def __init__(
        self,
        feedback_file: Optional[Path] = None,
        model_name: str = "regllm",
    ):
        self.feedback_file = feedback_file or self.DEFAULT_FEEDBACK_FILE
        self.model_name = model_name
        self._lock = threading.Lock()

        # Ensure directory exists
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

    def record_feedback(
        self,
        query: str,
        response: str,
        feedback: Literal["positive", "negative"],
        session_id: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Record a feedback entry.

        Args:
            query: The user's question
            response: The model's response
            feedback: "positive" for thumbs up, "negative" for thumbs down
            session_id: Optional session identifier
            context: Optional RAG context used
            metadata: Optional additional metadata

        Returns:
            The feedback entry ID
        """
        entry = FeedbackEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response,
            feedback=feedback,
            model_name=self.model_name,
            session_id=session_id,
            context=context,
            metadata=metadata,
        )

        with self._lock:
            with open(self.feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        return entry.id

    def get_all_feedback(self) -> List[FeedbackEntry]:
        """Load all feedback entries from file."""
        entries = []

        if not self.feedback_file.exists():
            return entries

        with open(self.feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entries.append(FeedbackEntry(**data))

        return entries

    def get_positive_feedback(self) -> List[FeedbackEntry]:
        """Get all positive (thumbs up) feedback entries."""
        return [e for e in self.get_all_feedback() if e.feedback == "positive"]

    def get_negative_feedback(self) -> List[FeedbackEntry]:
        """Get all negative (thumbs down) feedback entries."""
        return [e for e in self.get_all_feedback() if e.feedback == "negative"]

    def get_feedback_stats(self) -> Dict:
        """Get statistics about collected feedback."""
        entries = self.get_all_feedback()
        positive = sum(1 for e in entries if e.feedback == "positive")
        negative = len(entries) - positive

        return {
            "total": len(entries),
            "positive": positive,
            "negative": negative,
            "positive_ratio": positive / len(entries) if entries else 0,
            "file_path": str(self.feedback_file),
        }

    def create_preference_pairs(self) -> List[Dict]:
        """
        Create preference pairs for DPO training.

        Returns list of dicts with format:
        {
            "prompt": str,
            "chosen": str,  # Positive feedback response
            "rejected": str,  # Negative feedback response (or placeholder)
        }
        """
        entries = self.get_all_feedback()

        # Group by query
        by_query: Dict[str, Dict[str, List[str]]] = {}
        for entry in entries:
            if entry.query not in by_query:
                by_query[entry.query] = {"positive": [], "negative": []}
            by_query[entry.query][entry.feedback].append(entry.response)

        # Create pairs where we have both positive and negative
        pairs = []
        for query, responses in by_query.items():
            if responses["positive"] and responses["negative"]:
                # We have both - create pairs
                for chosen in responses["positive"]:
                    for rejected in responses["negative"]:
                        pairs.append({
                            "prompt": query,
                            "chosen": chosen,
                            "rejected": rejected,
                        })
            elif responses["positive"]:
                # Only positive - use a generic "bad" response as rejected
                for chosen in responses["positive"]:
                    pairs.append({
                        "prompt": query,
                        "chosen": chosen,
                        "rejected": "No tengo información suficiente para responder esta pregunta.",
                    })

        return pairs

    def export_for_dpo(self, output_path: Optional[Path] = None) -> Path:
        """Export preference pairs in format suitable for DPO training."""
        pairs = self.create_preference_pairs()

        output_path = output_path or Path("data/preferences/dpo_pairs.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

        return output_path

    def clear_feedback(self, backup: bool = True):
        """Clear all feedback data."""
        if backup and self.feedback_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.feedback_file.with_suffix(f".{timestamp}.jsonl.bak")
            self.feedback_file.rename(backup_path)
        elif self.feedback_file.exists():
            self.feedback_file.unlink()


def main():
    """Demo of feedback collection."""
    collector = FeedbackCollector()

    # Demo entries
    collector.record_feedback(
        query="¿Qué es la PD?",
        response="La PD (Probabilidad de Impago) es la probabilidad de que un deudor incumpla sus obligaciones en un horizonte de 12 meses.",
        feedback="positive",
    )

    collector.record_feedback(
        query="¿Qué es la PD?",
        response="PD significa algo relacionado con pagos.",
        feedback="negative",
    )

    # Show stats
    stats = collector.get_feedback_stats()
    print("Feedback Statistics:")
    print(f"  Total entries: {stats['total']}")
    print(f"  Positive: {stats['positive']}")
    print(f"  Negative: {stats['negative']}")

    # Create pairs
    pairs = collector.create_preference_pairs()
    print(f"\nGenerated {len(pairs)} preference pairs for DPO")


if __name__ == "__main__":
    main()
