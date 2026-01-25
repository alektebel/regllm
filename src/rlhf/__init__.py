"""
RLHF Module for RegLLM

Implements Direct Preference Optimization (DPO) for fine-tuning based on user feedback.
"""

from .feedback_collector import FeedbackCollector
from .preference_dataset import PreferenceDataset
from .dpo_trainer import RegulationDPOTrainer

__all__ = ["FeedbackCollector", "PreferenceDataset", "RegulationDPOTrainer"]
