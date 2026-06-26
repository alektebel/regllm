"""Autoresearch — self-improving research loop for RegLLM.

Orchestrates training experiments, generates synthetic tool-calling and
Spanish case data, evaluates the 4B model, and grows the knowledge graph
from research outcomes.
"""

from .loop import ResearchLoop
from .experiment import ExperimentTracker
from .synthesizer import SFTDataSynthesizer
from .evaluator import ToolCallingEvaluator
from .kb_sync import KBSynchronizer

__all__ = [
    "ResearchLoop",
    "ExperimentTracker",
    "SFTDataSynthesizer",
    "ToolCallingEvaluator",
    "KBSynchronizer",
]
