"""
Deterministic reward functions for GRPO training.

Computes fast, deterministic rewards from ground truth test cases.
No LLM judge needed during training — keyword overlap + source matching
provides strong signal for factual QA.
"""

import re
from typing import List, Dict, Any


# Common Spanish words for language detection heuristic
_SPANISH_MARKERS = {"el", "de", "las", "los", "en", "del", "que", "por", "con", "una", "un", "es", "se"}


class TestRewardComputer:
    """Computes rewards from ground truth test cases."""

    def keyword_reward(self, completion: str, datos_clave: List[str]) -> float:
        """
        Fraction of datos_clave found in the completion (case-insensitive).

        This is the primary reward signal — if the model mentions specific
        figures like "11.076 millones EUR", it gets rewarded.
        """
        if not datos_clave:
            return 0.0

        completion_lower = completion.lower()
        matched = sum(1 for dato in datos_clave if dato.lower() in completion_lower)
        return matched / len(datos_clave)

    def source_reward(self, completion: str, fuentes_esperadas: List[Dict[str, Any]]) -> float:
        """
        Check if expected source document names appear in the completion.

        Returns 1.0 if at least one expected source is mentioned, 0.0 otherwise.
        """
        if not fuentes_esperadas:
            return 0.0

        completion_lower = completion.lower()
        for fuente in fuentes_esperadas:
            doc_name = fuente.get("documento", "")
            if doc_name and doc_name.lower() in completion_lower:
                return 1.0
        return 0.0

    def format_reward(self, completion: str) -> float:
        """
        Check format quality: non-empty, reasonable length, Spanish language.

        Returns 1.0 if all checks pass, 0.0 otherwise.
        """
        # Must be non-empty
        stripped = completion.strip()
        if not stripped:
            return 0.0

        # Reasonable length (50-1000 chars)
        if len(stripped) < 50 or len(stripped) > 1000:
            return 0.0

        # Spanish language heuristic: at least 3 common Spanish words present
        words = set(re.findall(r'\b\w+\b', stripped.lower()))
        spanish_count = len(words & _SPANISH_MARKERS)
        if spanish_count < 3:
            return 0.0

        return 1.0

    def combined_reward(
        self,
        completion: str,
        entry: Dict[str, Any],
        weights: Dict[str, float] = None,
    ) -> float:
        """
        Compute weighted combination of all reward components.

        Args:
            completion: Model-generated text.
            entry: Ground truth entry with datos_clave, fuentes_esperadas, etc.
            weights: Optional dict with keys 'keyword', 'source', 'format'.
                     Defaults to 0.5/0.3/0.2.

        Returns:
            Combined reward score in [0.0, 1.0].
        """
        if weights is None:
            weights = {"keyword": 0.5, "source": 0.3, "format": 0.2}

        kw = self.keyword_reward(completion, entry.get("datos_clave", []))
        src = self.source_reward(completion, entry.get("fuentes_esperadas", []))
        fmt = self.format_reward(completion)

        return weights["keyword"] * kw + weights["source"] * src + weights["format"] * fmt
