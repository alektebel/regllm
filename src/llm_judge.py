"""
LLM Judge module for evaluating RAG response quality.

Uses Qwen 2.5-7B-Instruct with 4-bit quantization to judge whether
RAG responses are semantically adequate compared to ground truth.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Evaluation prompt template in Spanish
JUDGE_PROMPT_TEMPLATE = """Eres un evaluador experto de respuestas sobre regulación bancaria española.

Tu tarea es evaluar si la RESPUESTA GENERADA es adecuada comparada con la RESPUESTA ESPERADA.

PREGUNTA: {pregunta}

RESPUESTA ESPERADA: {respuesta_esperada}

RESPUESTA GENERADA: {respuesta_generada}

DATOS CLAVE que deben estar presentes: {datos_clave}

Evalúa la respuesta generada considerando:
1. ¿Contiene los datos clave esperados?
2. ¿Es factualmente correcta respecto a la respuesta esperada?
3. ¿Cubre los puntos principales de la respuesta esperada?

Responde EXACTAMENTE en este formato:
ADECUADA: [SI/NO]
PUNTUACION: [0.0-1.0]
DATOS_PRESENTES: [lista de datos clave encontrados, separados por coma]
DATOS_AUSENTES: [lista de datos clave no encontrados, separados por coma]
EXPLICACION: [breve explicación en una línea]"""


@dataclass
class JudgeResult:
    """Result of an LLM judge evaluation."""
    is_adequate: bool
    score: float
    explanation: str
    key_facts_matched: List[str] = field(default_factory=list)
    key_facts_missing: List[str] = field(default_factory=list)


class LLMJudge:
    """
    LLM-based judge for evaluating RAG response quality.

    Uses Qwen 2.5-7B-Instruct with 4-bit quantization as the judge model.
    Implements singleton pattern so the model is loaded only once per session.
    """

    _instance: Optional["LLMJudge"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if LLMJudge._initialized:
            return
        self.model = None
        self.tokenizer = None
        self._available = None
        LLMJudge._initialized = True

    @property
    def available(self) -> bool:
        """Check if GPU is available and model can be loaded."""
        if self._available is None:
            try:
                import torch
                self._available = torch.cuda.is_available()
                if not self._available:
                    logger.warning("No GPU available. LLM judge will be skipped.")
            except ImportError:
                self._available = False
                logger.warning("PyTorch not installed. LLM judge will be skipped.")
        return self._available

    def load_model(self):
        """Load the Qwen 2.5-7B-Instruct model with 4-bit quantization."""
        if self.model is not None:
            return

        if not self.available:
            raise RuntimeError("GPU not available for LLM judge")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_name = "Qwen/Qwen2.5-7B-Instruct"

        logger.info(f"Loading judge model: {model_name}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("Judge model loaded successfully")

    def evaluate(
        self,
        pregunta: str,
        respuesta_generada: str,
        respuesta_esperada: str,
        datos_clave: List[str],
    ) -> JudgeResult:
        """
        Evaluate a RAG response against ground truth using the LLM judge.

        Falls back to keyword-overlap heuristic if model output is malformed.
        """
        self.load_model()

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            pregunta=pregunta,
            respuesta_esperada=respuesta_esperada,
            respuesta_generada=respuesta_generada,
            datos_clave=", ".join(datos_clave),
        )

        messages = [
            {"role": "system", "content": "Eres un evaluador preciso y objetivo."},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        import torch

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        result = self._parse_judge_output(response_text, datos_clave, respuesta_generada)
        return result

    def _parse_judge_output(
        self,
        output: str,
        datos_clave: List[str],
        respuesta_generada: str,
    ) -> JudgeResult:
        """
        Parse structured output from the judge model.

        Falls back to keyword overlap heuristic on datos_clave if parsing fails.
        """
        try:
            # Try structured parsing
            adequate_match = re.search(r"ADECUADA:\s*(SI|NO|Sí|No|si|no)", output, re.IGNORECASE)
            score_match = re.search(r"PUNTUACION:\s*([\d.]+)", output)
            present_match = re.search(r"DATOS_PRESENTES:\s*(.+?)(?:\n|$)", output)
            missing_match = re.search(r"DATOS_AUSENTES:\s*(.+?)(?:\n|$)", output)
            explanation_match = re.search(r"EXPLICACION:\s*(.+?)(?:\n|$)", output)

            if adequate_match and score_match:
                is_adequate = adequate_match.group(1).upper() in ("SI", "SÍ")
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))

                matched = []
                missing = []
                if present_match:
                    matched = [d.strip() for d in present_match.group(1).split(",") if d.strip() and d.strip().lower() not in ("ninguno", "none", "n/a")]
                if missing_match:
                    missing = [d.strip() for d in missing_match.group(1).split(",") if d.strip() and d.strip().lower() not in ("ninguno", "none", "n/a")]

                explanation = explanation_match.group(1).strip() if explanation_match else "Evaluación completada"

                return JudgeResult(
                    is_adequate=is_adequate,
                    score=score,
                    explanation=explanation,
                    key_facts_matched=matched,
                    key_facts_missing=missing,
                )

        except (ValueError, AttributeError):
            pass

        # Fallback: keyword overlap heuristic
        logger.warning("Judge output malformed, falling back to keyword heuristic")
        return self._keyword_fallback(datos_clave, respuesta_generada)

    def _keyword_fallback(
        self,
        datos_clave: List[str],
        respuesta_generada: str,
    ) -> JudgeResult:
        """Fallback heuristic based on keyword overlap on datos_clave."""
        respuesta_lower = respuesta_generada.lower()

        matched = []
        missing = []
        for dato in datos_clave:
            if dato.lower() in respuesta_lower:
                matched.append(dato)
            else:
                missing.append(dato)

        total = len(datos_clave)
        score = len(matched) / total if total > 0 else 0.0
        is_adequate = score >= 0.5

        return JudgeResult(
            is_adequate=is_adequate,
            score=score,
            explanation=f"Fallback heurístico: {len(matched)}/{total} datos clave encontrados",
            key_facts_matched=matched,
            key_facts_missing=missing,
        )

    @classmethod
    def reset(cls):
        """Reset singleton (for testing purposes)."""
        cls._instance = None
        cls._initialized = False
