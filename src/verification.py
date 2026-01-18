"""
Verification System for Banking Regulatory Assistant.
Verifies response accuracy, citation validity, and language compliance.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import langdetect, but make it optional
try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available. Language verification will be limited.")


class SistemaVerificacion:
    """
    Verifies the accuracy of generated responses against source documents.
    """

    def __init__(self, model=None, tokenizer=None):
        """
        Initialize the verification system.

        Args:
            model: Optional LLM model for coherence verification
            tokenizer: Optional tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

    def verificar_respuesta(self, pregunta: str, respuesta: str, fuentes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify that the response is grounded in the sources.

        Args:
            pregunta: User's question
            respuesta: Generated response
            fuentes: List of source chunks used for generation

        Returns:
            Verification results with confidence scores
        """
        verificaciones = {
            'citaciones': self._verificar_citaciones(respuesta, fuentes),
            'coherencia': self._verificar_coherencia(respuesta, fuentes),
            'hallucination': self._detectar_hallucination(respuesta, fuentes),
            'idioma': self._verificar_español(respuesta)
        }

        # Calculate global score
        score_global = self._calcular_score_confianza(verificaciones)

        return {
            'verificaciones': verificaciones,
            'score_confianza': score_global,
            'aprobada': score_global >= 0.7,
            'nivel_confianza': self._get_nivel_confianza(score_global)
        }

    def _verificar_citaciones(self, respuesta: str, fuentes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify that citations in the response exist in the sources.

        Args:
            respuesta: Generated response
            fuentes: Source chunks

        Returns:
            Citation verification results
        """
        # Extract citations: [Articulo X], [CRR Art. Y], etc.
        patron = r'\[((?:Art(?:[ií]culo|\.)?|CRR|CRD|EBA|Reglamento|Directiva|Basel|Basilea)[^\]]*)\]'
        citaciones = re.findall(patron, respuesta, re.IGNORECASE)

        resultados = []
        for cita in citaciones:
            # Search in sources
            encontrada = False
            fuente_origen = None

            for fuente in fuentes:
                texto_fuente = fuente.get('texto', '').lower()
                metadata_str = str(fuente.get('metadata', {})).lower()

                if cita.lower() in texto_fuente or cita.lower() in metadata_str:
                    encontrada = True
                    fuente_origen = fuente.get('metadata', {}).get('documento',
                                   fuente.get('metadata', {}).get('source', 'Desconocido'))
                    break

                # Also check for partial matches (article numbers)
                numeros = re.findall(r'\d+', cita)
                for numero in numeros:
                    if f"art[ií]culo {numero}" in texto_fuente or f"art. {numero}" in texto_fuente:
                        encontrada = True
                        fuente_origen = fuente.get('metadata', {}).get('documento',
                                       fuente.get('metadata', {}).get('source', 'Desconocido'))
                        break

            resultados.append({
                'citacion': cita,
                'verificada': encontrada,
                'fuente': fuente_origen
            })

        tasa_verificacion = (sum(r['verificada'] for r in resultados) / len(resultados)
                           if resultados else 1.0)  # If no citations, consider OK

        return {
            'citaciones_encontradas': len(citaciones),
            'citaciones_verificadas': sum(r['verificada'] for r in resultados),
            'tasa_verificacion': tasa_verificacion,
            'detalles': resultados
        }

    def _verificar_coherencia(self, respuesta: str, fuentes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify coherence between response and sources.

        Args:
            respuesta: Generated response
            fuentes: Source chunks

        Returns:
            Coherence verification results
        """
        if self.model and self.tokenizer:
            return self._verificar_coherencia_con_llm(respuesta, fuentes)
        else:
            return self._verificar_coherencia_simple(respuesta, fuentes)

    def _verificar_coherencia_simple(self, respuesta: str, fuentes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simple coherence check without LLM (keyword overlap).
        """
        # Extract key terms from response
        palabras_respuesta = set(re.findall(r'\b\w{4,}\b', respuesta.lower()))

        # Extract key terms from sources
        palabras_fuentes = set()
        for fuente in fuentes:
            texto = fuente.get('texto', '')
            palabras_fuentes.update(re.findall(r'\b\w{4,}\b', texto.lower()))

        # Calculate overlap
        if not palabras_respuesta:
            return {
                'nivel': 'INDETERMINADO',
                'score': 0.5,
                'explicacion': 'No se encontraron palabras clave en la respuesta'
            }

        overlap = len(palabras_respuesta & palabras_fuentes) / len(palabras_respuesta)

        if overlap >= 0.6:
            nivel = 'COHERENTE'
            score = 1.0
        elif overlap >= 0.3:
            nivel = 'PARCIALMENTE_COHERENTE'
            score = 0.6
        else:
            nivel = 'POSIBLE_INCOHERENCIA'
            score = 0.3

        return {
            'nivel': nivel,
            'score': score,
            'explicacion': f'Solapamiento de palabras clave: {overlap:.1%}',
            'overlap_ratio': overlap
        }

    def _verificar_coherencia_con_llm(self, respuesta: str, fuentes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify coherence using LLM (if available).
        """
        # Concatenate sources
        contexto_fuentes = "\n\n".join([f.get('texto', '')[:500] for f in fuentes[:3]])

        prompt_verificacion = f"""Eres un verificador de exactitud. Compara esta respuesta con las fuentes proporcionadas.

RESPUESTA A VERIFICAR:
{respuesta}

FUENTES ORIGINALES:
{contexto_fuentes}

Responde solo: COHERENTE, PARCIALMENTE_COHERENTE, o INCOHERENTE.
Luego explica brevemente por que."""

        try:
            messages = [
                {"role": "system", "content": "Eres un verificador de exactitud que responde en espanol."},
                {"role": "user", "content": prompt_verificacion}
            ]

            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(input_text, return_tensors="pt")
            if hasattr(inputs, 'to'):
                inputs = inputs.to("cuda")

            outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.3)

            verificacion = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Parse result
            if "COHERENTE" in verificacion.upper() and "PARCIALMENTE" not in verificacion.upper():
                nivel = "COHERENTE"
                score = 1.0
            elif "PARCIALMENTE" in verificacion.upper():
                nivel = "PARCIALMENTE_COHERENTE"
                score = 0.6
            else:
                nivel = "INCOHERENTE"
                score = 0.2

            return {
                'nivel': nivel,
                'score': score,
                'explicacion': verificacion
            }

        except Exception as e:
            logger.error(f"Error in LLM coherence check: {e}")
            return self._verificar_coherencia_simple(respuesta, fuentes)

    def _detectar_hallucination(self, respuesta: str, fuentes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect potential hallucinations (fabricated information).

        Args:
            respuesta: Generated response
            fuentes: Source chunks

        Returns:
            Hallucination detection results
        """
        # Extract numerical claims from response
        patron_numeros = r'(\d+(?:[.,]\d+)?)\s*%'
        numeros_respuesta = re.findall(patron_numeros, respuesta)

        # Combine all source text
        texto_fuentes = ' '.join([f.get('texto', '') for f in fuentes])

        hallucination_detectada = False
        detalles = []

        for numero in numeros_respuesta:
            # Normalize number format
            numero_normalizado = numero.replace(',', '.')

            # Check if this number appears in sources
            encontrado = (numero in texto_fuentes or
                         numero_normalizado in texto_fuentes or
                         numero.replace('.', ',') in texto_fuentes)

            if not encontrado:
                # Check common regulatory values that are well-known
                valores_conocidos = ['4,5', '4.5', '6', '8', '10,5', '10.5', '2,5', '2.5', '3', '1', '0']
                if numero_normalizado not in valores_conocidos:
                    hallucination_detectada = True
                    detalles.append(f"Numero {numero}% no encontrado en fuentes")

        # Check for invented article references
        patron_articulos = r'art[ií]culo\s+(\d+[a-z]?)'
        articulos_respuesta = re.findall(patron_articulos, respuesta, re.IGNORECASE)

        for articulo in articulos_respuesta:
            encontrado = False
            for fuente in fuentes:
                texto = fuente.get('texto', '').lower()
                if f"articulo {articulo}" in texto or f"artículo {articulo}" in texto or f"art. {articulo}" in texto:
                    encontrado = True
                    break

            if not encontrado:
                detalles.append(f"Articulo {articulo} no encontrado en fuentes")

        return {
            'hallucination_detectada': hallucination_detectada or len(detalles) > 0,
            'confianza': 0.3 if hallucination_detectada else (0.7 if detalles else 0.9),
            'detalles': detalles
        }

    def _verificar_español(self, respuesta: str) -> Dict[str, Any]:
        """
        Verify that the response is in Spanish.

        Args:
            respuesta: Generated response

        Returns:
            Language verification results
        """
        if LANGDETECT_AVAILABLE:
            try:
                idioma = langdetect.detect(respuesta)
                es_español = (idioma == 'es')

                return {
                    'es_español': es_español,
                    'idioma_detectado': idioma,
                    'confianza': 1.0 if es_español else 0.0
                }
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")

        # Fallback: check for common Spanish words
        palabras_español = ['de', 'la', 'el', 'que', 'en', 'los', 'del', 'las', 'por', 'con',
                          'para', 'una', 'son', 'es', 'como', 'pero', 'sus', 'al', 'este']

        palabras_respuesta = respuesta.lower().split()
        count_español = sum(1 for p in palabras_respuesta if p in palabras_español)

        ratio = count_español / len(palabras_respuesta) if palabras_respuesta else 0
        es_español = ratio > 0.15  # At least 15% common Spanish words

        return {
            'es_español': es_español,
            'idioma_detectado': 'es' if es_español else 'desconocido',
            'confianza': 0.8 if es_español else 0.3,
            'metodo': 'heuristic'
        }

    def _calcular_score_confianza(self, verificaciones: Dict[str, Any]) -> float:
        """
        Calculate global confidence score (0-1).

        Args:
            verificaciones: Dictionary with all verification results

        Returns:
            Global confidence score
        """
        scores = [
            verificaciones['citaciones']['tasa_verificacion'] * 0.3,  # 30%
            verificaciones['coherencia']['score'] * 0.4,              # 40%
            verificaciones['hallucination']['confianza'] * 0.2,       # 20%
            verificaciones['idioma']['confianza'] * 0.1               # 10%
        ]

        return sum(scores)

    def _get_nivel_confianza(self, score: float) -> str:
        """Get confidence level description from score."""
        if score >= 0.8:
            return "ALTA"
        elif score >= 0.6:
            return "MEDIA"
        else:
            return "BAJA"


def presentar_respuesta(pregunta: str, respuesta: str, fuentes: List[Dict[str, Any]],
                       verificacion_resultado: Dict[str, Any]) -> str:
    """
    Format verified response for user presentation.

    Args:
        pregunta: User's question
        respuesta: Generated response
        fuentes: Source chunks
        verificacion_resultado: Verification results

    Returns:
        Formatted output string
    """
    verificacion = verificacion_resultado['verificaciones']
    score = verificacion_resultado['score_confianza']
    nivel = verificacion_resultado['nivel_confianza']

    # Determine confidence badge
    if score >= 0.8:
        badge_confianza = "ALTA CONFIANZA"
        indicador = "[OK]"
    elif score >= 0.6:
        badge_confianza = "CONFIANZA MEDIA"
        indicador = "[!]"
    else:
        badge_confianza = "BAJA CONFIANZA - REVISAR"
        indicador = "[!!]"

    # Format output
    output = f"""
{'='*70}
PREGUNTA: {pregunta}
{'='*70}

RESPUESTA:
{respuesta}

{'='*70}
VERIFICACION: {indicador} {badge_confianza}
{'='*70}

Score de Confianza: {score:.2%}

Citaciones:
  - Encontradas: {verificacion['citaciones']['citaciones_encontradas']}
  - Verificadas: {verificacion['citaciones']['citaciones_verificadas']}
  - Tasa de verificacion: {verificacion['citaciones']['tasa_verificacion']:.0%}

Coherencia: {verificacion['coherencia']['nivel']}
{verificacion['coherencia'].get('explicacion', '')}

Idioma: {'[OK] Espanol' if verificacion['idioma']['es_español'] else '[!] No espanol'}

{'='*70}
FUENTES CONSULTADAS:
{'='*70}
"""

    for i, fuente in enumerate(fuentes[:5], 1):
        meta = fuente.get('metadata', {})
        output += f"""
[{i}] {meta.get('documento', meta.get('source', 'Desconocido'))} - {meta.get('articulo', '')}
    {fuente.get('texto', '')[:200]}...
"""

    return output


def verificar_respuesta_simple(respuesta: str, fuentes: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
    """
    Simple verification function without instantiating the full class.

    Args:
        respuesta: Generated response
        fuentes: Source chunks

    Returns:
        Tuple of (is_valid, confidence_score, message)
    """
    verificador = SistemaVerificacion()
    resultado = verificador.verificar_respuesta("", respuesta, fuentes)

    return (
        resultado['aprobada'],
        resultado['score_confianza'],
        resultado['nivel_confianza']
    )


if __name__ == "__main__":
    # Example usage
    print("Testing Verification System...")

    # Sample data
    respuesta_ejemplo = """Segun Basilea III, el ratio minimo de capital CET1 (Common Equity Tier 1)
    es del 4,5% de los activos ponderados por riesgo [CRR Articulo 92].

    Adicionalmente, debe considerarse el colchon de conservacion de capital del 2,5%,
    lo que resulta en un requisito efectivo del 7% en condiciones normales."""

    fuentes_ejemplo = [
        {
            'texto': 'Articulo 92. Las entidades deberan cumplir en todo momento los siguientes requisitos de fondos propios: un ratio de capital de nivel 1 ordinario del 4,5%',
            'metadata': {'documento': 'CRR', 'articulo': 'Articulo 92'}
        }
    ]

    verificador = SistemaVerificacion()
    resultado = verificador.verificar_respuesta(
        "Cual es el ratio minimo CET1?",
        respuesta_ejemplo,
        fuentes_ejemplo
    )

    print(presentar_respuesta(
        "Cual es el ratio minimo CET1?",
        respuesta_ejemplo,
        fuentes_ejemplo,
        resultado
    ))
