#!/usr/bin/env python3
"""
Gradio Web Interface for Banking Regulatory Assistant.
Interactive web UI for querying regulatory information in Spanish.
"""

import gradio as gr
import sys
import glob
import logging
from pathlib import Path
from typing import Tuple, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RegulatoryRAGSystem
from src.verification import SistemaVerificacion

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global instances (initialized at startup)
rag_system = None
verificador = None


def initialize_system():
    """Initialize the RAG system and verifier."""
    global rag_system, verificador

    logger.info("Initializing RAG system...")
    rag_system = RegulatoryRAGSystem()
    verificador = SistemaVerificacion()

    # Auto-load most recent JSON if database is empty
    if rag_system.collection.count() == 0:
        json_files = sorted(glob.glob("data/raw/*.json"), reverse=True)
        if json_files:
            logger.info(f"Loading data from {json_files[0]}...")
            rag_system.load_from_json(json_files[0])

    logger.info(f"System initialized with {rag_system.collection.count()} documents")
    return rag_system.collection.count()


def responder_consulta(pregunta: str, num_fuentes: int, usar_hibrida: bool) -> Tuple[str, str, str]:
    """
    Process a question and return response, verification, and sources.

    Args:
        pregunta: User's question
        num_fuentes: Number of sources to retrieve
        usar_hibrida: Whether to use hybrid search

    Returns:
        Tuple of (response, verification info, sources)
    """
    if not pregunta.strip():
        return "Por favor, ingresa una pregunta.", "", ""

    if not rag_system:
        return "Sistema no inicializado. Recarga la pagina.", "", ""

    try:
        # Search for relevant sources
        if usar_hibrida:
            fuentes = rag_system.buscar_hibrida(pregunta, n_resultados=num_fuentes)
        else:
            fuentes = rag_system.buscar_contexto(pregunta, n_resultados=num_fuentes)

        if not fuentes:
            return (
                "No se encontraron fuentes relevantes para esta pregunta.",
                "**Estado:** Sin resultados",
                ""
            )

        # Generate response from sources
        respuesta = _generar_respuesta(pregunta, fuentes)

        # Verify response
        resultado = verificador.verificar_respuesta(pregunta, respuesta, fuentes)

        # Format verification
        verif = resultado['verificaciones']
        score = resultado['score_confianza']

        if score >= 0.8:
            badge = "[OK] ALTA CONFIANZA"
        elif score >= 0.6:
            badge = "[!] CONFIANZA MEDIA"
        else:
            badge = "[!!] BAJA CONFIANZA"

        verificacion_texto = f"""
**{badge}**

**Score de Confianza:** {score:.0%}

**Citaciones:**
- Encontradas: {verif['citaciones']['citaciones_encontradas']}
- Verificadas: {verif['citaciones']['citaciones_verificadas']}
- Tasa: {verif['citaciones']['tasa_verificacion']:.0%}

**Coherencia:** {verif['coherencia']['nivel']}

**Idioma:** {'[OK] Espanol' if verif['idioma']['es_español'] else '[!] No espanol'}
"""

        # Format sources
        fuentes_texto = ""
        for i, fuente in enumerate(fuentes[:num_fuentes], 1):
            meta = fuente.get('metadata', {})
            doc = meta.get('documento', meta.get('source', 'Desconocido'))
            art = meta.get('articulo', '')
            texto = fuente.get('texto', '')[:400]

            fuentes_texto += f"""
**[{i}] {doc}** {f'- {art}' if art else ''}

{texto}...

---
"""

        return respuesta, verificacion_texto, fuentes_texto

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error procesando la consulta: {str(e)}", "", ""


def _generar_respuesta(pregunta: str, fuentes: list) -> str:
    """Generate response from sources."""
    if not fuentes:
        return "No se encontro informacion relevante."

    respuesta_parts = [
        "Basandome en la documentacion regulatoria disponible:\n"
    ]

    for i, fuente in enumerate(fuentes[:3], 1):
        meta = fuente.get('metadata', {})
        texto = fuente.get('texto', '')[:500]
        doc = meta.get('documento', meta.get('source', 'Documento'))
        art = meta.get('articulo', '')

        if art:
            respuesta_parts.append(f"\n**Segun [{doc} - {art}]:**\n\n> \"{texto}...\"\n")
        else:
            respuesta_parts.append(f"\n**Segun [{doc}]:**\n\n> \"{texto}...\"\n")

    return ''.join(respuesta_parts)


def cargar_documentos(json_path: str) -> str:
    """Load documents from JSON file."""
    if not rag_system:
        return "Sistema no inicializado"

    try:
        chunks = rag_system.load_from_json(json_path)
        return f"Cargados {chunks} chunks. Total en BD: {rag_system.collection.count()}"
    except FileNotFoundError:
        return f"Archivo no encontrado: {json_path}"
    except Exception as e:
        return f"Error: {str(e)}"


def obtener_stats() -> str:
    """Get system statistics."""
    if not rag_system:
        return "Sistema no inicializado"

    stats = rag_system.get_stats()
    return f"""
**Estadisticas del Sistema**

- Total documentos: {stats['total_documents']}
- Modelo embeddings: {stats['embedding_model']}
- BM25 indexado: {'Si' if stats['bm25_indexed'] else 'No'}
- Tamano corpus: {stats['corpus_size']}
"""


def limpiar_bd() -> str:
    """Clear the vector database."""
    if not rag_system:
        return "Sistema no inicializado"

    rag_system.clear_collection()
    return "Base de datos limpiada. Total documentos: 0"


def crear_interfaz():
    """Create the Gradio interface."""
    # Initialize system
    doc_count = initialize_system()

    # Custom CSS
    custom_css = """
    .main-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .confidence-high { color: green; }
    .confidence-medium { color: orange; }
    .confidence-low { color: red; }
    """

    with gr.Blocks(title="Asistente Regulatorio Bancario", css=custom_css, theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown("""
        # Asistente de Regulacion Bancaria

        Sistema especializado en Basilea III, parametros de riesgo de credito y normativa prudencial.
        **Todas las respuestas estan basadas en documentacion regulatoria oficial.**
        """)

        gr.Markdown(f"*Sistema inicializado con {doc_count} documentos*")

        with gr.Tabs():
            # Main query tab
            with gr.Tab("Consultas"):
                with gr.Row():
                    with gr.Column(scale=2):
                        pregunta_input = gr.Textbox(
                            label="Tu pregunta sobre regulacion bancaria",
                            placeholder="Ej: Cual es el ratio minimo de capital CET1?",
                            lines=3
                        )

                        with gr.Row():
                            num_fuentes = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Numero de fuentes"
                            )

                            usar_hibrida = gr.Checkbox(
                                value=True,
                                label="Busqueda hibrida (semantica + keywords)"
                            )

                        submit_btn = gr.Button("Consultar", variant="primary")

                with gr.Row():
                    with gr.Column():
                        respuesta_output = gr.Markdown(label="Respuesta")

                with gr.Row():
                    with gr.Column():
                        verificacion_output = gr.Markdown(label="Verificacion")

                    with gr.Column():
                        fuentes_output = gr.Markdown(label="Fuentes")

                # Examples
                gr.Examples(
                    examples=[
                        ["Que es el capital CET1?"],
                        ["Como se calcula el ratio de apalancamiento?"],
                        ["Que metodologias existen para estimar la PD?"],
                        ["Cual es el tratamiento de las exposiciones hipotecarias?"],
                        ["Que es la LGD y como se calcula?"],
                        ["Cuales son los requisitos de capital de Basilea III?"],
                    ],
                    inputs=pregunta_input
                )

                # Connect submit button
                submit_btn.click(
                    fn=responder_consulta,
                    inputs=[pregunta_input, num_fuentes, usar_hibrida],
                    outputs=[respuesta_output, verificacion_output, fuentes_output]
                )

                # Also submit on Enter
                pregunta_input.submit(
                    fn=responder_consulta,
                    inputs=[pregunta_input, num_fuentes, usar_hibrida],
                    outputs=[respuesta_output, verificacion_output, fuentes_output]
                )

            # Administration tab
            with gr.Tab("Administracion"):
                gr.Markdown("### Gestion de Documentos")

                with gr.Row():
                    with gr.Column():
                        json_path_input = gr.Textbox(
                            label="Ruta al archivo JSON",
                            placeholder="data/raw/regulation_data_XXXXXX.json"
                        )
                        cargar_btn = gr.Button("Cargar JSON")
                        cargar_output = gr.Textbox(label="Resultado", interactive=False)

                        cargar_btn.click(
                            fn=cargar_documentos,
                            inputs=[json_path_input],
                            outputs=[cargar_output]
                        )

                    with gr.Column():
                        stats_btn = gr.Button("Ver Estadisticas")
                        stats_output = gr.Markdown(label="Estadisticas")

                        stats_btn.click(
                            fn=obtener_stats,
                            outputs=[stats_output]
                        )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Zona de Peligro")
                        limpiar_btn = gr.Button("Limpiar Base de Datos", variant="stop")
                        limpiar_output = gr.Textbox(label="Resultado", interactive=False)

                        limpiar_btn.click(
                            fn=limpiar_bd,
                            outputs=[limpiar_output]
                        )

            # About tab
            with gr.Tab("Acerca de"):
                gr.Markdown("""
                ## Asistente de Regulacion Bancaria

                Este sistema utiliza tecnologia RAG (Retrieval-Augmented Generation) para proporcionar
                informacion precisa sobre regulacion bancaria espanola y europea.

                ### Caracteristicas

                - **Busqueda Semantica**: Encuentra informacion relevante usando embeddings multilingues
                - **Busqueda Hibrida**: Combina busqueda semantica con keywords (BM25)
                - **Verificacion Automatica**: Valida citaciones y coherencia de respuestas
                - **Fuentes Trazables**: Todas las respuestas incluyen referencias a documentos originales

                ### Fuentes de Datos

                - Reglamento CRR (UE) 575/2013
                - Directivas EBA sobre PD y LGD
                - Normativa del Banco de Espana
                - Documentos del Comite de Basilea
                - BOE (Boletin Oficial del Estado)

                ### Tecnologia

                - **Embeddings**: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
                - **Vector DB**: ChromaDB
                - **Busqueda**: Hibrida (Semantica + BM25)

                ### Limitaciones

                - Las respuestas se basan unicamente en los documentos cargados
                - Siempre verificar informacion critica con las fuentes originales
                - Este sistema es una herramienta de apoyo, no reemplaza el criterio experto

                ---

                **Version**: 1.0.0
                """)

        return demo


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Asistente Regulatorio - Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--server-name", default="127.0.0.1", help="Server name")

    args = parser.parse_args()

    demo = crear_interfaz()

    print(f"\nIniciando servidor web en http://{args.server_name}:{args.port}")
    if args.share:
        print("Creando enlace publico...")

    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
