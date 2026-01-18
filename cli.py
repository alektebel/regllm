#!/usr/bin/env python3
"""
Command Line Interface for Banking Regulatory Assistant.
Interactive mode for querying regulatory information in Spanish.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RegulatoryRAGSystem
from src.verification import SistemaVerificacion, presentar_respuesta

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class RegulatoryCLI:
    """Interactive CLI for regulatory queries."""

    def __init__(self, rag: RegulatoryRAGSystem, verificador: SistemaVerificacion):
        """
        Initialize CLI with RAG system and verifier.

        Args:
            rag: RAG system instance
            verificador: Verification system instance
        """
        self.rag = rag
        self.verificador = verificador
        self.n_fuentes = 5
        self.usar_hibrida = True
        self.mostrar_fuentes = True

    def procesar_pregunta(self, pregunta: str) -> str:
        """
        Process a question and return formatted response.

        Args:
            pregunta: User's question

        Returns:
            Formatted response string
        """
        # Search for relevant sources
        if self.usar_hibrida:
            fuentes = self.rag.buscar_hibrida(pregunta, n_resultados=self.n_fuentes)
        else:
            fuentes = self.rag.buscar_contexto(pregunta, n_resultados=self.n_fuentes)

        if not fuentes:
            return "\n[!] No se encontraron fuentes relevantes para esta pregunta.\n"

        # Generate response from sources
        respuesta = self._generar_respuesta(pregunta, fuentes)

        # Verify response
        resultado = self.verificador.verificar_respuesta(pregunta, respuesta, fuentes)

        # Format output
        return presentar_respuesta(pregunta, respuesta, fuentes, resultado)

    def _generar_respuesta(self, pregunta: str, fuentes: list) -> str:
        """Generate response from sources."""
        if not fuentes:
            return "No se encontro informacion relevante."

        respuesta_parts = []

        for i, fuente in enumerate(fuentes[:3], 1):
            meta = fuente.get('metadata', {})
            texto = fuente.get('texto', '')[:400]
            doc = meta.get('documento', meta.get('source', 'Documento'))
            art = meta.get('articulo', '')

            if art:
                respuesta_parts.append(f"Segun [{doc} - {art}]:\n\"{texto}...\"\n")
            else:
                respuesta_parts.append(f"Segun [{doc}]:\n\"{texto}...\"\n")

        return '\n'.join(respuesta_parts)

    def modo_interactivo(self):
        """Run interactive mode."""
        print("\n" + "="*70)
        print("  ASISTENTE DE CONSULTORÍA EN REGULACIÓN BANCARIA")
        print("  Sistema RAG para consultas sobre regulacion bancaria en espanol")
        print("="*70)
        print(f"\nDocumentos en base de datos: {self.rag.collection.count()}")
        print("\nComandos especiales:")
        print("  /salir, /exit, /quit  - Salir del programa")
        print("  /fuentes N            - Cambiar numero de fuentes (actual: {})".format(self.n_fuentes))
        print("  /hibrida on|off       - Activar/desactivar busqueda hibrida")
        print("  /stats                - Mostrar estadisticas del sistema")
        print("  /help                 - Mostrar esta ayuda")
        print("\n" + "-"*70 + "\n")

        while True:
            try:
                pregunta = input("Pregunta: ").strip()

                if not pregunta:
                    continue

                # Handle commands
                if pregunta.startswith('/'):
                    if self._handle_command(pregunta):
                        continue
                    else:
                        break

                # Process question
                print("\nBuscando informacion...\n")
                resultado = self.procesar_pregunta(pregunta)
                print(resultado)
                print("\n" + "-"*70 + "\n")

            except KeyboardInterrupt:
                print("\n\n¡Hasta luego!")
                break
            except EOFError:
                print("\n\n¡Hasta luego!")
                break

    def _handle_command(self, comando: str) -> bool:
        """
        Handle CLI commands.

        Args:
            comando: Command string

        Returns:
            True to continue, False to exit
        """
        cmd = comando.lower().split()

        if cmd[0] in ['/salir', '/exit', '/quit']:
            print("\n¡Hasta luego!")
            return False

        elif cmd[0] == '/fuentes':
            if len(cmd) > 1:
                try:
                    n = int(cmd[1])
                    if 1 <= n <= 20:
                        self.n_fuentes = n
                        print(f"Numero de fuentes cambiado a {n}")
                    else:
                        print("El numero debe estar entre 1 y 20")
                except ValueError:
                    print("Uso: /fuentes <numero>")
            else:
                print(f"Fuentes actuales: {self.n_fuentes}")

        elif cmd[0] == '/hibrida':
            if len(cmd) > 1:
                if cmd[1] == 'on':
                    self.usar_hibrida = True
                    print("Busqueda hibrida activada")
                elif cmd[1] == 'off':
                    self.usar_hibrida = False
                    print("Busqueda hibrida desactivada (solo semantica)")
            else:
                estado = "activada" if self.usar_hibrida else "desactivada"
                print(f"Busqueda hibrida: {estado}")

        elif cmd[0] == '/stats':
            stats = self.rag.get_stats()
            print("\n--- Estadisticas del Sistema ---")
            print(f"Total documentos: {stats['total_documents']}")
            print(f"Modelo embeddings: {stats['embedding_model']}")
            print(f"BM25 indexado: {'Si' if stats['bm25_indexed'] else 'No'}")
            print(f"Tamano corpus: {stats['corpus_size']}")

        elif cmd[0] == '/help':
            print("\nComandos disponibles:")
            print("  /salir, /exit, /quit  - Salir del programa")
            print("  /fuentes N            - Cambiar numero de fuentes")
            print("  /hibrida on|off       - Activar/desactivar busqueda hibrida")
            print("  /stats                - Mostrar estadisticas")
            print("  /help                 - Mostrar esta ayuda")

        else:
            print(f"Comando desconocido: {cmd[0]}. Usa /help para ver comandos.")

        return True


def run_single_query(rag: RegulatoryRAGSystem, pregunta: str, n_fuentes: int = 5):
    """Run a single query and print result."""
    verificador = SistemaVerificacion()

    # Search
    fuentes = rag.buscar_hibrida(pregunta, n_resultados=n_fuentes)

    if not fuentes:
        print("\nNo se encontraron fuentes relevantes.")
        return

    # Generate response
    respuesta_parts = []
    for fuente in fuentes[:3]:
        meta = fuente.get('metadata', {})
        texto = fuente.get('texto', '')[:400]
        doc = meta.get('documento', meta.get('source', 'Doc'))
        art = meta.get('articulo', '')

        if art:
            respuesta_parts.append(f"[{doc} - {art}]: \"{texto}...\"")
        else:
            respuesta_parts.append(f"[{doc}]: \"{texto}...\"")

    respuesta = '\n'.join(respuesta_parts)

    # Verify and present
    resultado = verificador.verificar_respuesta(pregunta, respuesta, fuentes)
    print(presentar_respuesta(pregunta, respuesta, fuentes, resultado))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Asistente de Consultoria en Regulacion Bancaria (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s --interactive          Modo interactivo
  %(prog)s "Que es el capital CET1?"    Consulta unica
  %(prog)s --load-json data.json  Cargar datos antes de consultar

Para mas ayuda, visita la documentacion del proyecto.
        """
    )

    parser.add_argument(
        'pregunta',
        type=str,
        nargs='?',
        help='Pregunta sobre regulacion bancaria (opcional si se usa --interactive)'
    )

    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Modo interactivo'
    )

    parser.add_argument(
        '-f', '--fuentes',
        type=int,
        default=5,
        help='Numero de fuentes a recuperar (default: 5)'
    )

    parser.add_argument(
        '--load-json',
        type=str,
        help='Cargar documentos desde archivo JSON'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='./vector_db/chroma_db',
        help='Ruta a la base de datos vectorial'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostrar logs detallados'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Initialize RAG system
    print("Inicializando sistema RAG...")
    rag = RegulatoryRAGSystem(persist_directory=args.db_path)

    # Load JSON if provided
    if args.load_json:
        print(f"Cargando datos desde {args.load_json}...")
        chunks = rag.load_from_json(args.load_json)
        print(f"Cargados {chunks} chunks")

    # Auto-load most recent JSON if database is empty
    if rag.collection.count() == 0:
        import glob
        json_files = sorted(glob.glob("data/raw/*.json"), reverse=True)
        if json_files:
            print(f"Base de datos vacia. Cargando {json_files[0]}...")
            rag.load_from_json(json_files[0])

    print(f"Sistema listo. {rag.collection.count()} documentos en la base de datos.")

    # Run mode
    if args.interactive:
        verificador = SistemaVerificacion()
        cli = RegulatoryCLI(rag, verificador)
        cli.n_fuentes = args.fuentes
        cli.modo_interactivo()

    elif args.pregunta:
        run_single_query(rag, args.pregunta, args.fuentes)

    else:
        # Default to interactive if no question provided
        verificador = SistemaVerificacion()
        cli = RegulatoryCLI(rag, verificador)
        cli.n_fuentes = args.fuentes
        cli.modo_interactivo()


if __name__ == '__main__':
    main()
