#!/usr/bin/env python3
"""
Script para descargar y procesar cuentas anuales de bancos españoles.
Extrae información financiera y la guarda en formato JSON.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from urllib.parse import urlparse
import time

# Configuración
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINETUNING_DIR = DATA_DIR / "finetuning"
BANKS_URLS_FILE = DATA_DIR / "banks_urls.json"


class FinancialReportDownloader:
    """Descargador y procesador de informes financieros."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        
    def download_pdf(self, url: str, output_path: Path) -> bool:
        """Descarga un PDF desde una URL."""
        try:
            print(f"Descargando: {url}")
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Verificar que es un PDF
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
                print(f"⚠️  Advertencia: El contenido puede no ser un PDF ({content_type})")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
            print(f"✓ Descargado: {output_path.name}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error descargando {url}: {e}")
            return False
    
    def download_all_reports(self, banks_data: Dict[str, Any]) -> Dict[str, List[Path]]:
        """Descarga todos los informes de los bancos."""
        downloaded_files = {}
        
        for bank_id, bank_info in banks_data['bancos'].items():
            print(f"\n{'='*60}")
            print(f"Banco: {bank_info['nombre_completo']}")
            print(f"{'='*60}")
            
            bank_files = []
            for year, report_info in bank_info['cuentas_anuales'].items():
                url = report_info['url']
                filename = f"{bank_id}_{year}.pdf"
                output_path = RAW_DIR / bank_id / filename
                
                if output_path.exists():
                    print(f"⊙ Ya existe: {filename}")
                    bank_files.append(output_path)
                else:
                    if self.download_pdf(url, output_path):
                        bank_files.append(output_path)
                    time.sleep(2)  # Pausa entre descargas
            
            downloaded_files[bank_id] = bank_files
        
        return downloaded_files


class FinancialDataExtractor:
    """Extractor de datos financieros desde PDFs."""
    
    # Patrones comunes para extraer métricas financieras
    METRICS_PATTERNS = {
        'activo_total': r'(?:activo\s+total|total\s+activ[oa]s?)[:\s]+([0-9.,]+)',
        'patrimonio_neto': r'(?:patrimonio\s+neto|fondos\s+propios)[:\s]+([0-9.,]+)',
        'beneficio_neto': r'(?:beneficio\s+neto|resultado\s+neto)[:\s]+([0-9.,]+)',
        'ingresos': r'(?:ingresos?\s+totales?|margen\s+de\s+intereses)[:\s]+([0-9.,]+)',
        'creditos_clientes': r'(?:cr[ée]ditos?\s+a\s+clientes?|pr[ée]stamos)[:\s]+([0-9.,]+)',
        'depositos_clientes': r'(?:dep[óo]sitos?\s+de\s+clientes?)[:\s]+([0-9.,]+)',
        'ratio_capital': r'(?:ratio\s+de\s+capital|CET1)[:\s]+([0-9.,]+)',
        'roa': r'(?:ROA|rentabilidad\s+sobre\s+activos)[:\s]+([0-9.,]+)',
        'roe': r'(?:ROE|rentabilidad\s+sobre\s+recursos\s+propios)[:\s]+([0-9.,]+)',
        'morosidad': r'(?:ratio\s+de\s+morosidad|mora)[:\s]+([0-9.,]+)',
    }
    
    def extract_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extrae datos financieros de un PDF."""
        try:
            # Intentar con PyPDF2 primero
            try:
                import PyPDF2
                return self._extract_with_pypdf2(pdf_path)
            except ImportError:
                pass
            
            # Intentar con pdfplumber
            try:
                import pdfplumber
                return self._extract_with_pdfplumber(pdf_path)
            except ImportError:
                pass
            
            # Si no hay librerías disponibles, usar pdftotext
            return self._extract_with_pdftotext(pdf_path)
            
        except Exception as e:
            print(f"Error extrayendo datos de {pdf_path.name}: {e}")
            return self._create_empty_structure(pdf_path)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Dict[str, Any]:
        """Extrae texto usando PyPDF2."""
        import PyPDF2
        
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Extraer primeras 50 páginas (donde suelen estar los datos clave)
            for page_num in range(min(50, len(reader.pages))):
                text += reader.pages[page_num].extract_text()
        
        return self._parse_financial_data(text, pdf_path)
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """Extrae texto usando pdfplumber."""
        import pdfplumber
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            # Extraer primeras 50 páginas
            for page in pdf.pages[:50]:
                text += page.extract_text() or ""
        
        return self._parse_financial_data(text, pdf_path)
    
    def _extract_with_pdftotext(self, pdf_path: Path) -> Dict[str, Any]:
        """Extrae texto usando pdftotext (poppler-utils)."""
        import subprocess
        
        try:
            result = subprocess.run(
                ['pdftotext', '-l', '50', str(pdf_path), '-'],
                capture_output=True,
                text=True,
                timeout=120
            )
            text = result.stdout
            return self._parse_financial_data(text, pdf_path)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Error ejecutando pdftotext: {e}")
            return self._create_empty_structure(pdf_path)
    
    def _parse_financial_data(self, text: str, pdf_path: Path) -> Dict[str, Any]:
        """Parsea el texto para extraer datos financieros."""
        text_lower = text.lower()
        
        # Extraer año del nombre del archivo
        year_match = re.search(r'_(\d{4})\.pdf$', pdf_path.name)
        year = year_match.group(1) if year_match else "unknown"
        
        # Extraer banco del nombre del archivo
        bank_match = re.match(r'^([^_]+)_', pdf_path.name)
        bank_id = bank_match.group(1) if bank_match else "unknown"
        
        data = {
            'banco': bank_id,
            'año': year,
            'archivo_fuente': pdf_path.name,
            'metricas_financieras': {},
            'texto_resumen': self._extract_summary(text),
            'datos_clave': []
        }
        
        # Extraer métricas usando patrones
        for metric_name, pattern in self.METRICS_PATTERNS.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Tomar el primer valor encontrado y limpiarlo
                value = self._clean_numeric_value(matches[0])
                data['metricas_financieras'][metric_name] = value
        
        # Extraer tablas y secciones clave
        data['datos_clave'] = self._extract_key_sections(text)
        
        return data
    
    def _extract_summary(self, text: str, max_chars: int = 2000) -> str:
        """Extrae un resumen del inicio del documento."""
        # Buscar sección de resumen o ejecutivo
        summary_patterns = [
            r'(?:resumen\s+ejecutivo|executive\s+summary)(.*?)(?:\n\n|\r\n\r\n)',
            r'(?:cifras\s+clave|key\s+figures)(.*?)(?:\n\n|\r\n\r\n)',
            r'(?:principales\s+magnitudes)(.*?)(?:\n\n|\r\n\r\n)',
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:max_chars]
        
        # Si no se encuentra, devolver los primeros caracteres
        return text[:max_chars].strip()
    
    def _extract_key_sections(self, text: str) -> List[str]:
        """Extrae secciones clave del documento."""
        sections = []
        
        # Patrones de secciones importantes
        section_patterns = [
            r'(?:balance\s+de\s+situaci[óo]n)(.*?)(?=\n\s*\n|\Z)',
            r'(?:cuenta\s+de\s+p[ée]rdidas\s+y\s+ganancias)(.*?)(?=\n\s*\n|\Z)',
            r'(?:estado\s+de\s+flujos\s+de\s+efectivo)(.*?)(?=\n\s*\n|\Z)',
            r'(?:ratio\s+de\s+solvencia)(.*?)(?=\n\s*\n|\Z)',
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches[:1]:  # Solo primera coincidencia
                cleaned = re.sub(r'\s+', ' ', match.strip())[:500]
                if len(cleaned) > 50:
                    sections.append(cleaned)
        
        return sections
    
    def _clean_numeric_value(self, value: str) -> str:
        """Limpia valores numéricos."""
        # Eliminar espacios y normalizar separadores
        cleaned = value.strip().replace(' ', '').replace('\xa0', '')
        return cleaned
    
    def _create_empty_structure(self, pdf_path: Path) -> Dict[str, Any]:
        """Crea estructura vacía cuando no se puede extraer datos."""
        year_match = re.search(r'_(\d{4})\.pdf$', pdf_path.name)
        year = year_match.group(1) if year_match else "unknown"
        
        bank_match = re.match(r'^([^_]+)_', pdf_path.name)
        bank_id = bank_match.group(1) if bank_match else "unknown"
        
        return {
            'banco': bank_id,
            'año': year,
            'archivo_fuente': pdf_path.name,
            'metricas_financieras': {},
            'texto_resumen': '',
            'datos_clave': [],
            'nota': 'No se pudieron extraer datos automáticamente. Requiere procesamiento manual.'
        }


class FineTuningDatasetGenerator:
    """Generador de datasets para fine-tuning de LLMs."""
    
    def generate_training_data(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Genera datos de entrenamiento en formato conversacional."""
        training_examples = []
        
        for record in processed_data:
            banco = record.get('banco', 'unknown')
            año = record.get('año', 'unknown')
            metricas = record.get('metricas_financieras', {})
            
            # Generar múltiples ejemplos de preguntas y respuestas
            examples = self._generate_qa_pairs(banco, año, metricas, record)
            training_examples.extend(examples)
        
        return training_examples
    
    def _generate_qa_pairs(self, banco: str, año: str, metricas: Dict[str, Any], 
                           record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera pares de preguntas y respuestas."""
        pairs = []
        
        bank_names = {
            'santander': 'Banco Santander',
            'caixabank': 'CaixaBank',
            'bbva': 'BBVA',
            'sabadell': 'Banco Sabadell',
            'kutxabank': 'Kutxabank'
        }
        
        bank_display = bank_names.get(banco, banco.title())
        
        # Ejemplo 1: Pregunta general sobre el año
        if metricas:
            metrics_text = ", ".join([
                f"{k.replace('_', ' ')}: {v}" 
                for k, v in list(metricas.items())[:5]
            ])
            
            pairs.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"¿Cuáles fueron las principales métricas financieras de {bank_display} en {año}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"Las principales métricas financieras de {bank_display} en {año} fueron: {metrics_text}."
                    }
                ]
            })
        
        # Ejemplo 2: Preguntas específicas por métrica
        metric_questions = {
            'activo_total': f"¿Cuál fue el activo total de {bank_display} en {año}?",
            'beneficio_neto': f"¿Qué beneficio neto obtuvo {bank_display} en {año}?",
            'patrimonio_neto': f"¿Cuál fue el patrimonio neto de {bank_display} en {año}?",
            'ratio_capital': f"¿Cuál fue el ratio de capital de {bank_display} en {año}?",
            'morosidad': f"¿Cuál fue la tasa de morosidad de {bank_display} en {año}?",
        }
        
        for metric, question in metric_questions.items():
            if metric in metricas:
                value = metricas[metric]
                pairs.append({
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        },
                        {
                            "role": "assistant",
                            "content": f"El {metric.replace('_', ' ')} de {bank_display} en {año} fue de {value}."
                        }
                    ]
                })
        
        # Ejemplo 3: Comparativa (si hay datos suficientes)
        if 'activo_total' in metricas and 'beneficio_neto' in metricas:
            pairs.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"Compara el activo total y el beneficio neto de {bank_display} en {año}"
                    },
                    {
                        "role": "assistant",
                        "content": f"En {año}, {bank_display} tuvo un activo total de {metricas['activo_total']} y un beneficio neto de {metricas['beneficio_neto']}."
                    }
                ]
            })
        
        # Ejemplo 4: Resumen si existe
        if record.get('texto_resumen'):
            summary = record['texto_resumen'][:500]
            pairs.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"Dame un resumen del informe anual de {bank_display} de {año}"
                    },
                    {
                        "role": "assistant",
                        "content": f"Resumen del informe anual {año} de {bank_display}: {summary}"
                    }
                ]
            })
        
        return pairs
    
    def save_jsonl(self, training_data: List[Dict[str, Any]], output_path: Path):
        """Guarda los datos en formato JSONL para fine-tuning."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✓ Dataset guardado: {output_path}")
        print(f"  Total de ejemplos: {len(training_data)}")


def main():
    """Función principal."""
    print("="*60)
    print("DESCARGADOR DE CUENTAS ANUALES - BANCOS ESPAÑOLES")
    print("="*60)
    
    # Cargar URLs de bancos
    if not BANKS_URLS_FILE.exists():
        print(f"✗ Error: No se encuentra {BANKS_URLS_FILE}")
        return 1
    
    with open(BANKS_URLS_FILE, 'r', encoding='utf-8') as f:
        banks_data = json.load(f)
    
    # Paso 1: Descargar PDFs
    print("\n[1/4] DESCARGANDO INFORMES...")
    downloader = FinancialReportDownloader()
    downloaded_files = downloader.download_all_reports(banks_data)
    
    # Paso 2: Extraer datos
    print("\n[2/4] EXTRAYENDO DATOS FINANCIEROS...")
    extractor = FinancialDataExtractor()
    all_extracted_data = []
    
    for bank_id, files in downloaded_files.items():
        for pdf_file in files:
            print(f"Procesando: {pdf_file.name}")
            data = extractor.extract_from_pdf(pdf_file)
            all_extracted_data.append(data)
            
            # Guardar JSON individual
            json_filename = pdf_file.stem + '.json'
            json_path = PROCESSED_DIR / bank_id / json_filename
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Guardado: {json_path.name}")
    
    # Paso 3: Consolidar datos
    print("\n[3/4] CONSOLIDANDO DATOS...")
    consolidated_path = PROCESSED_DIR / 'consolidated_data.json'
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Datos consolidados guardados en: {consolidated_path}")
    
    # Paso 4: Generar dataset para fine-tuning
    print("\n[4/4] GENERANDO DATASET PARA FINE-TUNING...")
    generator = FineTuningDatasetGenerator()
    training_data = generator.generate_training_data(all_extracted_data)
    
    output_jsonl = FINETUNING_DIR / 'banking_qa_dataset.jsonl'
    generator.save_jsonl(training_data, output_jsonl)
    
    print("\n" + "="*60)
    print("✓ PROCESO COMPLETADO")
    print("="*60)
    print(f"Archivos procesados: {len(all_extracted_data)}")
    print(f"Ejemplos de entrenamiento: {len(training_data)}")
    print(f"\nDataset para fine-tuning: {output_jsonl}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
