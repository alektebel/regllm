#!/usr/bin/env python3
"""
Script de utilidades para el dataset de bancos.
Permite visualizar, validar y analizar el dataset generado.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FINETUNING_DIR = DATA_DIR / "finetuning"


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Carga un archivo JSONL."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def analyze_dataset(dataset_path: Path):
    """Analiza el dataset de fine-tuning."""
    print("="*60)
    print("ANÁLISIS DEL DATASET")
    print("="*60)
    
    examples = load_jsonl(dataset_path)
    
    print(f"\nTotal de ejemplos: {len(examples)}")
    
    # Estadísticas por tipo de pregunta
    question_types = defaultdict(int)
    bancos_mentioned = defaultdict(int)
    años_mentioned = defaultdict(int)
    
    for example in examples:
        messages = example.get('messages', [])
        if not messages:
            continue
        
        user_msg = messages[0].get('content', '')
        
        # Detectar tipo de pregunta
        if 'cuánto ganó' in user_msg.lower() or 'beneficio' in user_msg.lower():
            question_types['Beneficios'] += 1
        elif 'balance' in user_msg.lower() or 'activo total' in user_msg.lower():
            question_types['Balance/Activos'] += 1
        elif 'solvencia' in user_msg.lower() or 'capital' in user_msg.lower():
            question_types['Solvencia'] += 1
        elif 'rentabilidad' in user_msg.lower() or 'roe' in user_msg.lower():
            question_types['Rentabilidad'] += 1
        elif 'morosidad' in user_msg.lower():
            question_types['Morosidad'] += 1
        elif 'resumen' in user_msg.lower() or 'desempeño' in user_msg.lower():
            question_types['Resumen'] += 1
        elif 'destaca' in user_msg.lower() or 'aspectos' in user_msg.lower():
            question_types['Aspectos destacados'] += 1
        elif 'compara' in user_msg.lower():
            question_types['Comparativas'] += 1
        else:
            question_types['Generales'] += 1
        
        # Detectar bancos mencionados
        for banco in ['Santander', 'CaixaBank', 'BBVA', 'Sabadell', 'Kutxabank']:
            if banco.lower() in user_msg.lower():
                bancos_mentioned[banco] += 1
        
        # Detectar años
        for año in ['2021', '2022', '2023', '2024']:
            if año in user_msg:
                años_mentioned[año] += 1
    
    print("\n" + "-"*60)
    print("DISTRIBUCIÓN POR TIPO DE PREGUNTA")
    print("-"*60)
    for qtype, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(examples)) * 100
        print(f"  {qtype:25} {count:4} ({percentage:5.1f}%)")
    
    print("\n" + "-"*60)
    print("DISTRIBUCIÓN POR BANCO")
    print("-"*60)
    for banco, count in sorted(bancos_mentioned.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(examples)) * 100
        print(f"  {banco:25} {count:4} ({percentage:5.1f}%)")
    
    print("\n" + "-"*60)
    print("DISTRIBUCIÓN POR AÑO")
    print("-"*60)
    for año, count in sorted(años_mentioned.items()):
        percentage = (count / len(examples)) * 100
        print(f"  {año:25} {count:4} ({percentage:5.1f}%)")
    
    # Calcular longitud promedio
    user_lengths = []
    assistant_lengths = []
    
    for example in examples:
        messages = example.get('messages', [])
        if len(messages) >= 2:
            user_lengths.append(len(messages[0]['content']))
            assistant_lengths.append(len(messages[1]['content']))
    
    print("\n" + "-"*60)
    print("ESTADÍSTICAS DE LONGITUD")
    print("-"*60)
    print(f"  Longitud promedio pregunta:  {sum(user_lengths)/len(user_lengths):.0f} caracteres")
    print(f"  Longitud promedio respuesta: {sum(assistant_lengths)/len(assistant_lengths):.0f} caracteres")
    print(f"  Pregunta más corta:          {min(user_lengths)} caracteres")
    print(f"  Pregunta más larga:          {max(user_lengths)} caracteres")
    print(f"  Respuesta más corta:         {min(assistant_lengths)} caracteres")
    print(f"  Respuesta más larga:         {max(assistant_lengths)} caracteres")


def show_samples(dataset_path: Path, n: int = 5):
    """Muestra ejemplos del dataset."""
    print("\n" + "="*60)
    print(f"MOSTRANDO {n} EJEMPLOS DEL DATASET")
    print("="*60)
    
    examples = load_jsonl(dataset_path)
    
    import random
    samples = random.sample(examples, min(n, len(examples)))
    
    for i, example in enumerate(samples, 1):
        messages = example.get('messages', [])
        if len(messages) >= 2:
            print(f"\n[Ejemplo {i}]")
            print(f"Usuario: {messages[0]['content']}")
            print(f"Asistente: {messages[1]['content']}")
            print("-"*60)


def validate_dataset(dataset_path: Path):
    """Valida el formato del dataset."""
    print("\n" + "="*60)
    print("VALIDACIÓN DEL DATASET")
    print("="*60)
    
    examples = load_jsonl(dataset_path)
    errors = []
    
    for i, example in enumerate(examples):
        # Validar estructura
        if 'messages' not in example:
            errors.append(f"Línea {i+1}: Falta campo 'messages'")
            continue
        
        messages = example['messages']
        
        if not isinstance(messages, list):
            errors.append(f"Línea {i+1}: 'messages' debe ser una lista")
            continue
        
        if len(messages) < 2:
            errors.append(f"Línea {i+1}: Debe haber al menos 2 mensajes")
            continue
        
        # Validar roles
        if messages[0].get('role') != 'user':
            errors.append(f"Línea {i+1}: Primer mensaje debe tener role='user'")
        
        if messages[1].get('role') != 'assistant':
            errors.append(f"Línea {i+1}: Segundo mensaje debe tener role='assistant'")
        
        # Validar contenido
        for j, msg in enumerate(messages):
            if 'content' not in msg:
                errors.append(f"Línea {i+1}, Mensaje {j+1}: Falta campo 'content'")
            elif not msg['content'].strip():
                errors.append(f"Línea {i+1}, Mensaje {j+1}: Contenido vacío")
    
    if errors:
        print("\n⚠️  ERRORES ENCONTRADOS:")
        for error in errors[:10]:  # Mostrar primeros 10 errores
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... y {len(errors) - 10} errores más")
        return False
    else:
        print("\n✓ Dataset válido. No se encontraron errores.")
        return True


def show_consolidated_stats():
    """Muestra estadísticas de los datos consolidados."""
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE DATOS CONSOLIDADOS")
    print("="*60)
    
    consolidated_file = PROCESSED_DIR / 'consolidated_data.json'
    
    if not consolidated_file.exists():
        print("✗ Archivo consolidado no encontrado")
        return
    
    with open(consolidated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nTotal de registros: {len(data)}")
    
    # Estadísticas por banco
    bancos = defaultdict(int)
    años = defaultdict(int)
    metricas_disponibles = defaultdict(int)
    
    for record in data:
        bancos[record.get('banco', 'unknown')] += 1
        años[record.get('año', 'unknown')] += 1
        
        for metrica in record.get('metricas_financieras', {}).keys():
            metricas_disponibles[metrica] += 1
    
    print("\n" + "-"*60)
    print("REGISTROS POR BANCO")
    print("-"*60)
    for banco, count in sorted(bancos.items()):
        print(f"  {banco:20} {count:3} registros")
    
    print("\n" + "-"*60)
    print("REGISTROS POR AÑO")
    print("-"*60)
    for año, count in sorted(años.items()):
        print(f"  {año:20} {count:3} registros")
    
    print("\n" + "-"*60)
    print("MÉTRICAS MÁS COMUNES")
    print("-"*60)
    for metrica, count in sorted(metricas_disponibles.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(data)) * 100
        print(f"  {metrica:30} {count:3} ({percentage:5.1f}%)")


def main():
    """Función principal."""
    dataset_path = FINETUNING_DIR / 'banking_qa_dataset.jsonl'
    
    if not dataset_path.exists():
        print(f"✗ Error: No se encuentra {dataset_path}")
        print("\nEjecuta primero:")
        print("  python3 scripts/generate_example_data.py")
        print("o")
        print("  python3 scripts/download_financial_reports.py")
        return 1
    
    # Mostrar menú si no hay argumentos
    if len(sys.argv) < 2:
        print("="*60)
        print("UTILIDADES PARA DATASET DE BANCOS")
        print("="*60)
        print("\nUso:")
        print("  python3 scripts/dataset_utils.py [comando]")
        print("\nComandos disponibles:")
        print("  analyze    - Analiza el dataset de fine-tuning")
        print("  validate   - Valida el formato del dataset")
        print("  samples    - Muestra ejemplos aleatorios")
        print("  stats      - Estadísticas de datos consolidados")
        print("  all        - Ejecuta todos los análisis")
        print("\nEjemplo:")
        print("  python3 scripts/dataset_utils.py analyze")
        return 0
    
    command = sys.argv[1].lower()
    
    if command == 'analyze':
        analyze_dataset(dataset_path)
    elif command == 'validate':
        validate_dataset(dataset_path)
    elif command == 'samples':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        show_samples(dataset_path, n)
    elif command == 'stats':
        show_consolidated_stats()
    elif command == 'all':
        validate_dataset(dataset_path)
        analyze_dataset(dataset_path)
        show_consolidated_stats()
        show_samples(dataset_path, 3)
    else:
        print(f"✗ Comando desconocido: {command}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
