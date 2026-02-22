#!/usr/bin/env python3
"""
Export datasets to multiple formats for easy viewing and editing.
Supports: CSV (LibreOffice/Excel), JSON, JSONL
"""

import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FINETUNING_DIR = DATA_DIR / "finetuning"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORT_DIR = DATA_DIR / "exports"

def load_jsonl(filepath):
    """Load JSONL file."""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def export_sql_methodology_to_csv():
    """Export SQL methodology dataset to CSV for LibreOffice/Excel."""
    input_path = FINETUNING_DIR / "sql_methodology_comparison_dataset.jsonl"
    output_path = EXPORT_DIR / "sql_methodology_comparison.csv"

    examples = load_jsonl(input_path)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'system_prompt', 'user_query', 'assistant_response'])

        for i, ex in enumerate(examples, 1):
            messages = ex.get('messages', [])
            system = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user = next((m['content'] for m in messages if m['role'] == 'user'), '')
            assistant = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
            writer.writerow([i, system, user, assistant])

    print(f"✓ Exported: {output_path}")
    return len(examples)

def export_banking_to_csv():
    """Export banking Q&A to CSV."""
    input_path = FINETUNING_DIR / "banking_qa_dataset.jsonl"
    output_path = EXPORT_DIR / "banking_qa.csv"

    examples = load_jsonl(input_path)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'question', 'answer'])

        for i, ex in enumerate(examples, 1):
            messages = ex.get('messages', [])
            user = next((m['content'] for m in messages if m['role'] == 'user'), '')
            assistant = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
            writer.writerow([i, user, assistant])

    print(f"✓ Exported: {output_path}")
    return len(examples)

def export_bank_financials_to_csv():
    """Export bank financial data to CSV for easy viewing."""
    output_path = EXPORT_DIR / "spanish_banks_financials.csv"

    banks = ['santander', 'caixabank', 'bbva', 'sabadell', 'kutxabank']
    years = ['2022', '2023']

    all_data = []
    for bank in banks:
        for year in years:
            filepath = PROCESSED_DIR / bank / f"{bank}_{year}.json"
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metrics = data.get('metricas_financieras', {})
                    row = {
                        'banco': bank.title(),
                        'año': year,
                        'activo_total': metrics.get('activo_total', ''),
                        'beneficio_neto': metrics.get('beneficio_neto', ''),
                        'patrimonio_neto': metrics.get('patrimonio_neto', ''),
                        'creditos_clientes': metrics.get('creditos_clientes', ''),
                        'depositos_clientes': metrics.get('depositos_clientes', ''),
                        'ratio_capital_cet1': metrics.get('ratio_capital', ''),
                        'roe': metrics.get('roe', ''),
                        'tasa_morosidad': metrics.get('morosidad', ''),
                        'resumen': data.get('texto_resumen', '')[:200]
                    }
                    all_data.append(row)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['banco', 'año', 'activo_total', 'beneficio_neto', 'patrimonio_neto',
                      'creditos_clientes', 'depositos_clientes', 'ratio_capital_cet1',
                      'roe', 'tasa_morosidad', 'resumen']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"✓ Exported: {output_path}")
    return len(all_data)

def create_combined_training_json():
    """Create a combined training dataset in JSON format."""
    output_path = EXPORT_DIR / "combined_training_dataset.json"

    combined = {
        "metadata": {
            "description": "Combined training dataset for credit risk LLM",
            "datasets": []
        },
        "examples": []
    }

    # Load SQL methodology
    sql_examples = load_jsonl(FINETUNING_DIR / "sql_methodology_comparison_dataset.jsonl")
    for ex in sql_examples:
        ex['dataset_type'] = 'sql_methodology'
    combined['examples'].extend(sql_examples)
    combined['metadata']['datasets'].append({
        "name": "sql_methodology_comparison",
        "count": len(sql_examples),
        "description": "SQL code review against EBA/IRB methodology"
    })

    # Load banking Q&A
    banking_examples = load_jsonl(FINETUNING_DIR / "banking_qa_dataset.jsonl")
    for ex in banking_examples:
        ex['dataset_type'] = 'banking_qa'
    combined['examples'].extend(banking_examples)
    combined['metadata']['datasets'].append({
        "name": "banking_qa",
        "count": len(banking_examples),
        "description": "Spanish bank financial data Q&A"
    })

    # Load regulation training data
    reg_path = PROCESSED_DIR / "train_data.json"
    if reg_path.exists():
        with open(reg_path, 'r', encoding='utf-8') as f:
            reg_examples = json.load(f)
        for ex in reg_examples:
            ex['dataset_type'] = 'regulation'
        combined['examples'].extend(reg_examples)
        combined['metadata']['datasets'].append({
            "name": "regulation_training",
            "count": len(reg_examples),
            "description": "EBA guidelines and banking regulation Q&A"
        })

    combined['metadata']['total_examples'] = len(combined['examples'])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"✓ Exported: {output_path}")
    return len(combined['examples'])

def main():
    """Export all datasets."""
    print("="*60)
    print("EXPORTING DATASETS")
    print("="*60)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] SQL Methodology Comparison → CSV...")
    n1 = export_sql_methodology_to_csv()

    print("\n[2/4] Banking Q&A → CSV...")
    n2 = export_banking_to_csv()

    print("\n[3/4] Bank Financials → CSV...")
    n3 = export_bank_financials_to_csv()

    print("\n[4/4] Combined Training Dataset → JSON...")
    n4 = create_combined_training_json()

    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"\nFiles created in: {EXPORT_DIR}")
    print(f"  - sql_methodology_comparison.csv ({n1} rows)")
    print(f"  - banking_qa.csv ({n2} rows)")
    print(f"  - spanish_banks_financials.csv ({n3} rows)")
    print(f"  - combined_training_dataset.json ({n4} examples)")
    print("\nTo open in LibreOffice Calc:")
    print(f"  libreoffice --calc {EXPORT_DIR}/*.csv")

if __name__ == '__main__':
    main()
