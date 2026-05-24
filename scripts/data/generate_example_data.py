#!/usr/bin/env python3
"""
Generador de datos de ejemplo para probar el sistema de fine-tuning.
Crea datos sintéticos basados en información pública de los bancos.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FINETUNING_DIR = DATA_DIR / "finetuning"

# Datos de ejemplo basados en información pública
EXAMPLE_DATA = [
    {
        "banco": "santander",
        "año": "2023",
        "archivo_fuente": "santander_2023_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "1.720.000 millones EUR",
            "beneficio_neto": "11.076 millones EUR",
            "patrimonio_neto": "127.213 millones EUR",
            "creditos_clientes": "985.320 millones EUR",
            "depositos_clientes": "953.284 millones EUR",
            "ratio_capital": "12.8%",
            "roe": "14.8%",
            "morosidad": "3.14%"
        },
        "texto_resumen": "Banco Santander presentó resultados sólidos en 2023, con un beneficio neto atribuido de 11.076 millones de euros. El banco mantuvo una posición de capital robusta con un ratio CET1 de 12.8%, reflejando su solidez financiera. La transformación digital continuó siendo una prioridad estratégica.",
        "datos_clave": [
            "Total de clientes: 164 millones",
            "Presencia en 10 mercados principales",
            "Inversión en tecnología: 5.800 millones EUR",
            "Ratio de eficiencia: 43.7%"
        ]
    },
    {
        "banco": "santander",
        "año": "2022",
        "archivo_fuente": "santander_2022_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "1.661.486 millones EUR",
            "beneficio_neto": "9.605 millones EUR",
            "patrimonio_neto": "122.347 millones EUR",
            "creditos_clientes": "951.627 millones EUR",
            "depositos_clientes": "918.357 millones EUR",
            "ratio_capital": "12.2%",
            "roe": "13.2%",
            "morosidad": "3.21%"
        },
        "texto_resumen": "En 2022, Banco Santander logró un beneficio neto de 9.605 millones de euros, mostrando resiliencia en un entorno macroeconómico complejo. El banco fortaleció su posición de capital y continuó invirtiendo en su transformación digital.",
        "datos_clave": [
            "Total de clientes: 160 millones",
            "Lanzamiento de nuevas plataformas digitales",
            "Inversión en sostenibilidad: finanzas verdes"
        ]
    },
    {
        "banco": "caixabank",
        "año": "2023",
        "archivo_fuente": "caixabank_2023_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "696.414 millones EUR",
            "beneficio_neto": "5.787 millones EUR",
            "patrimonio_neto": "50.982 millones EUR",
            "creditos_clientes": "405.723 millones EUR",
            "depositos_clientes": "455.816 millones EUR",
            "ratio_capital": "13.1%",
            "roe": "12.5%",
            "morosidad": "2.81%"
        },
        "texto_resumen": "CaixaBank cerró 2023 con un beneficio neto de 5.787 millones de euros, consolidando su posición como líder bancario en España. El banco ha completado con éxito la integración de Bankia, generando importantes sinergias.",
        "datos_clave": [
            "Líder en España con 20 millones de clientes",
            "Red de 4.500 oficinas",
            "Sinergias de integración Bankia completadas",
            "Ratio de eficiencia: 47.2%"
        ]
    },
    {
        "banco": "caixabank",
        "año": "2022",
        "archivo_fuente": "caixabank_2022_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "683.596 millones EUR",
            "beneficio_neto": "4.812 millones EUR",
            "patrimonio_neto": "48.356 millones EUR",
            "creditos_clientes": "391.254 millones EUR",
            "depositos_clientes": "442.187 millones EUR",
            "ratio_capital": "12.8%",
            "roe": "11.3%",
            "morosidad": "3.02%"
        },
        "texto_resumen": "En 2022, CaixaBank obtuvo un beneficio neto de 4.812 millones de euros, avanzando en la integración con Bankia y fortaleciendo su liderazgo en el mercado español.",
        "datos_clave": [
            "Proceso de integración Bankia en marcha",
            "Expansión de servicios digitales",
            "Compromiso con la banca sostenible"
        ]
    },
    {
        "banco": "bbva",
        "año": "2023",
        "archivo_fuente": "bbva_2023_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "744.200 millones EUR",
            "beneficio_neto": "8.019 millones EUR",
            "patrimonio_neto": "60.437 millones EUR",
            "creditos_clientes": "430.542 millones EUR",
            "depositos_clientes": "476.829 millones EUR",
            "ratio_capital": "13.4%",
            "roe": "15.2%",
            "morosidad": "3.45%"
        },
        "texto_resumen": "BBVA alcanzó un beneficio neto de 8.019 millones de euros en 2023, destacando por su rentabilidad y eficiencia. El banco continuó su transformación digital y expandió sus servicios en mercados clave como España, México y Turquía.",
        "datos_clave": [
            "Presencia en más de 25 países",
            "88 millones de clientes",
            "Líder en banca digital",
            "Ratio de eficiencia: 44.5%"
        ]
    },
    {
        "banco": "bbva",
        "año": "2022",
        "archivo_fuente": "bbva_2022_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "714.628 millones EUR",
            "beneficio_neto": "6.452 millones EUR",
            "patrimonio_neto": "57.821 millones EUR",
            "creditos_clientes": "412.395 millones EUR",
            "depositos_clientes": "461.543 millones EUR",
            "ratio_capital": "13.0%",
            "roe": "13.8%",
            "morosidad": "3.62%"
        },
        "texto_resumen": "BBVA obtuvo un beneficio neto de 6.452 millones de euros en 2022, mostrando un crecimiento sólido en sus principales geografías. El banco avanzó en su estrategia de digitalización y sostenibilidad.",
        "datos_clave": [
            "Crecimiento en México y Turquía",
            "Innovación en pagos digitales",
            "Compromiso con finanzas sostenibles"
        ]
    },
    {
        "banco": "sabadell",
        "año": "2023",
        "archivo_fuente": "sabadell_2023_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "265.128 millones EUR",
            "beneficio_neto": "1.320 millones EUR",
            "patrimonio_neto": "18.947 millones EUR",
            "creditos_clientes": "152.374 millones EUR",
            "depositos_clientes": "178.456 millones EUR",
            "ratio_capital": "13.8%",
            "roe": "8.5%",
            "morosidad": "3.89%"
        },
        "texto_resumen": "Banco Sabadell cerró 2023 con un beneficio neto de 1.320 millones de euros, reflejando la mejora en sus operaciones tras la reestructuración. El banco se centra en el negocio doméstico y en empresas.",
        "datos_clave": [
            "Foco en banca de empresas",
            "12 millones de clientes",
            "Reestructuración completada",
            "Fortalecimiento de capital"
        ]
    },
    {
        "banco": "sabadell",
        "año": "2022",
        "archivo_fuente": "sabadell_2022_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "258.746 millones EUR",
            "beneficio_neto": "818 millones EUR",
            "patrimonio_neto": "17.856 millones EUR",
            "creditos_clientes": "148.921 millones EUR",
            "depositos_clientes": "173.684 millones EUR",
            "ratio_capital": "13.2%",
            "roe": "5.8%",
            "morosidad": "4.12%"
        },
        "texto_resumen": "En 2022, Banco Sabadell obtuvo un beneficio neto de 818 millones de euros, en un proceso de transformación y optimización de su estructura.",
        "datos_clave": [
            "Proceso de reestructuración",
            "Venta de activos no estratégicos",
            "Mejora de eficiencia operativa"
        ]
    },
    {
        "banco": "kutxabank",
        "año": "2023",
        "archivo_fuente": "kutxabank_2023_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "66.238 millones EUR",
            "beneficio_neto": "557 millones EUR",
            "patrimonio_neto": "7.842 millones EUR",
            "creditos_clientes": "38.452 millones EUR",
            "depositos_clientes": "48.356 millones EUR",
            "ratio_capital": "17.2%",
            "roe": "8.2%",
            "morosidad": "2.35%"
        },
        "texto_resumen": "Kutxabank registró un beneficio neto de 557 millones de euros en 2023, destacando por su sólida posición de capital y su arraigo en el País Vasco. El banco mantiene ratios de morosidad muy bajos.",
        "datos_clave": [
            "Banco regional con fuerte presencia en País Vasco",
            "1.6 millones de clientes",
            "Excelente calidad crediticia",
            "Ratio de morosidad más bajo del sector"
        ]
    },
    {
        "banco": "kutxabank",
        "año": "2022",
        "archivo_fuente": "kutxabank_2022_ejemplo.json",
        "metricas_financieras": {
            "activo_total": "64.125 millones EUR",
            "beneficio_neto": "475 millones EUR",
            "patrimonio_neto": "7.456 millones EUR",
            "creditos_clientes": "36.874 millones EUR",
            "depositos_clientes": "46.928 millones EUR",
            "ratio_capital": "16.8%",
            "roe": "7.5%",
            "morosidad": "2.48%"
        },
        "texto_resumen": "Kutxabank cerró 2022 con un beneficio neto de 475 millones de euros, manteniendo su perfil conservador y su fuerte posición de capital.",
        "datos_clave": [
            "Modelo de banca relacional",
            "Compromiso con el territorio",
            "Inversión en comunidad local"
        ]
    }
]


def generate_finetuning_examples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Genera ejemplos de fine-tuning en formato conversacional."""
    
    examples = []
    
    bank_names = {
        'santander': 'Banco Santander',
        'caixabank': 'CaixaBank',
        'bbva': 'BBVA',
        'sabadell': 'Banco Sabadell',
        'kutxabank': 'Kutxabank'
    }
    
    for record in data:
        banco = record['banco']
        año = record['año']
        metricas = record.get('metricas_financieras', {})
        resumen = record.get('texto_resumen', '')
        datos_clave = record.get('datos_clave', [])
        
        bank_display = bank_names.get(banco, banco.title())
        
        # Tipo 1: Pregunta general
        if metricas:
            examples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"¿Cuáles fueron los resultados de {bank_display} en {año}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"En {año}, {bank_display} obtuvo un beneficio neto de {metricas.get('beneficio_neto', 'N/A')}. El banco contaba con un activo total de {metricas.get('activo_total', 'N/A')} y mantuvo un ratio de capital del {metricas.get('ratio_capital', 'N/A')}."
                    }
                ]
            })
        
        # Tipo 2: Preguntas específicas
        if 'beneficio_neto' in metricas:
            examples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"¿Cuánto ganó {bank_display} en {año}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"{bank_display} obtuvo un beneficio neto de {metricas['beneficio_neto']} en {año}."
                    }
                ]
            })
        
        if 'activo_total' in metricas:
            examples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"¿Cuál es el tamaño del balance de {bank_display} en {año}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"El activo total de {bank_display} en {año} ascendió a {metricas['activo_total']}."
                    }
                ]
            })
        
        if 'ratio_capital' in metricas:
            examples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"¿Cómo está la solvencia de {bank_display} en {año}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"El ratio de capital (CET1) de {bank_display} en {año} fue del {metricas['ratio_capital']}, lo que refleja una sólida posición de solvencia."
                    }
                ]
            })
        
        if 'roe' in metricas:
            examples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"¿Qué rentabilidad tuvo {bank_display} sobre sus recursos propios en {año}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"El ROE (rentabilidad sobre recursos propios) de {bank_display} en {año} fue del {metricas['roe']}."
                    }
                ]
            })
        
        if 'morosidad' in metricas:
            examples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"¿Cuál fue la tasa de morosidad de {bank_display} en {año}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"La tasa de morosidad de {bank_display} en {año} se situó en el {metricas['morosidad']}."
                    }
                ]
            })
        
        # Tipo 3: Resumen
        if resumen:
            examples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"Dame un resumen del desempeño de {bank_display} en {año}"
                    },
                    {
                        "role": "assistant",
                        "content": resumen
                    }
                ]
            })
        
        # Tipo 4: Datos clave
        if datos_clave:
            examples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"¿Cuáles fueron los aspectos más destacados de {bank_display} en {año}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"Los aspectos más destacados de {bank_display} en {año} fueron: " + "; ".join(datos_clave)
                    }
                ]
            })
    
    # Añadir preguntas comparativas
    bancos_por_año = {}
    for record in data:
        año = record['año']
        if año not in bancos_por_año:
            bancos_por_año[año] = []
        bancos_por_año[año].append(record)
    
    for año, records in bancos_por_año.items():
        if len(records) >= 2:
            # Comparar dos bancos
            banco1 = records[0]
            banco2 = records[1]
            name1 = bank_names.get(banco1['banco'], banco1['banco'])
            name2 = bank_names.get(banco2['banco'], banco2['banco'])
            
            if 'beneficio_neto' in banco1['metricas_financieras'] and 'beneficio_neto' in banco2['metricas_financieras']:
                examples.append({
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Compara los beneficios de {name1} y {name2} en {año}"
                        },
                        {
                            "role": "assistant",
                            "content": f"En {año}, {name1} obtuvo un beneficio neto de {banco1['metricas_financieras']['beneficio_neto']}, mientras que {name2} alcanzó {banco2['metricas_financieras']['beneficio_neto']}."
                        }
                    ]
                })
    
    return examples


def main():
    """Genera archivos de ejemplo."""
    print("Generando datos de ejemplo...")
    
    # Crear directorios
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNING_DIR.mkdir(parents=True, exist_ok=True)
    
    # Guardar datos procesados individuales
    for record in EXAMPLE_DATA:
        banco = record['banco']
        año = record['año']
        banco_dir = PROCESSED_DIR / banco
        banco_dir.mkdir(exist_ok=True)
        
        output_file = banco_dir / f"{banco}_{año}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"✓ Creado: {output_file}")
    
    # Guardar datos consolidados
    consolidated_file = PROCESSED_DIR / 'consolidated_data.json'
    with open(consolidated_file, 'w', encoding='utf-8') as f:
        json.dump(EXAMPLE_DATA, f, ensure_ascii=False, indent=2)
    print(f"✓ Creado: {consolidated_file}")
    
    # Generar dataset de fine-tuning
    examples = generate_finetuning_examples(EXAMPLE_DATA)
    
    output_jsonl = FINETUNING_DIR / 'banking_qa_dataset.jsonl'
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✓ Creado: {output_jsonl}")
    print(f"  Total ejemplos: {len(examples)}")
    print("\n✓ Datos de ejemplo generados correctamente!")
    print("\nPuedes usar estos datos para probar el proceso de fine-tuning")
    print("mientras descargas y procesas los PDFs reales.")


if __name__ == '__main__':
    main()
