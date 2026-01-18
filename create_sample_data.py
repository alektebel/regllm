#!/usr/bin/env python3
"""
Create a small sample dataset for testing the training pipeline.
This allows you to test the training without waiting for scraping to complete.
"""

import json
from pathlib import Path
from datetime import datetime

# Sample regulation data (realistic Spanish banking regulation content)
sample_documents = [
    {
        "url": "https://www.bde.es/sample1",
        "source": "Bank of Spain",
        "type": "pdf",
        "title": "Circular sobre riesgo de crédito",
        "text": """
        BANCO DE ESPAÑA - CIRCULAR 4/2017

        Artículo 1. Probabilidad de Default (PD)

        La probabilidad de default (PD) es un parámetro fundamental en el cálculo de capital para riesgo de crédito.
        Según la normativa del Banco de España y los requisitos de Basilea III, las entidades que utilicen el método IRB
        (Internal Ratings-Based) deben estimar la PD de sus exposiciones de forma robusta y prudente.

        Para carteras retail (minoristas), la PD debe calcularse considerando un horizonte temporal de 12 meses y debe
        incorporar información de todo el ciclo económico (through-the-cycle). Las entidades deben documentar
        adecuadamente sus modelos de PD y someterlos a validación independiente.

        Los factores de riesgo típicos incluyen: historial crediticio, nivel de endeudamiento, ratio loan-to-value (LTV),
        ingresos del acreditado, y variables macroeconómicas relevantes.
        """,
        "scraped_at": datetime.now().isoformat(),
        "keywords": ["PD", "probability of default", "IRB", "retail", "credit risk"]
    },
    {
        "url": "https://www.eba.europa.eu/sample2",
        "source": "EBA",
        "type": "pdf",
        "title": "Guidelines on IRB Assessment Methodology",
        "text": """
        EUROPEAN BANKING AUTHORITY - EBA/GL/2020/01

        Section 3. Internal Ratings-Based Approach for Credit Risk

        The IRB approach allows banks to use their own internal estimates of risk components to calculate capital requirements.
        The key risk parameters are:

        1. Probability of Default (PD): The likelihood that a borrower will default within one year
        2. Loss Given Default (LGD): The percentage of exposure that will be lost if default occurs
        3. Exposure at Default (EAD): The expected exposure amount at the time of default

        For retail portfolios, institutions must segment exposures into homogeneous risk pools. Common segmentations include:
        - Residential mortgages (hipotecas residenciales)
        - Qualifying revolving retail exposures (tarjetas de crédito)
        - Other retail exposures (otros minoristas)

        Spanish banks must comply with both ECB supervisory guidance and Bank of Spain circulars when implementing IRB models.
        """,
        "scraped_at": datetime.now().isoformat(),
        "keywords": ["IRB", "PD", "LGD", "EAD", "retail", "credit risk"]
    },
    {
        "url": "https://www.bde.es/sample3",
        "source": "Bank of Spain",
        "type": "pdf",
        "title": "Guía sobre el cálculo de LGD",
        "text": """
        BANCO DE ESPAÑA - Guía Técnica 2019

        Cálculo de Loss Given Default (LGD) para Carteras Corporativas y Retail

        El LGD representa la pérdida esperada en caso de incumplimiento, expresada como porcentaje de la exposición.
        Para su cálculo, las entidades deben considerar:

        - Recuperaciones de colaterales (garantías reales y personales)
        - Costes directos e indirectos de recuperación
        - Tiempo hasta la recuperación (actualización temporal)
        - Experiencia histórica de pérdidas

        Para carteras minoristas (retail):
        - Hipotecas residenciales: LGD típico entre 10-30%, dependiendo del LTV
        - Préstamos personales sin garantía: LGD típico entre 50-75%
        - Tarjetas de crédito: LGD típico entre 70-90%

        Es fundamental aplicar ajustes por downturn (condiciones adversas del ciclo económico) según CRR Article 181.

        Las entidades deben realizar backtesting periódico comparando LGD estimado vs. observado.
        """,
        "scraped_at": datetime.now().isoformat(),
        "keywords": ["LGD", "loss given default", "retail", "corporate", "downturn"]
    },
    {
        "url": "https://www.eba.europa.eu/sample4",
        "source": "EBA",
        "type": "pdf",
        "title": "SME Supporting Factor - Opinion",
        "text": """
        EUROPEAN BANKING AUTHORITY - Opinion on SME Supporting Factor

        Treatment of SME Exposures under CRR

        Small and Medium Enterprises (SMEs / PYMEs) receive preferential regulatory treatment to encourage lending.

        Under Article 501 CRR, exposures to SMEs in the retail portfolio or corporate portfolio receive a 0.7619 scaling
        factor (23.81% reduction in risk-weighted assets).

        Definition of SME for regulatory purposes:
        - Annual turnover not exceeding €50 million
        - For retail treatment: additionally must meet retail exposure criteria

        Spanish banks must ensure proper identification and flagging of SME exposures in their risk systems to benefit
        from the supporting factor. The Bank of Spain conducts periodic reviews of SME classification processes.

        For IRB banks, the SME factor applies after calculating RWA using PD, LGD, and EAD parameters.
        """,
        "scraped_at": datetime.now().isoformat(),
        "keywords": ["SME", "PYME", "supporting factor", "CRR", "retail", "corporate"]
    },
    {
        "url": "https://www.boe.es/sample5",
        "source": "BOE",
        "type": "pdf",
        "title": "Ley 10/2014 de ordenación, supervisión y solvencia",
        "text": """
        BOLETÍN OFICIAL DEL ESTADO - LEY 10/2014

        Título III. Requisitos de Solvencia

        Artículo 45. Requerimientos de capital por riesgo de crédito

        Las entidades de crédito españolas deben mantener fondos propios suficientes para cubrir los riesgos de crédito
        derivados de sus actividades. Los requisitos mínimos se calculan según:

        1. Método estándar: Utiliza ponderaciones de riesgo fijas según tipo de exposición
        2. Método IRB básico: Usa PD interna, LGD y EAD supervisores
        3. Método IRB avanzado: Usa PD, LGD y EAD internos (requiere autorización del Banco de España)

        Para carteras minoristas bajo IRB, las entidades deben:
        - Segmentar adecuadamente las exposiciones
        - Estimar parámetros con datos históricos de al menos 5 años
        - Documentar metodologías y procesos de gobierno
        - Realizar validaciones independientes anuales

        El Banco de España supervisa el cumplimiento de estos requisitos y puede imponer requerimientos adicionales
        (capital guidance) si identifica deficiencias en modelos o gobernanza.
        """,
        "scraped_at": datetime.now().isoformat(),
        "keywords": ["capital requirements", "IRB", "PD", "retail", "solvency"]
    },
]

def main():
    print("Creating sample dataset for testing...")

    # Create output directory
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Save sample data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = raw_dir / f"regulation_data_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_documents, f, ensure_ascii=False, indent=2)

    print(f"✓ Created sample dataset with {len(sample_documents)} documents")
    print(f"  Saved to: {output_file}")
    print(f"\nSample topics covered:")
    print(f"  - Probability of Default (PD) calculation")
    print(f"  - IRB methodology for credit risk")
    print(f"  - Loss Given Default (LGD) estimation")
    print(f"  - SME / PYME supporting factor")
    print(f"  - Spanish banking regulation requirements")
    print(f"\nNow you can run:")
    print(f"  python run_pipeline.py --preprocess --train-small")

if __name__ == "__main__":
    main()
