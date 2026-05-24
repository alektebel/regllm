#!/usr/bin/env python3
"""Generate synthetic sample data for testing the compliance system.

Creates under <output_dir>/:
  field_dictionary.csv     — IRB/IFRS9 field definitions
  table_metadata.csv       — SAS-style PROC CONTENTS output
  analyst_notes.txt        — freeform notes on model decisions
  emails/email_001.txt     — synthetic governance email (LGD model decision)
  emails/email_002.txt     — synthetic governance email (PD observation window)
  cycles_sample.csv        — 500 data instances with planted violations
  sample_lgd.sas           — synthetic LGD calibration SAS script
  sample_pd.egp            — synthetic PD estimation EGP project

Usage:
    python scripts/generate_sample_data.py --output data/samples/
"""

import argparse
import csv
import io
import json
import os
import random
import zipfile
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

# ── Field dictionary ──────────────────────────────────────────────────────────

FIELD_DICTIONARY = [
    ("CICLO_ID",                "Identificador único del ciclo de recuperación",                   "ID",           "",          ""),
    ("CONTRATO",                "Número de contrato de crédito",                                   "ID",           "",          ""),
    ("SEGMENTO",                "Segmento de cartera IRB (CORP/RETAIL/MORTGAGE/SME)",               "categorico",   "",          "CORP, RETAIL, MORTGAGE, SME"),
    ("RATING_GRADO",            "Grado de rating interno del deudor (1=mejor, 10=peor)",            "entero",       "",          "1-10"),
    ("PD_ESTIMADA",             "Probabilidad de incumplimiento estimada por el modelo IRB",        "decimal",      "%",         "0.0003-1.0"),
    ("LGD_ESTIMADA",            "Pérdida dado incumplimiento estimada (downturn ajustada)",         "decimal",      "%",         "0.0-1.0"),
    ("EAD",                     "Exposición en el momento del incumplimiento",                      "decimal",      "EUR",       ">0"),
    ("DPDS",                    "Días en situación de impago al cierre del ciclo",                  "entero",       "días",      "0-9999"),
    ("STAGE_IFRS9",             "Etapa IFRS 9 asignada (1=normal, 2=SICR, 3=defaulted)",           "entero",       "",          "1, 2, 3"),
    ("ECL",                     "Pérdida crediticia esperada calculada (PD x LGD x EAD)",          "decimal",      "EUR",       ">=0"),
    ("PROVISION_PERIOD_MONTHS", "Duración del período de provisiones del ciclo en meses",           "entero",       "meses",     ">=9"),
    ("VENTANA_OBSERVACION_YEARS","Ventana de observación histórica usada en estimación de PD",      "decimal",      "años",      ">=5"),
    ("VENTANA_CALIBRACION_YEARS","Período de calibración del modelo IRB",                          "decimal",      "años",      ">=7"),
    ("COLATERAL_TIPO",          "Tipo de garantía asociada (NINGUNA/HIPOTECA/FINANCIERO/PERSONAL)", "categorico",   "",          "NINGUNA, HIPOTECA, FINANCIERO, PERSONAL"),
    ("LGD_FLOOR_APLICADO",      "Suelo regulatorio LGD aplicado según CRR Art. 161",               "decimal",      "%",         "0.30-0.45"),
    ("MoC",                     "Margen de conservadurismo aplicado sobre la estimación de PD",    "decimal",      "%",         ">=0"),
    ("FECHA_INCUMPLIMIENTO",    "Fecha en que el contrato entró en default según CRR Art. 178",    "fecha",        "",          ""),
    ("CURE_FLAG",               "Indicador de salida de incumplimiento (1=curado, 0=activo)",       "binario",      "",          "0, 1"),
    ("PERIODO_OBSERVACION",     "Fecha de referencia del período de observación",                  "fecha",        "",          ""),
    ("RWA",                     "Activos ponderados por riesgo calculados según IRB",               "decimal",      "EUR",       ">=0"),
]

# ── Table metadata (PROC CONTENTS style) ─────────────────────────────────────

TABLE_METADATA_ROWS = [
    ("MYLIB.CICLOS_RECUPERACION", col[0], "NUM" if col[2] in ("decimal","entero","binario") else "CHAR",
     8 if col[2] in ("decimal","entero","binario") else 32, col[0].replace("_", " ").title())
    for col in FIELD_DICTIONARY
]

# ── Analyst notes ─────────────────────────────────────────────────────────────

ANALYST_NOTES = """\
NOTAS MODELO IRB — CARTERA RETAIL/HIPOTECARIO
Fecha: 2025-11-15
Analista: Equipo de Riesgo de Crédito

1. VENTANA DE OBSERVACION PD
   Se confirmó con el comité de modelos que la ventana mínima de observación
   para PD en cartera retail debe ser al menos 5 años (EBA GL 2017/16 §5.3.2).
   En el ciclo de calibración 2024Q4, se detectaron 12 contratos con ventanas
   de solo 3-4 años. Se marcaron como pendientes de revisión.

2. SUELOS LGD
   Para exposiciones hipotecarias (COLATERAL_TIPO=HIPOTECA) el suelo de LGD
   es 30% según CRR Art. 154(3). Para corporativas es 45% (CRR Art. 161(1)(b)).
   El modelo legacy usaba 25% para hipotecas hasta junio 2024 — corregido en
   v3.2 del modelo. Verificar que ningún ciclo anterior a julio 2024 use LGD<30%
   en hipotecas sin ajuste.

3. PROVISION_PERIOD_MONTHS
   El período de provisiones debe reflejar el horizonte de observación.
   Internamente se acordó un mínimo de 9 meses. Ciclos con período inferior
   son datos incompletos y no deben usarse en calibración.

4. MoC (MARGEN DE CONSERVADURISMO)
   Según EBA GL 2017/16 §4.4.3, el MoC debe cubrir la incertidumbre de las
   estimaciones. El valor mínimo aprobado por el comité es 5 bps para PD retail.
   Valores de MoC=0 deben justificarse explícitamente.

5. CURE RATE
   El período de prueba tras cura es 12 meses para retail y 24 meses para
   corporativas (EBA GL 2017/16 §5.4). El campo CURE_FLAG=1 sin superar el
   período de prueba se considera incumplimiento encubierto.
"""

# ── Email threads ─────────────────────────────────────────────────────────────

EMAIL_001 = """\
De: maria.garcia@banco.es
Para: carlos.lopez@banco.es; juan.martinez@banco.es
Asunto: RE: Revisión LGD Downturn — Ciclo 2024Q3
Fecha: 2024-10-03

Carlos,

Revisé los resultados del modelo LGD downturn para el ciclo 2024Q3. Confirmo:

- Para SEGMENTO=MORTGAGE el suelo regulatorio es 30% (CRR Art. 154(3)).
  Actualmente el modelo calcula LGD_ESTIMADA = 0.27 para varios contratos,
  lo que viola el suelo. Hay que aplicar LGD_FLOOR_APLICADO = 0.30 y
  recalcular el ECL.

- Para SEGMENTO=CORP el suelo es 45% (CRR Art. 161(1)(b)). Ningún caso
  por debajo en este ciclo.

- Los contratos con DPDS >= 30 y STAGE_IFRS9 = 1 (hay 3 en el ciclo)
  violan el backstop IFRS 9 B5.5.12. Necesitan reclasificación a Stage 2
  antes de calcular ECL.

Adjunto la lista de contratos afectados. Por favor confirmar corrección
antes del cierre del 10 de octubre.

Saludos,
María García
Validación de Modelos IRB
"""

EMAIL_002 = """\
De: carlos.lopez@banco.es
Para: comite.modelos@banco.es
Asunto: Aprobación cambio ventana observación PD — Cartera CORP
Fecha: 2024-09-15

Estimado Comité,

Se solicita aprobación para ampliar la ventana de observación histórica
para PD en cartera corporativa de 5 a 7 años (VENTANA_OBSERVACION_YEARS).

Motivación:
- EBA GL 2017/16 §5.3.2 requiere mínimo 5 años, pero recomienda 7+
  para segmentos con ciclos crediticios largos (corporativas, PYMEs).
- Ciclos de calibración con VENTANA_OBSERVACION_YEARS < 5 han sido
  rechazados por el supervisor en la última revisión (carta SREP 2024).
- El período de calibración (VENTANA_CALIBRACION_YEARS) debe ser >= 7
  años según la misma directriz. Actualmente tenemos contratos con
  VENTANA_CALIBRACION_YEARS = 6 — no conformes.

Propuesta: establecer como política interna:
  - VENTANA_OBSERVACION_YEARS >= 7 para CORP y SME
  - VENTANA_CALIBRACION_YEARS >= 7 para todas las carteras
  - Cualquier ciclo con valores inferiores debe marcarse como
    DATO_INCOMPLETO y excluirse de la calibración.

Quedo a disposición para comentarios.

Carlos López
Director de Riesgo de Modelos
"""

# ── Sample LGD calibration SAS script ────────────────────────────────────────

SAMPLE_LGD_SAS = """\
/*****************************************************************************
 * LGD Calibration — Cartera Retail/Hipotecario
 * Autor: Equipo IRB
 * Fecha: 2024-Q4
 * Regulacion: CRR Art. 154(3), CRR Art. 161(1)(b), EBA GL 2017/16
 *****************************************************************************/

LIBNAME mylib '/data/irb/calibracion';

/* Carga de datos de ciclos de recuperación */
DATA work.ciclos;
    SET mylib.ciclos_recuperacion;
    WHERE PROVISION_PERIOD_MONTHS >= 9;  /* Filtro: ciclos completos */
RUN;

/* Aplicar suelos regulatorios de LGD */
DATA work.lgd_con_suelos;
    SET work.ciclos;

    /* Suelo LGD hipotecas: 30% (CRR Art. 154(3)) */
    IF COLATERAL_TIPO = 'HIPOTECA' THEN DO;
        IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
    END;

    /* Suelo LGD corporativas: 45% (CRR Art. 161(1)(b)) */
    IF SEGMENTO = 'CORP' AND LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;

    /* Suelo LGD retail sin colateral: 0% (sin suelo explícito en CRR) */
    IF COLATERAL_TIPO = 'NINGUNA' AND SEGMENTO = 'RETAIL' THEN DO;
        /* No aplica suelo mínimo — verificar con área regulatoria */
    END;

RUN;

/* Cálculo ECL = PD x LGD x EAD */
DATA work.ecl_calculo;
    SET work.lgd_con_suelos;

    /* Verificar PD mínima (CRR Art. 160(1): suelo 0.03%) */
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;

    ECL = PD_ESTIMADA * LGD_ESTIMADA * EAD;

    /* Clasificar STAGE IFRS 9 — Backstop 30 DPD (IFRS 9 B5.5.12) */
    IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
        STAGE_IFRS9 = 2;  /* Reclasificar a Stage 2 */
    END;

RUN;

/* Verificación ventana calibración (EBA GL 2017/16 §6.3: >= 7 años) */
PROC MEANS DATA=work.ecl_calculo N MEAN MIN MAX;
    VAR VENTANA_CALIBRACION_YEARS VENTANA_OBSERVACION_YEARS PD_ESTIMADA LGD_ESTIMADA ECL;
    TITLE 'Estadísticos descriptivos — Calibración LGD 2024Q4';
RUN;

/* Exportar registros no conformes */
DATA work.no_conformes;
    SET work.ecl_calculo;
    WHERE VENTANA_CALIBRACION_YEARS < 7
       OR VENTANA_OBSERVACION_YEARS < 5
       OR (COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30)
       OR PD_ESTIMADA < 0.0003;
RUN;

PROC EXPORT DATA=work.no_conformes
    OUTFILE='/data/irb/output/no_conformes_2024Q4.csv'
    DBMS=CSV REPLACE;
RUN;
"""


# ── EGP project XML ───────────────────────────────────────────────────────────

def _build_egp_xml(sas_code: str) -> str:
    """Build a minimal project.xml that embeds SAS code in <Text> elements."""
    escaped = sas_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ProjectCollection>
  <Project>
    <Name>PD_Calibration_2024Q4</Name>
    <CreatedOn>2024-10-01</CreatedOn>
    <Containers>
      <Container>
        <ContainerType>CodeTask</ContainerType>
        <DNA>
          <Name>PD_Estimation_Main</Name>
        </DNA>
        <Tasks>
          <Task>
            <Text>{escaped}</Text>
          </Task>
        </Tasks>
      </Container>
    </Containers>
  </Project>
</ProjectCollection>
"""


SAMPLE_PD_SAS = """\
/*****************************************************************************
 * PD Estimation — Calibración modelo IRB cartera CORP
 * EBA GL 2017/16, CRR Art. 160, CRR Art. 178
 *****************************************************************************/

LIBNAME irb '/data/irb';

/* Filtrar ventana de observación mínima (EBA GL §5.3.2: >= 5 años) */
DATA work.pd_base;
    SET irb.historico_defaults;
    WHERE VENTANA_OBSERVACION_YEARS >= 5;
    /* Definición de incumplimiento: CRR Art. 178 (90 días impago O probable incumplimiento) */
    IF DPDS >= 90 OR PROBABLE_INCUMPLIMIENTO = 1 THEN DEFAULT_FLAG = 1;
    ELSE DEFAULT_FLAG = 0;
RUN;

/* Calcular PD por grado de rating */
PROC FREQ DATA=work.pd_base;
    TABLES RATING_GRADO * DEFAULT_FLAG / OUT=work.pd_por_grado;
RUN;

DATA work.pd_calibrada;
    SET work.pd_por_grado;
    IF DEFAULT_FLAG = 1 THEN PD_ESTIMADA = COUNT / TOTAL_CONTRATOS;
    /* Aplicar suelo CRR Art. 160(1): PD mínima = 0.03% */
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;
    /* Aplicar MoC (EBA GL 2017/16 §4.4.3) */
    PD_CON_MoC = PD_ESTIMADA * (1 + MoC);
RUN;
"""


# ── Cycles sample dataset ─────────────────────────────────────────────────────

def generate_cycles(n: int = 500) -> list[dict]:
    segments = ["CORP", "RETAIL", "MORTGAGE", "SME"]
    collateral = {"CORP": "NINGUNA", "RETAIL": "PERSONAL", "MORTGAGE": "HIPOTECA", "SME": "FINANCIERO"}
    lgd_floors = {"CORP": 0.45, "RETAIL": 0.0, "MORTGAGE": 0.30, "SME": 0.0}
    rows = []
    base_date = date(2024, 1, 1)

    for i in range(n):
        seg = random.choice(segments)
        rating = random.randint(1, 10)
        pd_true = round(0.001 * (rating ** 1.4), 6)
        lgd_true = round(lgd_floors[seg] + random.uniform(0, 0.3), 4)
        ead = round(random.uniform(5000, 500000), 2)
        dpds = random.choices([0, random.randint(1, 29), random.randint(30, 90), random.randint(91, 360)],
                               weights=[55, 20, 15, 10])[0]
        stage = 3 if dpds >= 90 else (2 if dpds >= 30 else 1)
        obs_years = round(random.uniform(4.5, 9.0), 1)
        calib_years = round(random.uniform(5.5, 10.0), 1)
        provision_months = random.choices([random.randint(4, 8), random.randint(9, 36)],
                                           weights=[12, 88])[0]
        moc = round(random.uniform(0.0, 0.03), 4)
        cycle_date = base_date + timedelta(days=random.randint(0, 365))

        # ── Plant intentional violations ──────────────────────────────────────
        # ~8% PD below floor
        if random.random() < 0.08:
            pd_true = round(random.uniform(0.00001, 0.00029), 6)
        # ~10% LGD below floor for MORTGAGE
        if seg == "MORTGAGE" and random.random() < 0.10:
            lgd_true = round(random.uniform(0.10, 0.29), 4)
        # ~12% provision period too short (< 9 months)
        if random.random() < 0.12:
            provision_months = random.randint(1, 8)
        # ~6% calibration window too short
        if random.random() < 0.06:
            calib_years = round(random.uniform(3.0, 6.9), 1)
        # ~8% observation window too short
        if random.random() < 0.08:
            obs_years = round(random.uniform(2.0, 4.9), 1)
        # ~5% STAGE mismatch: dpds>=30 but stage=1
        if dpds >= 30 and random.random() < 0.05:
            stage = 1

        ecl = round(pd_true * lgd_true * ead, 2)
        rows.append({
            "CICLO_ID": f"CIC_{i:05d}",
            "CONTRATO": f"CONT_{random.randint(100000, 999999)}",
            "SEGMENTO": seg,
            "RATING_GRADO": rating,
            "PD_ESTIMADA": pd_true,
            "LGD_ESTIMADA": lgd_true,
            "EAD": ead,
            "DPDS": dpds,
            "STAGE_IFRS9": stage,
            "ECL": ecl,
            "PROVISION_PERIOD_MONTHS": provision_months,
            "VENTANA_OBSERVACION_YEARS": obs_years,
            "VENTANA_CALIBRACION_YEARS": calib_years,
            "COLATERAL_TIPO": collateral[seg],
            "LGD_FLOOR_APLICADO": lgd_floors[seg],
            "MoC": moc,
            "FECHA_INCUMPLIMIENTO": (cycle_date - timedelta(days=dpds)).isoformat() if dpds > 0 else "",
            "CURE_FLAG": 0,
            "PERIODO_OBSERVACION": cycle_date.isoformat(),
            "RWA": round(ead * lgd_true * 12.5 * 0.06, 2),
        })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate synthetic IRB compliance sample data")
    p.add_argument("--output", "-o", default="data/samples", help="Output directory")
    p.add_argument("--cycles", type=int, default=500, help="Number of cycle rows to generate")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "emails").mkdir(exist_ok=True)

    # Field dictionary
    with open(out / "field_dictionary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field_name", "description", "domain", "units", "valid_range"])
        w.writerows(FIELD_DICTIONARY)
    print(f"  field_dictionary.csv — {len(FIELD_DICTIONARY)} fields")

    # Table metadata
    with open(out / "table_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["table_name", "column_name", "sas_type", "length", "label"])
        w.writerows(TABLE_METADATA_ROWS)
    print(f"  table_metadata.csv — {len(TABLE_METADATA_ROWS)} columns")

    # Analyst notes
    (out / "analyst_notes.txt").write_text(ANALYST_NOTES, encoding="utf-8")
    print("  analyst_notes.txt")

    # Emails
    (out / "emails" / "email_001.txt").write_text(EMAIL_001, encoding="utf-8")
    (out / "emails" / "email_002.txt").write_text(EMAIL_002, encoding="utf-8")
    print("  emails/email_001.txt  emails/email_002.txt")

    # Cycles CSV
    cycles = generate_cycles(args.cycles)
    with open(out / "cycles_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cycles[0].keys()))
        w.writeheader()
        w.writerows(cycles)
    print(f"  cycles_sample.csv — {len(cycles)} rows")

    # Sample .sas file
    (out / "sample_lgd.sas").write_text(SAMPLE_LGD_SAS, encoding="utf-8")
    print("  sample_lgd.sas")

    # Sample .egp file
    egp_path = out / "sample_pd.egp"
    xml_content = _build_egp_xml(SAMPLE_PD_SAS)
    with zipfile.ZipFile(egp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.xml", xml_content)
        zf.writestr("programs/pd_estimation.sas", SAMPLE_PD_SAS)
    print("  sample_pd.egp")

    print(f"\nSample data written to: {out.resolve()}")


if __name__ == "__main__":
    main()
