#!/usr/bin/env python3
"""
Ejemplo de uso del modelo Qwen2.5-7B fine-tuneado con datos bancarios.
Este script muestra cómo cargar y usar el modelo entrenado.
"""

import json
import re
import sys
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch
except ImportError:
    print("Error: Instala transformers, peft y torch primero:")
    print("  pip install transformers peft torch bitsandbytes")
    sys.exit(1)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT_BASE = (
    "Eres un asistente experto en el sector bancario español.\n"
    "REGLA FUNDAMENTAL: Responde ÚNICAMENTE con los datos proporcionados a continuación.\n"
    "Si no aparecen datos para responder la pregunta, di \"No dispongo de esa información.\"\n"
    "NUNCA inventes cifras ni aproximes valores."
)

# Metric display names for formatting
METRIC_LABELS = {
    "activo_total": "Activo total",
    "beneficio_neto": "Beneficio neto",
    "patrimonio_neto": "Patrimonio neto",
    "creditos_clientes": "Créditos a clientes",
    "depositos_clientes": "Depósitos de clientes",
    "ratio_capital": "Ratio CET1",
    "roe": "ROE",
    "morosidad": "Morosidad (NPL)",
}

# Bank name aliases for keyword matching
BANK_ALIASES = {
    "santander": ["santander", "banco santander"],
    "bbva": ["bbva"],
    "caixabank": ["caixabank", "caixa bank", "la caixa"],
    "sabadell": ["sabadell", "banco sabadell"],
    "kutxabank": ["kutxabank", "kutxa"],
}


def normalize_spanish_number(text: str) -> float | None:
    """Normalize a Spanish-format number string to a float.

    Handles European number formatting (dots as thousands separators,
    commas as decimal separators) and percentage signs.
    """
    s = text.strip().rstrip("%").strip()
    # Remove currency/unit suffixes
    for suffix in ["millones EUR", "millones de euros", "millones", "EUR"]:
        s = s.replace(suffix, "").strip()

    if not s:
        return None

    # European format: 1.720.000 or 11.076 (thousands dots) vs 2,35 (decimal comma)
    # If there are multiple dots, they are thousand separators
    dot_count = s.count(".")
    has_comma = "," in s
    if dot_count > 1:
        # All dots are thousand separators: 1.720.000
        s = s.replace(".", "")
    elif dot_count == 1 and has_comma:
        # Dot is thousands, comma is decimal: 1.320,50
        s = s.replace(".", "").replace(",", ".")
    elif has_comma and dot_count == 0:
        # Comma is decimal: 2,35
        s = s.replace(",", ".")
    # else: single dot is decimal: 12.8

    try:
        return float(s)
    except ValueError:
        return None


class BankingFactStore:
    """Loads consolidated banking data and retrieves facts relevant to a query."""

    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Index by (bank, year)
        self.facts = {}
        for entry in raw:
            key = (entry["banco"], entry["año"])
            self.facts[key] = entry

        # Load official source URLs from banks_urls.json
        self.official_sources = {}  # (bank, year) -> {url, descripcion, nombre_completo, web_oficial}
        urls_path = Path(data_path).parent.parent / "banks_urls.json"
        if urls_path.exists():
            with open(urls_path, "r", encoding="utf-8") as f:
                banks_data = json.load(f)
            for bank_id, bank_info in banks_data.get("bancos", {}).items():
                for year, report in bank_info.get("cuentas_anuales", {}).items():
                    self.official_sources[(bank_id, year)] = {
                        "url": report["url"],
                        "descripcion": report.get("descripcion", ""),
                        "nombre_completo": bank_info.get("nombre_completo", bank_id),
                        "web_oficial": bank_info.get("web_oficial", ""),
                    }
            print(f"Official sources loaded: {len(self.official_sources)} bank-year reports")
        else:
            print(f"Warning: banks_urls.json not found at {urls_path}")

        print(f"Fact store loaded: {len(self.facts)} bank-year entries")

    def _match_query(self, query: str) -> tuple[set[str], set[str]]:
        """Extract matched banks and years from a query string."""
        query_lower = query.lower()

        matched_banks = set()
        for bank_id, aliases in BANK_ALIASES.items():
            for alias in aliases:
                if alias in query_lower:
                    matched_banks.add(bank_id)
                    break

        matched_years = set(re.findall(r"20[0-9]{2}", query))
        known_years = {entry["año"] for entry in self.facts.values()}
        matched_years = matched_years & known_years
        if not matched_years:
            matched_years = known_years
        if not matched_banks:
            matched_banks = {entry["banco"] for entry in self.facts.values()}

        return matched_banks, matched_years

    def find_relevant_facts(self, query: str) -> str:
        """Match bank names and years in the query, return formatted fact block."""
        formatted_text, _ = self.find_relevant_facts_structured(query)
        return formatted_text

    def find_relevant_facts_structured(self, query: str) -> tuple[str, dict]:
        """Match bank names and years, return formatted text AND raw metrics dict.

        Returns:
            (formatted_text, raw_facts) where raw_facts is keyed by
            (bank, year) containing the full entry dict.
        """
        matched_banks, matched_years = self._match_query(query)

        blocks = []
        raw_facts = {}
        for bank in sorted(matched_banks):
            for year in sorted(matched_years):
                entry = self.facts.get((bank, year))
                if not entry:
                    continue
                raw_facts[(bank, year)] = entry
                lines = [f"[{bank.upper()} {year}]"]
                for metric_key, label in METRIC_LABELS.items():
                    value = entry["metricas_financieras"].get(metric_key)
                    if value:
                        lines.append(f"- {label}: {value}")
                for dato in entry.get("datos_clave", []):
                    lines.append(f"- {dato}")
                blocks.append("\n".join(lines))

        if blocks:
            formatted = (
                "DATOS VERIFICADOS (usa SOLO estos datos, NO inventes cifras):\n\n"
                + "\n\n".join(blocks)
            )
        else:
            formatted = "No dispongo de datos verificados para esta consulta. Indica que no tienes esa información."

        return formatted, raw_facts


def extract_numeric_claims(response: str) -> list[dict]:
    """Extract numeric claims (percentages, monetary amounts) from a response.

    Returns a list of dicts with keys: value (original string),
    normalized (float), context (surrounding text snippet).
    """
    claims = []
    seen_values = set()

    patterns = [
        # Percentages: 12.8%, 2,35% (must not be preceded by another digit/dot)
        r"(?<![.\d])(\d+[.,]\d+)\s*%",
        # Whole-number percentages: 15% (must not be preceded by digit, dot, or comma)
        r"(?<![.,\d])(\d+)\s*%",
        # Monetary with "millones EUR" or "millones de euros"
        r"(?<![.\d])([\d.]+(?:,\d+)?)\s*millones\s*(?:de\s*euros|EUR)",
        # Monetary with just "millones"
        r"(?<![.\d])([\d.]+(?:,\d+)?)\s*millones",
        # European large numbers standalone (at least X.XXX pattern): 1.720.000
        r"(?<!\d)(\d{1,3}(?:\.\d{3}){2,})",
    ]

    for pattern in patterns:
        for m in re.finditer(pattern, response, re.IGNORECASE):
            raw_value = m.group(0).strip()
            num_part = m.group(1)
            normalized = normalize_spanish_number(num_part)
            if normalized is None or normalized == 0:
                continue

            # Deduplicate by position
            span_key = (m.start(), m.end())
            if span_key in seen_values:
                continue
            seen_values.add(span_key)

            # Also deduplicate by the same normalized value from overlapping matches
            # Get surrounding context (up to 40 chars each side)
            start = max(0, m.start() - 40)
            end = min(len(response), m.end() + 40)
            context = response[start:end].strip()

            claims.append({
                "value": raw_value,
                "normalized": normalized,
                "context": context,
            })

    # Deduplicate: if two claims overlap in position, keep the longer match
    claims.sort(key=lambda c: c["value"], reverse=True)
    final = []
    used_values = set()
    for claim in claims:
        # Skip if a claim with the same normalized value is already present
        if claim["normalized"] in used_values:
            continue
        used_values.add(claim["normalized"])
        final.append(claim)

    return final


def verify_response(response: str, relevant_facts: dict,
                    official_sources: dict | None = None) -> dict:
    """Verify numeric claims in a response against the fact store.

    Args:
        response: The model's text response.
        relevant_facts: Dict keyed by (bank, year) with entry dicts.
        official_sources: Dict keyed by (bank, year) with official URL info.

    Returns:
        Dict with 'claims' (list of per-claim results) and 'summary'.
    """
    if official_sources is None:
        official_sources = {}

    claims = extract_numeric_claims(response)

    # Build a flat list of all known numeric values from relevant facts
    # Each entry: (normalized_float, label_str, source_info_dict)
    known_values = []
    for (bank, year), entry in relevant_facts.items():
        # Build source info: prefer official URL, fall back to archivo_fuente
        src = official_sources.get((bank, year))
        source_info = {
            "archivo_fuente": entry.get("archivo_fuente", ""),
            "official_url": src["url"] if src else "",
            "official_desc": src["descripcion"] if src else "",
            "nombre_completo": src["nombre_completo"] if src else bank.upper(),
            "web_oficial": src["web_oficial"] if src else "",
        }

        for metric_key, label in METRIC_LABELS.items():
            raw = entry["metricas_financieras"].get(metric_key)
            if not raw:
                continue
            norm = normalize_spanish_number(raw)
            if norm is not None:
                desc = f"{bank.upper()} {year}: {label} = {raw}"
                known_values.append((norm, desc, source_info))
        # Also check datos_clave for numbers
        for dato in entry.get("datos_clave", []):
            nums = re.findall(r"([\d.]+(?:,\d+)?)", dato)
            for n in nums:
                norm = normalize_spanish_number(n)
                if norm is not None and norm > 0:
                    desc = f"{bank.upper()} {year}: {dato}"
                    known_values.append((norm, desc, source_info))

    results = []
    verified = 0
    contradicted = 0
    unverified = 0

    for claim in claims:
        claim_val = claim["normalized"]
        best_match = None
        best_source_info = {}
        best_status = "UNVERIFIED"

        for fact_val, fact_desc, fact_source_info in known_values:
            if fact_val == 0:
                continue

            # Check exact match with tolerance
            is_percentage = "%" in claim["value"]
            if is_percentage:
                # Absolute tolerance ±0.01 for percentages
                if abs(claim_val - fact_val) <= 0.01:
                    best_match = fact_desc
                    best_source_info = fact_source_info
                    best_status = "VERIFIED"
                    break
                # Close but wrong → contradicted (same order of magnitude)
                elif abs(claim_val - fact_val) / max(abs(fact_val), 1e-9) < 0.5:
                    best_match = fact_desc
                    best_source_info = fact_source_info
                    best_status = "CONTRADICTED"
            else:
                # Relative tolerance ±0.1% for large numbers
                if abs(claim_val - fact_val) / max(abs(fact_val), 1e-9) <= 0.001:
                    best_match = fact_desc
                    best_source_info = fact_source_info
                    best_status = "VERIFIED"
                    break
                # Close but wrong (within 50% → same order of magnitude)
                elif abs(claim_val - fact_val) / max(abs(fact_val), 1e-9) < 0.5:
                    if best_status != "VERIFIED":
                        best_match = fact_desc
                        best_source_info = fact_source_info
                        best_status = "CONTRADICTED"

        if best_status == "VERIFIED":
            verified += 1
        elif best_status == "CONTRADICTED":
            contradicted += 1
        else:
            unverified += 1

        results.append({
            "claim": claim["value"],
            "normalized": claim["normalized"],
            "status": best_status,
            "fact_match": best_match,
            "source": best_source_info,
            "context": claim["context"],
        })

    total = len(results)
    accuracy = (verified / total * 100) if total > 0 else 100.0

    return {
        "claims": results,
        "summary": {
            "total": total,
            "verified": verified,
            "contradicted": contradicted,
            "unverified": unverified,
            "accuracy": accuracy,
        },
    }


def print_verification_report(verification: dict):
    """Print a formatted verification report to stdout."""
    claims = verification["claims"]
    summary = verification["summary"]

    if summary["total"] == 0:
        print("\n  VERIFICATION REPORT: No numeric claims found in response.")
        return

    print(f"\n  {'='*60}")
    print("  VERIFICATION REPORT")
    print(f"  {'='*60}")

    for c in claims:
        if c["status"] == "VERIFIED":
            icon = "\u2713"
            label = "VERIFIED"
        elif c["status"] == "CONTRADICTED":
            icon = "\u2717"
            label = "CONTRADICTED"
        else:
            icon = "?"
            label = "UNVERIFIED"

        line = f"  {icon} {label}: \"{c['claim']}\""
        if c["fact_match"]:
            line += f" -> {c['fact_match']}"
        src = c.get("source", {})
        if src.get("official_url"):
            line += f"\n         Fuente: {src['official_desc']} - {src['official_url']}"
        elif src.get("archivo_fuente"):
            line += f"  [src: {src['archivo_fuente']}]"
        print(line)

    print(f"  {'-'*60}")
    s = summary
    print(f"  Accuracy: {s['accuracy']:.1f}% "
          f"({s['verified']}/{s['total']} verified, "
          f"{s['contradicted']} contradicted, "
          f"{s['unverified']} unverified)")
    print(f"  {'='*60}")


def format_verification_markdown(verification: dict | None) -> str:
    """Format verification result as Markdown for the Gradio UI."""
    if verification is None:
        return ""

    summary = verification["summary"]
    claims = verification["claims"]

    if summary["total"] == 0:
        return (
            "### Verification Report\n\n"
            "No se encontraron cifras numéricas en la respuesta."
        )

    # Accuracy badge
    acc = summary["accuracy"]
    if acc >= 80:
        badge_color = "#16a34a"
        badge_label = "ALTA"
    elif acc >= 50:
        badge_color = "#d97706"
        badge_label = "MEDIA"
    else:
        badge_color = "#dc2626"
        badge_label = "BAJA"

    lines = [
        "### Verification Report\n",
        (f'<div style="display:inline-block;padding:4px 12px;border-radius:6px;'
         f'background:{badge_color};color:white;font-weight:bold;margin-bottom:8px">'
         f'Precision: {acc:.0f}% &mdash; {badge_label}</div>\n'),
        f"**{summary['verified']}** verificadas, "
        f"**{summary['contradicted']}** contradichas, "
        f"**{summary['unverified']}** sin verificar "
        f"(de {summary['total']} cifras)\n",
        "---\n",
    ]

    for c in claims:
        if c["status"] == "VERIFIED":
            icon = '<span style="color:#16a34a;font-weight:bold">\u2713 VERIFICADO</span>'
        elif c["status"] == "CONTRADICTED":
            icon = '<span style="color:#dc2626;font-weight:bold">\u2717 CONTRADICHO</span>'
        else:
            icon = '<span style="color:#d97706;font-weight:bold">? SIN VERIFICAR</span>'

        claim_line = f"- {icon} &nbsp; `{c['claim']}`"
        if c["fact_match"]:
            arrow = "\u2192" if c["status"] == "VERIFIED" else "\u2260"
            claim_line += f" &nbsp;{arrow}&nbsp; *{c['fact_match']}*"
        # Show official source link inline
        src = c.get("source", {})
        if src.get("official_url"):
            claim_line += (
                f'<br><span style="margin-left:24px;color:#6b7280;font-size:0.85em">'
                f'\U0001F4C4 <a href="{src["official_url"]}" target="_blank">'
                f'{src["official_desc"]} - {src["nombre_completo"]}</a></span>'
            )
        lines.append(claim_line)

    # When accuracy is high, add a prominent provenance block with official links
    if acc >= 80:
        # Collect unique official sources from verified claims
        seen_urls = set()
        official_refs = []
        for c in claims:
            if c["status"] != "VERIFIED":
                continue
            src = c.get("source", {})
            url = src.get("official_url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                official_refs.append(src)

        if official_refs:
            lines.append("\n---\n")
            lines.append(
                '<div style="background:#f0fdf4;border:1px solid #bbf7d0;'
                'border-radius:8px;padding:12px;margin-top:8px">'
            )
            lines.append(
                '<strong style="color:#15803d">\u2713 Datos respaldados '
                'por fuentes oficiales:</strong>\n'
            )
            for src in official_refs:
                lines.append(
                    f'<div style="margin-left:8px;margin-top:4px">'
                    f'\U0001F4C4 <strong>{src["nombre_completo"]}</strong> &mdash; '
                    f'<a href="{src["official_url"]}" target="_blank">'
                    f'{src["official_desc"]}</a>'
                    f'</div>'
                )
                if src.get("web_oficial"):
                    lines.append(
                        f'<div style="margin-left:24px;color:#6b7280;font-size:0.85em">'
                        f'Portal inversor: <a href="{src["web_oficial"]}" target="_blank">'
                        f'{src["web_oficial"]}</a></div>'
                    )
            lines.append("</div>")

    return "\n".join(lines)


def format_sources_markdown(raw_facts: dict | None,
                           official_sources: dict | None = None) -> str:
    """Format the source facts used for grounding as Markdown."""
    if not raw_facts:
        return "*No se cargaron datos de referencia.*"
    if official_sources is None:
        official_sources = {}

    lines = ["### Datos de Referencia Utilizados\n"]
    for (bank, year) in sorted(raw_facts.keys()):
        entry = raw_facts[(bank, year)]
        src = official_sources.get((bank, year))

        header = f"#### {bank.upper()} {year}"
        if src:
            header += (
                f' &nbsp; <a href="{src["url"]}" target="_blank">'
                f'\U0001F4C4 {src["descripcion"]} - {src["nombre_completo"]}</a>'
            )
        lines.append(header + "\n")

        lines.append("| Metrica | Valor |")
        lines.append("|---------|-------|")
        for metric_key, label in METRIC_LABELS.items():
            value = entry["metricas_financieras"].get(metric_key)
            if value:
                lines.append(f"| {label} | {value} |")

        datos = entry.get("datos_clave", [])
        if datos:
            lines.append("\n**Datos clave:** " + " / ".join(datos))

        if src and src.get("web_oficial"):
            lines.append(
                f'\n<span style="font-size:0.85em;color:#6b7280">'
                f'Portal inversor: <a href="{src["web_oficial"]}" target="_blank">'
                f'{src["web_oficial"]}</a></span>'
            )
        lines.append("")

    return "\n".join(lines)


def ui_mode(model, tokenizer, fact_store: BankingFactStore,
            port: int = 7862, share: bool = False):
    """Launch a Gradio web UI with real-time claim verification."""
    try:
        import gradio as gr
    except ImportError:
        print("Error: Gradio is required for UI mode. Install with: pip install gradio")
        return

    def handle_query(question: str) -> tuple[str, str, str]:
        """Process a question and return (response, verification, sources)."""
        if not question.strip():
            return "", "", ""

        facts_context, raw_facts = fact_store.find_relevant_facts_structured(question)
        system_prompt = f"{SYSTEM_PROMPT_BASE}\n\n{facts_context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        verification = verify_response(response, raw_facts, fact_store.official_sources)
        verif_md = format_verification_markdown(verification)
        sources_md = format_sources_markdown(raw_facts, fact_store.official_sources)

        return response, verif_md, sources_md

    def run_adversarial() -> tuple[str, str]:
        """Run adversarial testing and return (results_md, summary_md)."""
        questions = _generate_adversarial_questions(model, tokenizer)

        result_lines = []
        all_verifications = []

        for i, question in enumerate(questions, 1):
            response, verification = generate_response(
                model, tokenizer, question,
                fact_store=fact_store, verify=True,
            )

            result_lines.append(f"### Pregunta {i}: {question}\n")
            result_lines.append(f"**Respuesta:** {response}\n")
            if verification:
                result_lines.append(format_verification_markdown(verification))
                all_verifications.append(verification)
            result_lines.append("\n---\n")

        # Aggregate summary
        summary_md = ""
        if all_verifications:
            total_claims = sum(v["summary"]["total"] for v in all_verifications)
            total_verified = sum(v["summary"]["verified"] for v in all_verifications)
            total_contra = sum(v["summary"]["contradicted"] for v in all_verifications)
            total_unver = sum(v["summary"]["unverified"] for v in all_verifications)
            acc = (total_verified / total_claims * 100) if total_claims > 0 else 100.0

            if acc >= 80:
                color = "#16a34a"
            elif acc >= 50:
                color = "#d97706"
            else:
                color = "#dc2626"

            summary_md = (
                "### Resumen del Test Adversarial\n\n"
                f'<div style="display:inline-block;padding:6px 16px;border-radius:8px;'
                f'background:{color};color:white;font-size:1.2em;font-weight:bold;'
                f'margin-bottom:12px">Precision Global: {acc:.1f}%</div>\n\n'
                f"| Metrica | Valor |\n"
                f"|---------|-------|\n"
                f"| Preguntas | {len(all_verifications)} |\n"
                f"| Total cifras | {total_claims} |\n"
                f"| \u2713 Verificadas | {total_verified} ({total_verified/max(total_claims,1)*100:.1f}%) |\n"
                f"| ? Sin verificar | {total_unver} ({total_unver/max(total_claims,1)*100:.1f}%) |\n"
                f"| \u2717 Contradichas | {total_contra} ({total_contra/max(total_claims,1)*100:.1f}%) |\n"
            )

        return "\n".join(result_lines), summary_md

    # Build the Gradio interface
    custom_css = """
    .verified { border-left: 4px solid #16a34a; padding-left: 8px; }
    .contradicted { border-left: 4px solid #dc2626; padding-left: 8px; }
    .source-table th { background: #f3f4f6; }
    """

    with gr.Blocks(
        title="Banking Model - Verificacion en Tiempo Real",
        css=custom_css,
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Modelo Bancario con Verificacion en Tiempo Real\n\n"
            "Consulta datos financieros de bancos espanoles. "
            "Cada cifra en la respuesta se verifica automaticamente contra los datos de referencia."
        )

        with gr.Tabs():
            # --- Query Tab ---
            with gr.Tab("Consulta"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Pregunta",
                            placeholder="Ej: Cual fue el beneficio neto de Kutxabank en 2023?",
                            lines=2,
                        )
                        submit_btn = gr.Button("Consultar", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**Bancos disponibles:** Santander, BBVA, "
                            "CaixaBank, Sabadell, Kutxabank\n\n"
                            "**Periodos:** 2022, 2023"
                        )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        response_output = gr.Markdown(label="Respuesta")
                    with gr.Column(scale=2):
                        verification_output = gr.Markdown(label="Verificacion")

                sources_output = gr.Markdown(label="Datos de Referencia")

                gr.Examples(
                    examples=[
                        ["Cual fue la morosidad de Kutxabank en 2023?"],
                        ["Compara el ROE de BBVA y Santander en 2023"],
                        ["Cual fue el beneficio neto de CaixaBank en 2022 y 2023?"],
                        ["Que banco tuvo el ratio CET1 mas alto en 2023?"],
                        ["Dame los activos totales de Sabadell en 2023"],
                    ],
                    inputs=question_input,
                )

                submit_btn.click(
                    fn=handle_query,
                    inputs=[question_input],
                    outputs=[response_output, verification_output, sources_output],
                )
                question_input.submit(
                    fn=handle_query,
                    inputs=[question_input],
                    outputs=[response_output, verification_output, sources_output],
                )

            # --- Adversarial Tab ---
            with gr.Tab("Test Adversarial"):
                gr.Markdown(
                    "Genera preguntas desafiantes automaticamente, responde con el modelo "
                    "fine-tuneado y verifica todas las cifras.\n\n"
                    "**Nota:** Este proceso tarda varios minutos (genera y responde 10 preguntas)."
                )

                adv_btn = gr.Button("Ejecutar Test Adversarial", variant="primary")
                adv_summary = gr.Markdown(label="Resumen")
                adv_results = gr.Markdown(label="Resultados Detallados")

                adv_btn.click(
                    fn=run_adversarial,
                    outputs=[adv_results, adv_summary],
                )

        gr.Markdown(
            "---\n*Modelo: Qwen2.5-7B fine-tuneado con LoRA | "
            "Datos: consolidated_data.json | "
            "Verificacion automatica post-hoc*"
        )

    print(f"\nLaunching UI at http://127.0.0.1:{port}")
    demo.launch(server_name="127.0.0.1", server_port=port, share=share)


def load_model(model_path: str, use_4bit: bool = True):
    """Carga el modelo base + adaptador LoRA fine-tuneado."""
    print(f"Cargando adaptador LoRA desde: {model_path}")
    print(f"Modelo base: {BASE_MODEL}")

    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configurar cuantización
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }

    if use_4bit:
        try:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization")
        except Exception:
            print("bitsandbytes not available, using FP16")

    # Cargar modelo base + LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    print("Model loaded successfully")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 300,
                      fact_store: BankingFactStore = None,
                      verify: bool = False) -> tuple[str, dict | None]:
    """Genera una respuesta usando el chat template del modelo.

    Returns:
        (response_text, verification_result_or_None)
    """

    # Build system prompt with grounded facts
    raw_facts = None
    if fact_store:
        facts_context, raw_facts = fact_store.find_relevant_facts_structured(prompt)
        system_prompt = f"{SYSTEM_PROMPT_BASE}\n\n{facts_context}"
    else:
        system_prompt = SYSTEM_PROMPT_BASE

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    verification = None
    if verify and raw_facts:
        official = fact_store.official_sources if fact_store else None
        verification = verify_response(response, raw_facts, official)

    return response, verification


def interactive_mode(model, tokenizer, fact_store: BankingFactStore = None,
                     verify: bool = True):
    """Modo interactivo de preguntas y respuestas."""
    print("\n" + "="*60)
    print("MODO INTERACTIVO - ANALISIS BANCARIO")
    print("="*60)
    print("Escribe tus preguntas sobre bancos espanoles.")
    print("Comandos: 'salir' para terminar, 'ejemplos' para ver ejemplos")
    print("="*60 + "\n")

    ejemplos = [
        "Cual fue el beneficio neto de BBVA en 2023?",
        "Como esta la solvencia de CaixaBank en 2023?",
        "Compara el ROE de Santander y BBVA en 2023",
        "Cual fue la tasa de morosidad de Kutxabank en 2022?",
        "Dame un resumen del desempeno de Banco Sabadell en 2023"
    ]

    while True:
        try:
            user_input = input("Tu pregunta: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("\nHasta luego!")
                break

            if user_input.lower() == 'ejemplos':
                print("\nPreguntas de ejemplo:")
                for i, ejemplo in enumerate(ejemplos, 1):
                    print(f"   {i}. {ejemplo}")
                print()
                continue

            # Generar respuesta
            print("\nProcesando...", end="", flush=True)
            response, verification = generate_response(
                model, tokenizer, user_input,
                fact_store=fact_store, verify=verify,
            )
            print("\r" + " "*20 + "\r", end="")  # Limpiar "Procesando..."
            print(f"Respuesta: {response}\n")

            if verification:
                print_verification_report(verification)

        except KeyboardInterrupt:
            print("\n\nHasta luego!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def batch_test(model, tokenizer, fact_store: BankingFactStore = None,
               verify: bool = True):
    """Prueba el modelo con un conjunto de preguntas predefinidas."""
    print("\n" + "="*60)
    print("PRUEBA BATCH - PREGUNTAS PREDEFINIDAS")
    print("="*60 + "\n")

    test_questions = [
        "Cual fue el beneficio neto de Banco Santander en 2023?",
        "Que ratio de capital tuvo CaixaBank en 2022?",
        "Cual es la tasa de morosidad de Kutxabank en 2023?",
        "Compara los activos totales de BBVA y Sabadell en 2023",
    ]

    all_verifications = []
    for i, question in enumerate(test_questions, 1):
        print(f"[{i}/{len(test_questions)}] Pregunta: {question}")
        response, verification = generate_response(
            model, tokenizer, question,
            fact_store=fact_store, verify=verify,
        )
        print(f"    Respuesta: {response}\n")
        if verification:
            print_verification_report(verification)
            all_verifications.append(verification)

    if all_verifications:
        _print_aggregate_summary(all_verifications, "BATCH TEST")


def adversarial_mode(model, tokenizer, fact_store: BankingFactStore):
    """Generate challenging questions with base model, answer with fine-tuned, verify."""
    print("\n" + "="*60)
    print("ADVERSARIAL TESTING MODE")
    print("="*60 + "\n")

    questions = _generate_adversarial_questions(model, tokenizer)
    print(f"Testing with {len(questions)} questions...\n")

    all_verifications = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        response, verification = generate_response(
            model, tokenizer, question,
            fact_store=fact_store, verify=True,
        )
        print(f"    Respuesta: {response}\n")
        if verification:
            print_verification_report(verification)
            all_verifications.append(verification)
        print()

    if all_verifications:
        _print_aggregate_summary(all_verifications, "ADVERSARIAL TESTING")


FALLBACK_ADVERSARIAL_QUESTIONS = [
    "Cual fue exactamente el beneficio neto de Kutxabank en 2023?",
    "Compara la morosidad de Santander y BBVA en 2023. Cual es mayor?",
    "Que banco tuvo el ROE mas alto en 2023 y cual fue su valor exacto?",
    "Cual fue la diferencia en activos totales entre CaixaBank y Sabadell en 2023?",
    "Como evoluciono el ratio CET1 de BBVA entre 2022 y 2023?",
    "Cual es la tasa de morosidad de Sabadell en 2022 y como se compara con Kutxabank?",
    "Dame las cifras exactas de depositos de clientes de Santander en 2022 y 2023.",
    "Que banco tuvo el patrimonio neto mas bajo en 2023?",
    "Compara el beneficio neto de CaixaBank en 2022 vs 2023. Cuanto crecio?",
    "Cual fue el ratio de capital de Kutxabank en 2022? Es el mas alto del sector?",
]


def _generate_adversarial_questions(model, tokenizer) -> list[str]:
    """Use the base model (without LoRA) to generate diverse banking questions."""
    generation_prompt = (
        "Genera exactamente 10 preguntas desafiantes y diversas sobre el sector bancario espanol. "
        "Las preguntas deben cubrir: cifras exactas, comparaciones entre bancos, tendencias entre anos, "
        "y casos limite. Los bancos disponibles son: Santander, BBVA, CaixaBank, Sabadell, Kutxabank. "
        "Los anos disponibles son: 2022 y 2023. "
        "Formato: una pregunta por linea, numeradas del 1 al 10."
    )

    try:
        # Disable LoRA to use base model for question generation
        model.disable_adapter_layers()

        messages = [
            {"role": "system", "content": "Eres un generador de preguntas de test para un sistema bancario."},
            {"role": "user", "content": generation_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Re-enable LoRA for answering
        model.enable_adapter_layers()

        # Parse numbered questions
        questions = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            # Match lines starting with a number followed by . or )
            m = re.match(r"^\d+[.)]\s*(.+)", line)
            if m:
                q = m.group(1).strip()
                if len(q) > 10:  # sanity check
                    questions.append(q)

        if len(questions) >= 5:
            print(f"Generated {len(questions)} questions using base model.")
            return questions[:10]

    except Exception as e:
        print(f"Base model question generation failed: {e}")
        # Re-enable LoRA in case of error
        try:
            model.enable_adapter_layers()
        except Exception:
            pass

    print("Using fallback question bank.")
    return FALLBACK_ADVERSARIAL_QUESTIONS


def _print_aggregate_summary(verifications: list[dict], title: str):
    """Print aggregate verification summary across multiple questions."""
    total_claims = sum(v["summary"]["total"] for v in verifications)
    total_verified = sum(v["summary"]["verified"] for v in verifications)
    total_contradicted = sum(v["summary"]["contradicted"] for v in verifications)
    total_unverified = sum(v["summary"]["unverified"] for v in verifications)

    accuracy = (total_verified / total_claims * 100) if total_claims > 0 else 100.0

    print(f"\n{'='*60}")
    print(f"  {title} SUMMARY")
    print(f"{'='*60}")
    print(f"  Questions Tested: {len(verifications)}")
    print(f"  Total Claims:     {total_claims}")
    print(f"    \u2713 Verified:     {total_verified:3d} ({total_verified/max(total_claims,1)*100:5.1f}%)")
    print(f"    ? Unverified:   {total_unverified:3d} ({total_unverified/max(total_claims,1)*100:5.1f}%)")
    print(f"    \u2717 Contradicted: {total_contradicted:3d} ({total_contradicted/max(total_claims,1)*100:5.1f}%)")
    print(f"  Overall Accuracy: {accuracy:.1f}%")
    print(f"{'='*60}\n")


def main():
    """Funcion principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Usa el modelo Qwen2.5-7B fine-tuneado con datos bancarios"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/finetuned/run_20260215_195521/final_model",
        help="Ruta al adaptador LoRA fine-tuneado"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch", "single", "adversarial", "ui"],
        default="interactive",
        help="Modo de uso (ui = web interface with real-time verification)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Pregunta unica (solo para modo 'single')"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="No usar cuantizacion 4-bit"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable post-hoc claim verification"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7862,
        help="Port for UI mode (default: 7862)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (UI mode only)"
    )

    args = parser.parse_args()
    verify = not args.no_verify

    # Verificar que el modelo existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Modelo no encontrado en {model_path}")
        print("\nAsegurate de haber entrenado el modelo primero.")
        print("Ver instrucciones en BANKING_README.md")
        return 1

    # Load fact store for RAG-based grounding
    fact_store = None
    data_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "consolidated_data.json"
    if data_path.exists():
        try:
            fact_store = BankingFactStore(str(data_path))
        except Exception as e:
            print(f"Warning: Could not load fact store: {e}")
    else:
        print(f"Warning: Fact store not found at {data_path}")

    # Cargar modelo
    try:
        model, tokenizer = load_model(args.model, use_4bit=not args.no_4bit)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return 1

    # Ejecutar segun modo
    if args.mode == "ui":
        if not fact_store:
            print("Error: UI mode requires a fact store (consolidated_data.json)")
            return 1
        ui_mode(model, tokenizer, fact_store, port=args.port, share=args.share)
    elif args.mode == "interactive":
        interactive_mode(model, tokenizer, fact_store=fact_store, verify=verify)
    elif args.mode == "batch":
        batch_test(model, tokenizer, fact_store=fact_store, verify=verify)
    elif args.mode == "adversarial":
        if not fact_store:
            print("Error: Adversarial mode requires a fact store (consolidated_data.json)")
            return 1
        adversarial_mode(model, tokenizer, fact_store)
    elif args.mode == "single":
        if not args.question:
            print("Error: Debes proporcionar --question en modo single")
            return 1
        response, verification = generate_response(
            model, tokenizer, args.question,
            fact_store=fact_store, verify=verify,
        )
        print(f"\nPregunta: {args.question}")
        print(f"Respuesta: {response}")
        if verification:
            print_verification_report(verification)

    return 0


if __name__ == '__main__':
    sys.exit(main())
