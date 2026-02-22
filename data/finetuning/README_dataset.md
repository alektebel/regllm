---
license: mit
language:
- es
tags:
- banking
- regulation
- llm
- fine-tuning
- rag
- credit-risk
- compliance
- eba
- basel
- qa
pretty_name: Banking Regulation QA Dataset (Spanish)
size_categories:
- 1K<n<10K
---

# Banking Regulation QA Dataset (Spanish)

A domain-specific question-answer dataset for fine-tuning LLMs on banking regulation and credit risk topics. Generated and curated as part of the [regllm](https://github.com/alektebel/regllm) project.

## What's in it

~960 Q&A pairs in Spanish covering:

- **EBA Guidelines** (ICAAP, ILAAP, credit risk, NPL management)
- **CRR/CRD IV/V** (capital requirements, leverage ratio, liquidity)
- **Basel III/IV** (Pillar 1/2/3, stress testing)
- **Spanish bank financials** (Santander, BBVA, CaixaBank, Sabadell, Kutxabank — 2022/2023)
- **Credit risk methodology** (PD, LGD, EAD, IRB, IFRS 9)
- **SQL methodology review** (credit risk SQL vs. EBA guidelines)

## Files

| File | Examples | Domain |
|---|---|---|
| `generated_qa.jsonl` | 846 | EBA/CRR/Basel regulation (auto-generated via Qwen2.5-7B) |
| `banking_qa_dataset.jsonl` | 82 | Spanish bank financials 2022–2023 |
| `sql_methodology_comparison_dataset.jsonl` | 20 | Credit risk SQL vs. EBA guidelines |
| `banking_annual_accounts_extra.jsonl` | 12 | Additional accounting Q&A |

## Format

Each line is a JSONL record with chat-style messages ready for SFT:

```json
{
  "messages": [
    {"role": "system", "content": "Eres un experto en regulación bancaria..."},
    {"role": "user", "content": "¿Qué establece el artículo 92 del CRR sobre los requisitos de capital?"},
    {"role": "assistant", "content": "El artículo 92 del CRR establece que las entidades deberán mantener..."}
  ],
  "metadata": {
    "source_file": "crr_regulation.pdf",
    "chunk_index": 4,
    "generated_at": "2026-02-20T10:30:00"
  }
}
```

## How it was generated

The `generated_qa.jsonl` file was produced using [regllm](https://github.com/alektebel/regllm)'s QA generator pipeline:

1. Regulation PDFs chunked into ~800-word segments
2. Each chunk sent to **Qwen2.5-7B-Instruct** with a RAG-enriched prompt (ChromaDB + BM25)
3. Citation tree (regulation → article → paragraph → point) injected into context
4. 3 Q&A pairs generated per chunk, in Spanish

## Use cases

- Fine-tuning open-source LLMs (Qwen, Phi, Gemma) on banking regulation
- RAG pipeline evaluation for compliance use cases
- Benchmarking domain adaptation in the financial sector

## Related

- 🔧 Full pipeline: [github.com/alektebel/regllm](https://github.com/alektebel/regllm)
- 📊 Supported models: Qwen2.5-7B, Qwen2.5-3B, Phi-3-Mini, Gemma-2B

## License

MIT — free to use, modify, and distribute with attribution.
