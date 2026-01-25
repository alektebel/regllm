# Dataset Management Tools

Tools for managing, editing, and cleaning your RegLLM training dataset.

## Overview

RegLLM provides multiple interfaces for dataset management in `src/tools/`:

| Tool | Purpose |
|------|---------|
| `dataset_ui.py` | Web interface for manual editing |
| `dataset_cli.py` | CLI for programmatic access |
| `analyze_quality.py` | Quality analysis and issue detection |
| `clean_dataset.py` | Dataset cleaning and regeneration |
| `process_pdfs.py` | Process local PDF files |

All tools work with the same dataset format and create automatic backups.

---

## Web UI - Interactive Editor

### Launch

```bash
python src/tools/dataset_ui.py
```

Opens at: **http://localhost:7861**

### Features

1. **Browse & Search** - Search samples by query and field
2. **View Sample** - Detailed sample view
3. **Add Sample** - Add new Q&A pairs
4. **Edit Sample** - Modify existing samples
5. **Delete Sample** - Remove with backup
6. **Statistics** - Dataset stats and source distribution

---

## CLI - Programmatic Access

### Basic Usage

```bash
python src/tools/dataset_cli.py <command> [options]
```

All commands output JSON for easy parsing.

### Commands

#### View & Search
```bash
# List all samples
python src/tools/dataset_cli.py list

# Get sample by index
python src/tools/dataset_cli.py get 5

# Search in all fields
python src/tools/dataset_cli.py search "PD"

# Search in specific field
python src/tools/dataset_cli.py search "capital" --field question
python src/tools/dataset_cli.py search "IRB" --field answer
python src/tools/dataset_cli.py search "EBA" --field source
```

#### Modify Dataset
```bash
# Add new sample
python src/tools/dataset_cli.py add \
  --question "¿Qué es el PD floor?" \
  --answer "El PD floor es..." \
  --source "EBA"

# Update sample
python src/tools/dataset_cli.py update 5 --answer "New answer text"

# Delete sample
python src/tools/dataset_cli.py delete 10
```

#### Analysis
```bash
# Get statistics
python src/tools/dataset_cli.py stats

# Validate dataset structure
python src/tools/dataset_cli.py validate

# Export dataset
python src/tools/dataset_cli.py export --output backup.json

# Import dataset
python src/tools/dataset_cli.py import --input new_data.json
```

---

## Quality Analysis

Analyze dataset for quality issues:

```bash
python src/tools/analyze_quality.py
```

Detects issues like:
- Filename-based questions (not conceptual)
- Wrong language (English instead of Spanish)
- Unstructured text dumps
- Missing citations
- Empty or very short answers

Outputs a report to `dataset_quality_report.json`.

---

## Dataset Cleaning

Clean and regenerate the dataset:

```bash
python src/tools/clean_dataset.py
```

This:
1. Backs up existing data
2. Loads raw documents
3. Filters Spanish-only content
4. Generates conceptual Q&A pairs
5. Removes duplicates
6. Saves cleaned dataset

---

## PDF Processing

Process local PDF files:

```bash
# Place PDFs in data/pdf/
python src/tools/process_pdfs.py
```

Supported naming patterns for source detection:
- `*bde*.pdf` or `*banco*.pdf` → Bank of Spain
- `*eba*.pdf` → EBA
- `*ecb*.pdf` → ECB
- `*crr*.pdf` or `*crd*.pdf` → EUR-Lex

---

## Dataset Format

Training data uses ChatML format:

```json
{
  "messages": [
    {"role": "system", "content": "System prompt..."},
    {"role": "user", "content": "Question here"},
    {"role": "assistant", "content": "Answer here"}
  ],
  "metadata": {
    "source": "EBA",
    "url": "https://...",
    "title": "Document title"
  }
}
```

---

## Backups

All modifications create timestamped backups in `data/backups/`:

```
data/backups/train_data_backup_20260117_021500.json
```

---

## Integration Examples

### With jq
```bash
# Get all EBA samples
python src/tools/dataset_cli.py list | \
  jq '.samples[] | select(.source == "EBA")'

# Count by source
python src/tools/dataset_cli.py stats | jq '.sources'
```

### With Python
```python
import subprocess
import json

result = subprocess.run(
    ['python', 'src/tools/dataset_cli.py', 'get', '5'],
    capture_output=True, text=True
)
sample = json.loads(result.stdout)
```
