# Dataset Management Guide

This guide covers the tools for managing, editing, and cleaning your RegLLM training dataset.

## Overview

RegLLM provides two interfaces for dataset management:

1. **Web UI** (`dataset_manager_ui.py`) - Interactive web interface for manual editing
2. **CLI** (`dataset_manager_cli.py`) - Command-line tool for programmatic access (LLMs, agents, scripts)

Both tools work with the same dataset format and create automatic backups before modifications.

---

## Web UI - Interactive Dataset Editor

### Launch Web Interface

```bash
python dataset_manager_ui.py
```

The web interface will start at: **http://localhost:7861**

### Features

#### 1. Browse & Search
- Search across all samples or specific fields (question/answer/source)
- View results in a sortable table
- Quick navigation to samples

#### 2. View Sample
- Load and view any sample by index
- See full question, answer, and metadata
- Read-only detailed view

#### 3. Add Sample
- Add new Q&A pairs manually
- Specify source, URL, and title
- Automatic backup before saving

#### 4. Edit Sample
- Load existing sample for editing
- Modify question, answer, or metadata
- Save changes with automatic backup

#### 5. Delete Sample
- Preview sample before deletion
- Delete with confirmation
- Automatic backup created

#### 6. Statistics
- View dataset statistics
- See distribution by source
- Track average lengths

#### 7. Settings
- Load different datasets
- Configure dataset path
- View backup information

### Backups

All modifications create timestamped backups in `data/backups/`:
```
data/backups/train_data_backup_20260117_021500.json
```

---

## CLI - Programmatic Access

### Basic Usage

```bash
python dataset_manager_cli.py <command> [options]
```

All commands output JSON for easy parsing by LLMs and scripts.

### Commands

#### 1. List All Samples

```bash
python dataset_manager_cli.py list
```

Output:
```json
{
  "total": 98,
  "samples": [
    {
      "index": 0,
      "question": "¿Qué dice la regulación sobre...",
      "source": "EBA",
      "title": "EBA BS 2017"
    }
  ]
}
```

#### 2. Get Specific Sample

```bash
python dataset_manager_cli.py get 5
```

Output:
```json
{
  "index": 5,
  "question": "Full question text...",
  "answer": "Full answer text...",
  "metadata": {
    "source": "EBA",
    "url": "https://...",
    "title": "..."
  }
}
```

#### 3. Search Samples

```bash
# Search in all fields
python dataset_manager_cli.py search "PD estimation"

# Search only in questions
python dataset_manager_cli.py search "capital" --field question

# Search only in answers
python dataset_manager_cli.py search "IRB" --field answer

# Search only in sources
python dataset_manager_cli.py search "EBA" --field source
```

Output:
```json
{
  "query": "PD estimation",
  "field": "all",
  "total_results": 15,
  "results": [...]
}
```

#### 4. Add New Sample

```bash
python dataset_manager_cli.py add \
  --question "¿Qué es el PD floor?" \
  --answer "El PD floor es..." \
  --source "EBA" \
  --url "https://..." \
  --title "EBA Guidelines"
```

Output:
```json
{
  "success": true,
  "message": "Added new sample",
  "index": 98,
  "total_samples": 99
}
```

#### 5. Update Existing Sample

```bash
# Update question only
python dataset_manager_cli.py update 5 --question "New question text"

# Update answer only
python dataset_manager_cli.py update 5 --answer "New answer text"

# Update multiple fields
python dataset_manager_cli.py update 5 \
  --question "New question" \
  --answer "New answer" \
  --source "Bank of Spain"
```

Output:
```json
{
  "success": true,
  "message": "Updated sample #5"
}
```

#### 6. Delete Sample

```bash
python dataset_manager_cli.py delete 10
```

Output:
```json
{
  "success": true,
  "message": "Deleted sample #10",
  "question": "¿Qué dice la regulación...",
  "remaining_samples": 97
}
```

#### 7. Get Statistics

```bash
python dataset_manager_cli.py stats
```

Output:
```json
{
  "total_samples": 98,
  "avg_question_length": 73,
  "avg_answer_length": 6082,
  "sources": {
    "EBA": 71,
    "ECB": 13,
    "BOE": 12,
    "Bank of Spain": 2
  }
}
```

#### 8. Export Dataset

```bash
python dataset_manager_cli.py export --output backup.json
```

#### 9. Import Dataset

```bash
# Replace current dataset
python dataset_manager_cli.py import --input new_data.json

# Append to current dataset
python dataset_manager_cli.py import --input additional_data.json --append
```

#### 10. Validate Dataset

```bash
python dataset_manager_cli.py validate
```

Output:
```json
{
  "valid": true,
  "total_samples": 98,
  "errors": [],
  "warnings": []
}
```

---

## Using CLI with Other LLMs/Agents

The CLI is designed for easy integration with other LLMs and agents.

### Example: Using with Claude via CLI

```bash
# Get a sample and pipe to Claude
python dataset_manager_cli.py get 5 | claude "Review this Q&A pair for quality"

# Search and process results
python dataset_manager_cli.py search "PD" | jq '.results[] | .index' | while read idx; do
  python dataset_manager_cli.py get $idx | claude "Check if this answer is accurate"
done
```

### Example: Python Script with LLM

```python
import subprocess
import json

def get_sample(index):
    """Get sample using CLI."""
    result = subprocess.run(
        ['python', 'dataset_manager_cli.py', 'get', str(index)],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

def update_sample(index, new_answer):
    """Update sample using CLI."""
    subprocess.run([
        'python', 'dataset_manager_cli.py', 'update',
        str(index), '--answer', new_answer
    ])

# Example: Clean all samples with an LLM
stats = json.loads(subprocess.check_output(['python', 'dataset_manager_cli.py', 'stats']))
total = stats['total_samples']

for i in range(total):
    sample = get_sample(i)

    # Send to LLM for review/cleaning
    improved_answer = your_llm_function(sample['question'], sample['answer'])

    # Update if improved
    if improved_answer != sample['answer']:
        update_sample(i, improved_answer)
        print(f"Updated sample {i}")
```

### Example: Batch Processing with jq

```bash
# Get all samples with a specific source
python dataset_manager_cli.py list | jq '.samples[] | select(.source == "EBA")'

# Count samples by source
python dataset_manager_cli.py stats | jq '.sources'

# Export only EBA samples
python dataset_manager_cli.py list | \
  jq '.samples[] | select(.source == "EBA") | .index' | \
  while read idx; do
    python dataset_manager_cli.py get $idx
  done | jq -s '.' > eba_samples.json
```

---

## Dataset Format

The dataset uses the instruction-response format:

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "Eres un experto en regulación bancaria española..."
      },
      {
        "role": "user",
        "content": "¿Qué es el PD floor?"
      },
      {
        "role": "assistant",
        "content": "Según EBA (source): El PD floor es..."
      }
    ],
    "metadata": {
      "source": "EBA",
      "url": "https://...",
      "title": "Document title"
    }
  }
]
```

### Field Descriptions

- **messages**: Array of 3 messages (system, user, assistant)
  - `system`: Instructions for the model
  - `user`: The question
  - `assistant`: The answer with source citation

- **metadata**: Source information
  - `source`: Regulatory source (EBA, ECB, Bank of Spain, etc.)
  - `url`: Source document URL
  - `title`: Document title

---

## Workflow Examples

### Manual Dataset Cleaning

1. **Launch Web UI**:
   ```bash
   python dataset_manager_ui.py
   ```

2. **Browse and Search**: Find samples that need cleaning

3. **Edit**: Click on "Edit Sample", modify content, save

4. **Verify**: Use "Statistics" tab to track progress

### Automated Cleaning with LLM

1. **Export dataset**:
   ```bash
   python dataset_manager_cli.py export --output original.json
   ```

2. **Process with your LLM**:
   ```bash
   python your_cleaning_script.py
   ```

3. **Validate results**:
   ```bash
   python dataset_manager_cli.py validate
   ```

4. **Compare stats**:
   ```bash
   python dataset_manager_cli.py stats
   ```

### Adding New Samples

#### Via Web UI
1. Go to "Add Sample" tab
2. Fill in question, answer, source
3. Click "Add Sample"

#### Via CLI
```bash
python dataset_manager_cli.py add \
  --question "¿Cómo se calcula el LGD?" \
  --answer "El LGD se calcula según..." \
  --source "Bank of Spain"
```

### Quality Control Process

1. **Validate structure**:
   ```bash
   python dataset_manager_cli.py validate
   ```

2. **Check statistics**:
   ```bash
   python dataset_manager_cli.py stats
   ```

3. **Search for issues** (e.g., very short answers):
   ```bash
   python dataset_manager_cli.py list | \
     jq '.samples[] | select(.answer | length < 100)'
   ```

4. **Review and fix** using Web UI or CLI

---

## Tips for Dataset Quality

### Good Practices

1. **Source Citations**: Always include proper source citations in answers
2. **Specificity**: Questions should be specific and clear
3. **Completeness**: Answers should be comprehensive but concise
4. **Accuracy**: Verify regulatory information is correct
5. **Consistency**: Maintain consistent formatting and style

### Quality Checks

```bash
# Find samples with no source
python dataset_manager_cli.py list | jq '.samples[] | select(.source == "")'

# Find very short answers (potential issues)
python dataset_manager_cli.py list | jq '.samples[] | select(.answer | length < 200)'

# Find duplicate questions
python dataset_manager_cli.py list | jq '[.samples[].question] | group_by(.) | map(select(length > 1))'
```

### Using LLMs for Quality Improvement

```python
# Example: Improve answer quality with LLM
for i in range(total_samples):
    sample = get_sample(i)

    prompt = f"""
    Review this Q&A pair for a banking regulation chatbot:

    Question: {sample['question']}
    Answer: {sample['answer']}
    Source: {sample['metadata']['source']}

    Improve the answer if needed:
    - Ensure source citation is present
    - Check for accuracy
    - Improve clarity
    - Keep Spanish language

    Return improved answer or 'OK' if no changes needed.
    """

    improved = llm.query(prompt)

    if improved != 'OK':
        update_sample(i, improved)
```

---

## Backups

### Automatic Backups

Every save operation creates a backup:
- Location: `data/backups/`
- Format: `train_data_backup_YYYYMMDD_HHMMSS.json`
- Retention: Manual cleanup (keep important ones)

### Manual Backup

```bash
# Using CLI export
python dataset_manager_cli.py export --output manual_backup_$(date +%Y%m%d).json

# Using system copy
cp data/processed/train_data.json data/backups/manual_backup_$(date +%Y%m%d).json
```

### Restore from Backup

```bash
# Copy backup to main dataset
cp data/backups/train_data_backup_20260117_021500.json data/processed/train_data.json

# Or use CLI import
python dataset_manager_cli.py import --input data/backups/train_data_backup_20260117_021500.json
```

---

## Troubleshooting

### "Dataset not found"
- Check that `data/processed/train_data.json` exists
- Run the preprocessing pipeline first
- Or specify custom path: `--dataset path/to/dataset.json`

### "Invalid JSON output"
- The dataset file may be corrupted
- Restore from backup
- Validate with: `python -m json.tool data/processed/train_data.json`

### Web UI won't start
- Check port 7861 is available
- Install dependencies: `pip install gradio pandas`
- Check error messages in terminal

### CLI command fails
- Ensure dataset path is correct
- Check command syntax: `python dataset_manager_cli.py --help`
- Validate dataset structure first

---

## Next Steps

After cleaning your dataset:

1. **Validate**: `python dataset_manager_cli.py validate`
2. **Check stats**: `python dataset_manager_cli.py stats`
3. **Train model**: `python train_quick.py --epochs 5`
4. **Test quality**: Use the chat UI to test model responses

---

## Advanced: MCP Integration (TODO)

**Future Feature**: Model Context Protocol (MCP) integration for reading financial reports.

### Planned Capabilities
- Read FINREP reports (XLSX format)
- Read COREP reports (XLSX format)
- Extract stress test data
- Compare reports across periods
- Quality assessment of regulatory data
- Automated data extraction for training

### Use Cases
- Compare bank stress test results
- Extract regulatory reporting data
- Validate compliance data
- Generate Q&A from regulatory reports
- Automated dataset enrichment from FINREP/COREP

**Status**: Planned for future development

---

## Summary

| Tool | Purpose | Use Case |
|------|---------|----------|
| **Web UI** | Manual editing | Browse, edit, add, delete samples interactively |
| **CLI** | Programmatic access | LLM integration, batch processing, automation |

Both tools:
- ✅ Work with the same dataset
- ✅ Create automatic backups
- ✅ Support full CRUD operations
- ✅ Validate data structure
- ✅ Provide statistics and search

Choose the tool that fits your workflow:
- **Manual review**: Use Web UI
- **Automated cleaning**: Use CLI
- **LLM-assisted editing**: Use CLI with your LLM
- **Batch operations**: Use CLI with scripts
