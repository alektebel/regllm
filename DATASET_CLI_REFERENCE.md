# Dataset CLI Quick Reference

## Common Commands

### View & Search
```bash
# List all samples
python dataset_manager_cli.py list

# Get sample by index
python dataset_manager_cli.py get 5

# Search in all fields
python dataset_manager_cli.py search "PD"

# Search in specific field
python dataset_manager_cli.py search "capital" --field question
python dataset_manager_cli.py search "IRB" --field answer
python dataset_manager_cli.py search "EBA" --field source
```

### Modify Dataset
```bash
# Add new sample
python dataset_manager_cli.py add \
  --question "¿Qué es el PD floor?" \
  --answer "El PD floor es..." \
  --source "EBA"

# Update sample
python dataset_manager_cli.py update 5 --answer "New answer text"

# Delete sample
python dataset_manager_cli.py delete 10
```

### Analysis
```bash
# Get statistics
python dataset_manager_cli.py stats

# Validate dataset
python dataset_manager_cli.py validate

# Export dataset
python dataset_manager_cli.py export --output backup.json

# Import dataset
python dataset_manager_cli.py import --input new_data.json
```

## LLM Integration Examples

### Using with jq
```bash
# Get all EBA samples
python dataset_manager_cli.py list | \
  jq '.samples[] | select(.source == "EBA")'

# Count by source
python dataset_manager_cli.py stats | jq '.sources'

# Find samples with short answers
python dataset_manager_cli.py list | \
  jq '.samples[] | select(.answer | length < 200)'
```

### Using with Python
```python
import subprocess
import json

# Get sample
result = subprocess.run(
    ['python', 'dataset_manager_cli.py', 'get', '5'],
    capture_output=True, text=True
)
sample = json.loads(result.stdout)

# Update sample
subprocess.run([
    'python', 'dataset_manager_cli.py', 'update', '5',
    '--answer', 'New improved answer'
])
```

### Batch Processing
```bash
# Process all samples
python dataset_manager_cli.py list | \
  jq '.samples[].index' | \
  while read idx; do
    python dataset_manager_cli.py get $idx | your_llm_processor
  done

# Search and fix
python dataset_manager_cli.py search "error_pattern" | \
  jq '.results[].index' | \
  while read idx; do
    # Process and update each
    python dataset_manager_cli.py update $idx --answer "Fixed"
  done
```

## Output Format

All commands output JSON:

```json
{
  "success": true,
  "message": "Operation completed",
  "additional_fields": "..."
}
```

Errors:
```json
{
  "error": "Error message"
}
```

## Tips

- All modifications create automatic backups in `data/backups/`
- Use `--dataset path/to/file.json` to work with different datasets
- Pipe output to `jq` for advanced JSON processing
- Check exit codes: 0 = success, 1 = error
- Use `validate` before and after bulk operations
