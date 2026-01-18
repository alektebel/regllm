# Local PDF Files

Place your banking regulation PDF files here for processing.

## How to Use

### 1. Add PDF Files
Copy your PDF files to this directory:
```bash
cp /path/to/your/pdfs/*.pdf data/pdf/
```

### 2. Process PDFs
Run the PDF processor:
```bash
python process_pdfs.py
```

Or include in the full scraping pipeline:
```bash
python src/scraper/regulation_scraper.py
# This will automatically process PDFs in data/pdf/
```

## Automatic Source Detection

The scraper automatically detects the source from filename:

| Filename Pattern | Detected Source |
|-----------------|-----------------|
| `*bde*.pdf`, `*banco*.pdf`, `*spain*.pdf` | Bank of Spain |
| `*eba*.pdf` | European Banking Authority |
| `*ecb*.pdf` | European Central Bank |
| `*boe*.pdf` | BOE (Spanish Official Gazette) |
| `*basel*.pdf`, `*bis*.pdf` | Basel Committee |
| `*cnmv*.pdf` | CNMV |
| `*crr*.pdf`, `*crd*.pdf` | EUR-Lex (EU Law) |
| Other files | Generic "Local PDF" |

### Recommended Naming Convention

For best results, name your PDFs descriptively:
```
bde-circular-4-2017.pdf
eba-irb-guidelines-2020.pdf
ecb-supervisory-manual.pdf
boe-ley-10-2014.pdf
basel-III-final-rule.pdf
crr-consolidated-2025.pdf
```

## Supported PDF Types

✅ **Text-based PDFs**: PDFs with selectable text
❌ **Scanned PDFs**: Image-only PDFs (requires OCR - not supported yet)
⚠️ **Encrypted PDFs**: May fail to extract text

## Example PDFs to Include

### Spanish Banking Regulations
- Bank of Spain Circulars (Circulares BdE)
- BOE banking laws (Ley 10/2014, etc.)
- CNMV regulations

### European Regulations
- CRR (Capital Requirements Regulation)
- CRD (Capital Requirements Directive)
- EBA Guidelines (IRB, Credit Risk, etc.)
- ECB supervisory documents

### International Standards
- Basel III/IV framework documents
- BCBS consultation papers

## Processing Details

The PDF processor will:
1. Extract text from each PDF
2. Identify keywords (PD, LGD, IRB, etc.)
3. Detect source from filename
4. Save to `data/raw/local_pdfs_*.json`

## Troubleshooting

### PDF extraction fails
- **Encrypted PDFs**: Remove password protection first
- **Scanned PDFs**: Convert to text-based PDF using OCR tools
- **Corrupted PDFs**: Try opening in PDF reader to verify integrity

### No text extracted
- Check if PDF is image-based (scanned document)
- Try opening in PDF reader and selecting text
- Use OCR tools like `ocrmypdf` for scanned documents

### Out of memory
- Process PDFs in smaller batches
- Split large PDFs into smaller files

## Integration with Pipeline

### Manual Processing
```bash
# Process PDFs only
python process_pdfs.py

# Then preprocess
python src/preprocessing/data_processor.py
```

### Automatic Processing
```bash
# Full pipeline (includes PDFs automatically)
python run_pipeline.py --scrape --preprocess
```

### Skip Local PDFs
```python
# In your script
from scraper.regulation_scraper import RegulationScraper

scraper = RegulationScraper()
scraper.run(include_local_pdfs=False)  # Skip local PDFs
```

## Current Status

```bash
# Check PDFs in this directory
ls -lah data/pdf/

# Count PDFs
ls -1 data/pdf/*.pdf 2>/dev/null | wc -l
```

## Tips

1. **Organize by source**: Create subdirectories if you have many PDFs
2. **Use descriptive names**: Include source and topic in filename
3. **Test small batch first**: Process a few PDFs to verify extraction works
4. **Check output**: Review `data/raw/local_pdfs_*.json` after processing

## Need Help?

See main documentation:
- `README.md` - Complete guide
- `QUICKSTART.md` - Quick start
- `USAGE_GUIDE.md` - Common commands
