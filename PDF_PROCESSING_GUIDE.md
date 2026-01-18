# Local PDF Processing Guide

## Overview

RegLLM now supports processing local PDF files! Simply place your banking regulation PDFs in `data/pdf/` and they'll be automatically extracted and included in training.

## Quick Start

### 1. Add Your PDFs
```bash
# Copy PDFs to the directory
cp /path/to/your/regulation/pdfs/*.pdf data/pdf/

# Or move them
mv /path/to/pdfs/*.pdf data/pdf/
```

### 2. Process PDFs

**Option A: Standalone Processing**
```bash
python process_pdfs.py
```

**Option B: Include in Full Pipeline**
```bash
# Scraper automatically includes PDFs
python src/scraper/regulation_scraper.py

# Or use the pipeline
python run_pipeline.py --scrape --preprocess
```

## Automatic Source Detection

The system automatically detects the regulatory source from filename patterns:

| Filename Contains | Detected Source |
|-------------------|-----------------|
| `bde`, `banco`, `spain` | Bank of Spain |
| `eba` | European Banking Authority |
| `ecb` | European Central Bank |
| `boe` | BOE (Spanish Official Gazette) |
| `basel`, `bis` | Basel Committee |
| `cnmv` | CNMV |
| `crr`, `crd`, `eur-lex` | EUR-Lex (EU Law) |
| Others | Generic "Local PDF" |

## Naming Best Practices

### Recommended Format
```
[source]-[topic]-[date/version].pdf
```

### Examples
```
bde-circular-4-2017-credit-risk.pdf
eba-guidelines-irb-assessment-2020.pdf
ecb-supervisory-manual-2023.pdf
boe-ley-10-2014-consolidada.pdf
basel-III-framework-complete.pdf
crr-regulation-575-2013-consolidated.pdf
```

## Supported PDF Types

| Type | Support | Notes |
|------|---------|-------|
| **Text-based PDFs** | ✅ Full support | PDFs with selectable text |
| **Searchable PDFs** | ✅ Full support | Scanned with OCR layer |
| **Image-only PDFs** | ❌ Not supported | Requires OCR preprocessing |
| **Encrypted PDFs** | ⚠️ Limited | May fail extraction |
| **Password-protected** | ❌ Not supported | Remove password first |

## Processing Details

### What Happens

1. **PDF Discovery**: Finds all `.pdf` files in `data/pdf/`
2. **Text Extraction**: Extracts text page-by-page using PyPDF2
3. **Source Detection**: Identifies source from filename
4. **Keyword Extraction**: Finds banking regulation keywords (PD, LGD, IRB, etc.)
5. **JSON Output**: Saves to `data/raw/local_pdfs_[timestamp].json`

### Output Format

```json
{
  "url": "file:///path/to/pdf",
  "source": "Bank of Spain (Local PDF)",
  "type": "pdf",
  "title": "filename-without-extension",
  "text": "extracted text content...",
  "scraped_at": "2026-01-17T12:00:00",
  "keywords": ["PD", "IRB", "credit risk"],
  "local_file": true,
  "filename": "original-filename.pdf"
}
```

## Common Use Cases

### Case 1: Processing Bank of Spain Circulars

```bash
# Download circulars from BdE website
# Place in data/pdf/ with descriptive names
cp ~/Downloads/Circular-4-2017.pdf data/pdf/bde-circular-4-2017.pdf

# Process
python process_pdfs.py
```

### Case 2: Adding EBA Guidelines

```bash
# Name files clearly for source detection
cp ~/Documents/EBA-IRB-Guidelines.pdf data/pdf/eba-irb-guidelines-2020.pdf
cp ~/Documents/EBA-LGD-Estimation.pdf data/pdf/eba-lgd-estimation-2021.pdf

# Process all at once
python process_pdfs.py
```

### Case 3: Mixed Sources

```bash
# Multiple sources in one batch
data/pdf/
  ├── bde-circular-3-2018.pdf
  ├── eba-stress-testing-2021.pdf
  ├── ecb-guide-2023.pdf
  └── basel-iv-summary.pdf

# All processed with correct sources
python process_pdfs.py
```

### Case 4: Integration with Web Scraping

```bash
# Scrape from URLs AND process local PDFs
python run_pipeline.py --scrape --preprocess

# This combines:
# - Web-scraped documents
# - Local PDF documents
# Into one training dataset
```

## Troubleshooting

### Problem: "No text extracted"

**Possible Causes:**
- PDF is image-based (scanned without OCR)
- PDF is encrypted or corrupted
- PDF has non-standard encoding

**Solutions:**
1. Open PDF in a reader - can you select text?
2. If not, use OCR tool first:
   ```bash
   # Using ocrmypdf (install: pip install ocrmypdf)
   ocrmypdf input.pdf output.pdf
   cp output.pdf data/pdf/
   ```
3. Try saving PDF with "Save As" in PDF reader

### Problem: "Extraction very slow"

**Causes:**
- Very large PDFs (100+ pages)
- Complex formatting

**Solutions:**
1. Split large PDFs into chapters:
   ```bash
   # Using pdftk
   pdftk large.pdf cat 1-50 output part1.pdf
   pdftk large.pdf cat 51-100 output part2.pdf
   ```
2. Process in smaller batches

### Problem: "Incorrect source detected"

**Solution:**
Rename file to include source keywords:
```bash
# Before (detected as "Local PDF")
mv regulation.pdf data/pdf/bde-regulation-2023.pdf

# Now detected as "Bank of Spain"
```

### Problem: "PDF encrypted"

**Solution:**
Remove password protection:
```bash
# Using qpdf
qpdf --decrypt --password=PASSWORD input.pdf output.pdf

# Or use GUI tool like Adobe Acrobat
```

## Advanced Usage

### Skip Local PDFs in Scraper

```python
from scraper.regulation_scraper import RegulationScraper

scraper = RegulationScraper()
scraper.run(include_local_pdfs=False)  # Only scrape URLs
```

### Process Custom Directory

```python
scraper = RegulationScraper()
docs = scraper.process_local_pdfs(pdf_dir="path/to/other/pdfs")
```

### Check Extracted Keywords

```bash
# After processing, check what keywords were found
cat data/raw/local_pdfs_*.json | grep -o '"keywords": \[[^]]*\]' | head -5
```

## Integration with Pipeline

### Manual Workflow

```bash
# 1. Add PDFs
cp ~/regulations/*.pdf data/pdf/

# 2. Process PDFs
python process_pdfs.py

# 3. Preprocess data
python src/preprocessing/data_processor.py

# 4. Train model
python train_quick.py --epochs 3
```

### Automatic Workflow

```bash
# One command - does everything
python run_pipeline.py --scrape --preprocess --train-small
```

## Performance

### Extraction Speed

| PDF Type | Pages | Time |
|----------|-------|------|
| Simple text | 50 | ~5 seconds |
| Complex layout | 50 | ~10 seconds |
| Large document | 500 | ~2 minutes |

### Memory Usage

- **Small PDFs** (< 10MB): Negligible
- **Medium PDFs** (10-50MB): ~100MB RAM
- **Large PDFs** (> 50MB): May require 500MB+ RAM

## Quality Tips

1. **High-Quality PDFs**: Use official sources when possible
2. **Text-Based**: Prefer text-based over scanned PDFs
3. **Consolidated Versions**: Use latest consolidated regulations
4. **Complete Documents**: Include full text, not excerpts
5. **Proper Naming**: Name files descriptively for source detection

## Example PDFs to Collect

### Bank of Spain
- Circular 4/2017 (Credit risk)
- Circular 3/2018 (Accounting)
- Supervisory guidelines

### EBA
- Guidelines on IRB Assessment Methodology
- Guidelines on PD estimation
- Guidelines on LGD estimation
- Guidelines on treatment of defaulted exposures

### ECB
- Supervisory Manual
- TRIM guides
- Stress testing methodologies

### BOE
- Ley 10/2014 (Banking law)
- Royal Decree 84/2015
- Other banking legislation

### EUR-Lex
- CRR (Regulation 575/2013) - Download PDF version
- CRD IV/V directives
- MiFID regulations (if relevant)

### Basel Committee
- Basel III framework
- Basel IV changes
- Consultation papers

## Current Status

Check what PDFs you have:
```bash
# List PDFs
ls -lah data/pdf/

# Count PDFs
ls -1 data/pdf/*.pdf 2>/dev/null | wc -l

# Show with sizes
du -h data/pdf/*.pdf
```

## Tips & Best Practices

1. **Start Small**: Test with 2-3 PDFs first
2. **Verify Extraction**: Check `data/raw/local_pdfs_*.json` output
3. **Name Consistently**: Use clear, consistent naming scheme
4. **Organize**: Consider subdirectories for different sources
5. **Document**: Keep notes on which PDFs you've added
6. **Version Control**: Note PDF versions/dates in filenames
7. **Clean Up**: Remove duplicate or superseded versions

## Need Help?

- **Setup Issues**: Run `python verify_setup.py`
- **PDF Problems**: See troubleshooting section above
- **General Questions**: Check `README.md` or `USAGE_GUIDE.md`
- **Updates**: See `UPDATES.md` for latest changes

## Summary

✅ **Easy**: Just copy PDFs to `data/pdf/`
✅ **Automatic**: Source detection from filename
✅ **Integrated**: Works with full pipeline
✅ **Flexible**: Standalone or combined with web scraping
✅ **Documented**: Clear output in JSON format

Start adding your regulation PDFs now! 📄
