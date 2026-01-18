# RegLLM - Recent Updates

## Updates Made (January 17, 2026)

### Latest: Local PDF Processing Added! 🎉

**New Feature**: Process local PDF files from `data/pdf/` directory

#### What's New
- ✅ **Local PDF Support**: Place PDFs in `data/pdf/` for automatic processing
- ✅ **Automatic Source Detection**: Detects source from filename (BdE, EBA, ECB, etc.)
- ✅ **Standalone Script**: `process_pdfs.py` for processing PDFs separately
- ✅ **Integrated Pipeline**: Automatically included in main scraper
- ✅ **Better PDF Extraction**: Enhanced error handling and page-by-page processing

#### How to Use
```bash
# Place PDFs in data/pdf/
cp /path/to/your/pdfs/*.pdf data/pdf/

# Option 1: Process PDFs only
python process_pdfs.py

# Option 2: Include in full pipeline (automatic)
python src/scraper/regulation_scraper.py
```

#### Automatic Source Detection
The scraper detects sources from filenames:
- `*bde*.pdf` → Bank of Spain
- `*eba*.pdf` → European Banking Authority
- `*ecb*.pdf` → European Central Bank
- `*boe*.pdf` → BOE (Spanish Official Gazette)
- `*basel*.pdf` → Basel Committee
- `*crr*.pdf`, `*crd*.pdf` → EUR-Lex (EU Law)

#### Files Added/Modified
- ✅ `process_pdfs.py` - Standalone PDF processor
- ✅ `data/pdf/README.md` - Documentation for PDF directory
- ✅ `src/scraper/regulation_scraper.py`:
  - Added `extract_pdf_from_file()` - Extract from local PDF
  - Added `process_local_pdfs()` - Process directory of PDFs
  - Added `_guess_source_from_filename()` - Auto-detect source
  - Updated `run()` - Include local PDFs by default

### 1. ✅ Added Qwen2.5-7B Model Support

**Model Configuration** (`config.py` & `src/training/model_setup.py`)

- Added **Qwen2.5-7B-Instruct** as the primary model
- Set as default in config.py
- Better quality than Phi-2 (2.78B), more parameters = better understanding
- Still runs efficiently with 4-bit quantization (~5-6GB GPU)

**Available Models:**
```
qwen2.5-7b (7B)    - Best quality (NEW DEFAULT)
phi-3-mini (3.8B)  - Good balance
phi-2 (2.7B)       - Fast, previously default
gemma-2b (2B)      - Lightweight
qwen-1.8b (1.8B)   - Fastest
```

**Your GPU (RTX 5060 Ti - 15.48GB)**: Perfect for 7B model! 🚀

### 2. ✅ Improved HTML Scraping

**Enhanced HTML Extraction** (`src/scraper/regulation_scraper.py`)

Added specialized extractors for:

#### EUR-Lex Pages (CRR, CRD, EU Regulations)
- Extracts structured legal content (articles, paragraphs)
- Preserves article numbers and hierarchical structure
- Handles consolidated regulations (like CRR III)
- **Tested successfully**: CRR III extracted 2.27 million characters!

#### BOE Pages (Spanish Official Gazette)
- Extracts "artículos", "títulos", "capítulos"
- Finds consolidated text sections
- Proper handling of Spanish legal structure

#### Bank of Spain Pages
- Extracts main content from circulars
- Handles BDE-specific HTML structure
- Preserves regulatory text

#### General Improvements
- Better removal of navigation/UI elements
- Cleaner text extraction (removes cookies, menus, etc.)
- Preserves meaningful line breaks
- Filters out very short lines (navigation fragments)

### 3. ✅ Enhanced Configuration

**Updated LoRA Parameters** (`config.py`)

You increased the LoRA rank for better model adaptation:
```python
'lora': {
    'r': 64,              # Increased from 16 (4x more capacity)
    'lora_alpha': 128,    # Increased from 32 (4x)
    'lora_dropout': 0.05, # Reduced from 0.1 (less regularization)
}
```

**Benefits:**
- More trainable parameters (still only ~1-2% of model)
- Better capacity to learn regulatory patterns
- May require slightly more GPU memory but still very efficient

**Updated System Prompt** (`config.py`)

Now explicitly instructs:
- "Responde en Español, castellano" - Always answer in Spanish
- Better source citation requirements
- Clearer uncertainty handling

### 4. ✅ Updated Data Sources

**Added to regurl.txt:**
```
# CRR III (Capital Requirements Regulation)
https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:02013R0575-20250101

# Additional BOE document
https://www.boe.es/buscar/doc.php?id=BOE-A-2017-14334
```

## Testing the Improvements

### Test Qwen2.5-7B Model

```bash
# The config now defaults to qwen2.5-7b
python train_quick.py --small-subset --epochs 3 --batch-size 2
```

### Test Improved Scraping

```bash
# Test EUR-Lex scraping
python test_scraper.py

# Scrape all updated URLs
python run_pipeline.py --scrape --preprocess
```

### Full Pipeline with New Model

```bash
# Complete pipeline with Qwen2.5-7B
python run_pipeline.py --scrape --preprocess --train-small
```

## Expected Improvements

### Model Quality (Qwen2.5-7B)
- **Better Spanish understanding** (Qwen models excel at multilingual)
- **Improved reasoning** (7B > 2.7B parameters)
- **Better instruction following**
- **More accurate regulatory responses**

### Data Quality (Improved Scraping)
- **CRR III fully extracted** (2.27M chars of EU banking law)
- **Better structured content** (articles, sections preserved)
- **Cleaner text** (no navigation/UI noise)
- **More Spanish content** (BOE, EUR-Lex in Spanish)

## Memory Requirements

### Updated Estimates

**Qwen2.5-7B with 4-bit quantization:**
```
Model loading: ~5-6GB GPU
Training (LoRA): ~6-8GB GPU
Your RTX 5060 Ti (15.48GB): ✅ Plenty of headroom!
```

**Batch size recommendations:**
- Training: 2-4 (depending on sequence length)
- Inference: 1 (fastest)

## Performance Comparison

| Metric | Phi-2 (old) | Qwen2.5-7B (new) |
|--------|-------------|------------------|
| Parameters | 2.78B | 7B |
| GPU Memory | 1.7GB | 5-6GB |
| Quality | Good | Better ✨ |
| Speed | Fast | Moderate |
| Spanish | Decent | Excellent |
| Multilingual | Limited | Strong |

## Configuration Files Modified

1. **config.py**
   - Model: phi-2 → qwen2.5-7b
   - LoRA rank: 16 → 64
   - LoRA alpha: 32 → 128
   - System prompt: Spanish emphasis added

2. **src/training/model_setup.py**
   - Added Qwen2.5-7B to AVAILABLE_MODELS

3. **src/scraper/regulation_scraper.py**
   - Added `_extract_eurlex_content()`
   - Added `_extract_boe_content()`
   - Added `_extract_bde_content()`
   - Added `_clean_text()` with better filtering
   - Updated `get_source_name()` with EUR-Lex

4. **regurl.txt**
   - Added CRR III EUR-Lex URL
   - Added additional BOE document

## Quick Commands

### Retrain with Qwen2.5-7B
```bash
# Quick test (2-3 minutes)
python train_quick.py --small-subset --epochs 3 --batch-size 2

# Full training (10-20 minutes)
python train_quick.py --epochs 5 --batch-size 4
```

### Re-scrape with Improved Scraper
```bash
# Scrape all sources including CRR III
python src/scraper/regulation_scraper.py

# Or use pipeline
python run_pipeline.py --scrape --preprocess
```

### Test EUR-Lex Extraction
```bash
python test_scraper.py
```

## Troubleshooting

### Out of Memory with Qwen2.5-7B

If you get OOM errors:

1. **Reduce batch size:**
   ```bash
   python train_quick.py --batch-size 1 --epochs 3
   ```

2. **Reduce sequence length** (edit `config.py`):
   ```python
   'max_seq_length': 384,  # Down from 512
   ```

3. **Use smaller model** (edit `config.py`):
   ```python
   'base_model': 'phi-2',  # Fall back to 2.7B
   ```

### EUR-Lex Not Scraping

If EUR-Lex blocks requests:
- Add delay: Edit `scraper.delay = 5` in scraper
- Use VPN if rate-limited
- Download manually and place in `data/raw/`

## Next Steps

1. **Test the new model:**
   ```bash
   python train_quick.py --small-subset --epochs 3
   ```

2. **Verify quality improvement:**
   - Test with Spanish questions
   - Check source citations
   - Compare with old model

3. **Full training:**
   ```bash
   python run_pipeline.py --train-full --epochs 5
   ```

4. **Launch UI with new model:**
   ```bash
   ./launch_ui.sh
   ```

## Summary

✅ **Qwen2.5-7B**: Better quality, Spanish-optimized
✅ **EUR-Lex Scraping**: CRR III fully extracted
✅ **Improved HTML**: Cleaner, structured extraction
✅ **Better Config**: Higher LoRA rank, Spanish emphasis
✅ **Your GPU**: Perfect for 7B model (15.48GB)

**Ready to train!** 🚀

---

*Updated: January 17, 2026*
