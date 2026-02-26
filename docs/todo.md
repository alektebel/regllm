# RegLLM - TODO List

## Current Status: RAG System Complete ✅

### Completed Features
- [x] Web scraper for regulatory sources (EUR-Lex, BOE, BdE, EBA, ECB, Basel)
- [x] Local PDF processing
- [x] Data preprocessing pipeline
- [x] Model training with LoRA and 4-bit quantization
- [x] Qwen2.5-7B model support
- [x] Web UI for chatbot interaction
- [x] CLI for chatbot interaction
- [x] **Dataset management web UI**
- [x] **Dataset management CLI**
- [x] Automatic backups
- [x] Dataset validation
- [x] **RAG System with ChromaDB** (NEW)
- [x] **Hybrid Search (Semantic + BM25)** (NEW)
- [x] **Response Verification System** (NEW)
- [x] **Enhanced Scraper with LinkedIn support** (NEW)
- [x] **FastAPI REST API** (NEW)
- [x] **Enhanced Gradio Web UI** (NEW)
- [x] **Interactive CLI** (NEW)

---

## Post-MVP: Planned Features

### 1. MCP (Model Context Protocol) Integration 📋

#### Overview
Integrate MCP to read and process structured financial reports (FINREP, COREP) in XLSX format for automated data extraction and quality assessment.

#### Components

##### A. FINREP Processing
- [ ] **FINREP Reader Module**
  - [ ] Parse XLSX format (multiple sheets)
  - [ ] Extract financial position data
  - [ ] Extract P&L statements
  - [ ] Extract regulatory capital calculations
  - [ ] Handle different FINREP versions/templates

- [ ] **Data Extraction**
  - [ ] Extract key financial metrics
  - [ ] Extract regulatory ratios
  - [ ] Extract capital adequacy data
  - [ ] Extract risk-weighted assets
  - [ ] Map to standardized format

- [ ] **Quality Checks**
  - [ ] Validate data completeness
  - [ ] Check for inconsistencies
  - [ ] Cross-reference with regulations
  - [ ] Flag potential errors

##### B. COREP Processing
- [ ] **COREP Reader Module**
  - [ ] Parse XLSX format
  - [ ] Extract own funds data (CR1, CR2)
  - [ ] Extract credit risk data (CR3-CR10)
  - [ ] Extract market risk data
  - [ ] Extract operational risk data
  - [ ] Handle different COREP templates

- [ ] **Risk Metrics Extraction**
  - [ ] IRB parameters (PD, LGD, CCF, maturity)
  - [ ] RWA calculations
  - [ ] Capital requirements by risk type
  - [ ] Large exposures
  - [ ] Leverage ratio components

- [ ] **Validation Engine**
  - [ ] Check calculation consistency
  - [ ] Validate against CRR/CRD requirements
  - [ ] Compare with EBA guidelines
  - [ ] Flag regulatory breaches

##### C. Stress Testing Module
- [ ] **Report Comparison**
  - [ ] Compare FINREP/COREP across periods
  - [ ] Calculate period-over-period changes
  - [ ] Identify trends
  - [ ] Detect anomalies

- [ ] **Stress Scenario Analysis**
  - [ ] Extract baseline scenarios
  - [ ] Extract adverse scenarios
  - [ ] Compare stress impacts
  - [ ] Assess capital adequacy under stress

- [ ] **Quality Assessment**
  - [ ] Validate stress test assumptions
  - [ ] Check scenario consistency
  - [ ] Compare against peer banks
  - [ ] Generate quality scores

##### D. Dataset Integration
- [ ] **Automated Q&A Generation**
  - [ ] Generate questions from FINREP data
  - [ ] Generate questions from COREP data
  - [ ] Create Q&A from stress test results
  - [ ] Include source citations

- [ ] **Data Enrichment**
  - [ ] Link FINREP/COREP to regulations
  - [ ] Add context from EBA/ECB guidelines
  - [ ] Create comparative analyses
  - [ ] Generate explanatory content

- [ ] **Training Data Pipeline**
  - [ ] Auto-extract from new FINREP/COREP files
  - [ ] Validate generated Q&A
  - [ ] Add to training dataset
  - [ ] Trigger model retraining

#### Technical Stack
- [ ] **Libraries**
  - [ ] openpyxl or xlrd for XLSX reading
  - [ ] pandas for data manipulation
  - [ ] MCP SDK (when available)
  - [ ] Integration with existing RegLLM pipeline

- [ ] **Architecture**
  ```
  FINREP/COREP XLSX Files
         ↓
  MCP Reader Module
         ↓
  Data Extraction & Validation
         ↓
  Quality Assessment
         ↓
  Q&A Generation
         ↓
  Dataset Integration
         ↓
  Model Training
  ```

#### Use Cases

##### Use Case 1: Stress Test Analysis
```python
# Read stress test reports
finrep_baseline = read_finrep("bank_2024_baseline.xlsx")
finrep_adverse = read_finrep("bank_2024_adverse.xlsx")

# Compare scenarios
comparison = compare_scenarios(finrep_baseline, finrep_adverse)

# Generate Q&A
qa_pairs = generate_stress_qa(comparison)
# Q: "¿Cuál es el impacto del escenario adverso en el ratio CET1?"
# A: "Según el test de estrés de 2024, el ratio CET1 disminuye de 15.2% (baseline) a 11.8% (adverse)..."

# Add to training dataset
dataset.add_samples(qa_pairs)
```

##### Use Case 2: COREP Quality Check
```python
# Read COREP report
corep = read_corep("bank_q1_2024.xlsx")

# Validate data
validation = validate_corep(corep)

# Generate quality report
report = generate_quality_report(validation)

# Create Q&A from findings
qa_pairs = generate_qa_from_report(report)
# Q: "¿Está el banco cumpliendo con los requisitos de capital de Pilar 1?"
# A: "Sí, según el COREP Q1 2024, el banco mantiene un CET1 de 15.2%, por encima del mínimo regulatorio de 4.5%..."
```

##### Use Case 3: Regulatory Compliance Check
```python
# Read multiple reports
finrep = read_finrep("bank_q1_2024.xlsx")
corep = read_corep("bank_q1_2024.xlsx")

# Cross-validate
consistency_check = cross_validate(finrep, corep)

# Check against regulations
compliance = check_regulatory_compliance(corep, regulations=["CRR", "CRD V"])

# Generate compliance Q&A
qa_pairs = generate_compliance_qa(compliance)
```

#### Implementation Phases

**Phase 1: Basic Reading (1-2 weeks)**
- [ ] FINREP XLSX reader
- [ ] COREP XLSX reader
- [ ] Basic data extraction
- [ ] Simple validation

**Phase 2: Data Processing (2-3 weeks)**
- [ ] Advanced extraction logic
- [ ] Cross-validation
- [ ] Quality checks
- [ ] Report comparison

**Phase 3: Q&A Generation (1-2 weeks)**
- [ ] Q&A generation from data
- [ ] Source citation
- [ ] Dataset integration
- [ ] Testing and validation

**Phase 4: Automation (1 week)**
- [ ] Automated pipeline
- [ ] Batch processing
- [ ] Error handling
- [ ] Documentation

#### Success Criteria
- [ ] Successfully read FINREP/COREP XLSX files
- [ ] Extract key metrics with 100% accuracy
- [ ] Generate high-quality Q&A pairs
- [ ] Integrate seamlessly with existing pipeline
- [ ] Provide quality assessment reports

#### Dependencies
- openpyxl (`pip install openpyxl`)
- pandas (`pip install pandas`)
- MCP SDK (when available)
- Access to sample FINREP/COREP files for testing

---

### 2. Enhanced Scraping
- [ ] Add retry logic with exponential backoff
- [ ] Implement rate limiting per domain
- [ ] Add proxy support for blocked sources
- [ ] Cache scraped pages to avoid re-scraping
- [ ] Parallel scraping for faster collection

### 3. Improved Preprocessing
- [ ] Better keyword extraction (TF-IDF, embeddings)
- [ ] Duplicate detection and removal
- [ ] Question quality scoring
- [ ] Answer completeness validation
- [ ] Multi-language support expansion

### 4. Training Enhancements
- [ ] Experiment tracking (MLflow, Weights & Biases)
- [ ] Hyperparameter tuning
- [ ] Multi-GPU training support
- [ ] Model ensembling
- [ ] Continuous training pipeline

### 5. Model Improvements
- [x] Add retrieval-augmented generation (RAG)
- [x] Implement document embeddings
- [x] Vector database integration (ChromaDB, Pinecone)
- [x] Source verification system
- [x] Confidence scoring for answers

### 6. UI/UX Enhancements
- [ ] Chat history persistence
- [ ] Multi-turn conversation support
- [ ] Source highlighting in responses
- [ ] Export chat to PDF
- [ ] Mobile-friendly interface

### 7. Evaluation & Monitoring
- [ ] Create evaluation dataset
- [ ] Automated testing pipeline
- [ ] Performance metrics tracking
- [ ] A/B testing framework
- [ ] User feedback collection

### 8. Deployment
- [ ] Docker containerization
- [x] API endpoint creation (FastAPI)
- [ ] Load balancing
- [ ] Monitoring and logging
- [ ] Production deployment guide

### 9. Documentation
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Tutorial videos
- [ ] Example use cases
- [ ] Troubleshooting guide

---

## Quick Wins (Low Effort, High Impact)

- [ ] Add more regulatory sources to regurl.txt
- [ ] Collect more PDFs for data/pdf/
- [ ] Improve system prompt with examples
- [ ] Add model comparison script
- [ ] Create evaluation questions set

---

## Long-term Vision

### Banking Regulation Assistant Platform
- Multi-language support (Spanish, English, French, German)
- Real-time regulatory updates monitoring
- Custom fine-tuning per bank
- Integration with bank systems
- Compliance reporting automation
- **Automated FINREP/COREP analysis** 🎯

---

## Contributing

To add a new TODO item:
1. Choose appropriate section
2. Add checkbox [ ]
3. Include brief description
4. Mark with priority if needed (🔴 High, 🟡 Medium, 🟢 Low)

To mark complete:
- Change [ ] to [x]
- Add completion date if significant

---

**Last Updated**: 2026-01-17
**Next Review**: After MCP Phase 1 completion
