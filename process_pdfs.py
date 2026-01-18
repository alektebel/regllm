#!/usr/bin/env python3
"""
Standalone script to process local PDF files.
Place your PDF files in data/pdf/ and run this script.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from scraper.regulation_scraper import RegulationScraper

def main():
    print("=" * 70)
    print("RegLLM - Local PDF Processor")
    print("=" * 70)
    print()

    # Check if PDF directory exists and has files
    pdf_dir = Path("data/pdf")
    if not pdf_dir.exists():
        print(f"✗ PDF directory not found: {pdf_dir}")
        print("\nCreating directory...")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {pdf_dir}")
        print("\nPlease place your PDF files in data/pdf/ and run this script again.")
        return

    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))

    if not pdf_files:
        print(f"✗ No PDF files found in {pdf_dir}")
        print("\nPlease place your PDF files in data/pdf/ and run this script again.")
        print("\nSupported naming patterns for automatic source detection:")
        print("  - *bde*.pdf or *banco*.pdf → Bank of Spain")
        print("  - *eba*.pdf → EBA")
        print("  - *ecb*.pdf → ECB")
        print("  - *boe*.pdf → BOE")
        print("  - *basel*.pdf or *bis*.pdf → Basel Committee")
        print("  - *crr*.pdf or *crd*.pdf → EUR-Lex")
        print("  - Other files → Generic 'Local PDF'")
        return

    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    print()

    # Process PDFs
    scraper = RegulationScraper()
    documents = scraper.process_local_pdfs()

    if documents:
        # Save to raw data directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("data/raw") / f"local_pdfs_{timestamp}.json"

        scraper.save_documents(documents, output_file.name)

        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"Processed {len(documents)} PDF documents")
        print(f"Output: {output_file}")
        print()
        print("Next steps:")
        print("  1. Run preprocessing: python src/preprocessing/data_processor.py")
        print("  2. Train model: python train_quick.py --small-subset --epochs 3")
        print("=" * 70)
    else:
        print("\n✗ No documents were successfully processed")
        print("\nPossible issues:")
        print("  - PDFs may be encrypted or password-protected")
        print("  - PDFs may be scanned images (OCR required)")
        print("  - PDFs may have extraction errors")


if __name__ == "__main__":
    main()
