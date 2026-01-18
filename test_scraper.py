#!/usr/bin/env python3
"""Test the improved HTML scraping, especially for EUR-Lex CRR III."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from scraper.regulation_scraper import RegulationScraper

def test_eurlex_scraping():
    """Test scraping of EUR-Lex CRR III page."""
    print("Testing improved HTML scraping for EUR-Lex (CRR III)...")
    print("=" * 70)

    scraper = RegulationScraper()

    # Test CRR III URL
    test_url = "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:02013R0575-20250101"

    print(f"\nFetching: {test_url}")
    print("-" * 70)

    documents = scraper.scrape_page(test_url)

    if documents:
        print(f"\n✓ Successfully extracted {len(documents)} document(s)\n")

        for i, doc in enumerate(documents, 1):
            print(f"Document {i}:")
            print(f"  Title: {doc['title']}")
            print(f"  Source: {doc['source']}")
            print(f"  Type: {doc['type']}")
            print(f"  Text length: {len(doc['text'])} characters")
            print(f"  First 500 chars:\n")
            print(f"  {doc['text'][:500]}...")
            print()

            # Check for CRR-specific content
            crr_keywords = ['artículo', 'reglamento', 'capital', 'riesgo', 'crédito']
            found_keywords = [kw for kw in crr_keywords if kw.lower() in doc['text'].lower()]

            if found_keywords:
                print(f"  ✓ Found relevant CRR keywords: {', '.join(found_keywords)}")
            print()
    else:
        print("✗ No documents extracted")
        print("\nThis might be due to:")
        print("  - Network issues")
        print("  - EUR-Lex blocking the request")
        print("  - Page structure changed")
        print("\nTry visiting the URL in your browser to verify it's accessible.")

    print("=" * 70)

if __name__ == "__main__":
    test_eurlex_scraping()
