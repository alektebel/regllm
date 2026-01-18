"""
Web scraper for Spanish banking regulation documents.
Collects documents from Bank of Spain, ECB, BOE, CNMV, and Basel Committee.
"""

import requests
from bs4 import BeautifulSoup
import PyPDF2
import json
import time
import os
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegulationScraper:
    """Scraper for banking regulation documents."""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.delay = 2  # Delay between requests in seconds

    def load_urls(self, url_file: str = "regurl.txt") -> List[str]:
        """Load URLs from the regurl.txt file."""
        urls = []
        with open(url_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)
        logger.info(f"Loaded {len(urls)} URLs from {url_file}")
        return urls

    def fetch_url(self, url: str) -> Optional[requests.Response]:
        """Fetch a URL with error handling."""
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            time.sleep(self.delay)  # Rate limiting
            return response
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def extract_pdf_text(self, pdf_url: str) -> Optional[str]:
        """Extract text from a PDF document URL."""
        try:
            response = self.fetch_url(pdf_url)
            if not response:
                return None

            # Save PDF temporarily
            temp_pdf = self.output_dir / "temp.pdf"
            with open(temp_pdf, 'wb') as f:
                f.write(response.content)

            # Extract text
            text = ""
            with open(temp_pdf, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            # Clean up
            temp_pdf.unlink()
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_url}: {e}")
            return None

    def extract_pdf_from_file(self, pdf_path: Path) -> Optional[str]:
        """Extract text from a local PDF file."""
        try:
            logger.info(f"Extracting text from local PDF: {pdf_path.name}")

            # Extract text
            text = ""
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                logger.info(f"  Pages: {num_pages}")

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"  Error extracting page {page_num}: {e}")
                        continue

            if not text.strip():
                logger.warning(f"  No text extracted from {pdf_path.name}")
                return None

            logger.info(f"  Extracted {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error processing local PDF {pdf_path}: {e}")
            return None

    def extract_html_text(self, html_content: str, url: str) -> Dict[str, any]:
        """Extract relevant text from HTML content with improved structure handling."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside",
                            "iframe", "noscript", "meta", "link"]):
            element.decompose()

        # Special handling for EUR-Lex pages (CRR, CRD, etc.)
        if 'eur-lex.europa.eu' in url:
            text = self._extract_eurlex_content(soup)
        # Special handling for BOE (Spanish official gazette)
        elif 'boe.es' in url:
            text = self._extract_boe_content(soup)
        # Special handling for Bank of Spain
        elif 'bde.es' in url:
            text = self._extract_bde_content(soup)
        # General HTML extraction
        else:
            text = self._extract_general_html(soup)

        # Extract title
        title = soup.find('title')
        title = title.get_text(strip=True) if title else urlparse(url).path

        return {
            'title': title,
            'text': text,
            'links': self.extract_pdf_links(soup, url)
        }

    def _extract_eurlex_content(self, soup: BeautifulSoup) -> str:
        """Extract structured content from EUR-Lex legal documents."""
        content_parts = []

        # EUR-Lex typically uses specific div classes for legal content
        main_content = soup.find('div', id='text') or soup.find('div', class_='eli-main-content')

        if main_content:
            # Extract articles with their structure
            articles = main_content.find_all(['div', 'p', 'table'],
                                            class_=lambda x: x and ('article' in str(x).lower() or
                                                                   'paragraph' in str(x).lower() or
                                                                   'title' in str(x).lower()))

            if articles:
                for article in articles:
                    # Get article number/title
                    title_elem = article.find(['span', 'p'], class_=lambda x: x and 'title' in str(x).lower())
                    if title_elem:
                        content_parts.append(f"\n{title_elem.get_text(strip=True)}\n")

                    # Get article text
                    text = article.get_text(separator='\n', strip=True)
                    if text and len(text) > 20:  # Avoid very short fragments
                        content_parts.append(text)
            else:
                # Fallback: get all meaningful paragraphs
                content_parts.append(main_content.get_text(separator='\n', strip=True))
        else:
            # Fallback to body content
            body = soup.find('body')
            if body:
                content_parts.append(body.get_text(separator='\n', strip=True))

        text = '\n\n'.join(content_parts)
        return self._clean_text(text)

    def _extract_boe_content(self, soup: BeautifulSoup) -> str:
        """Extract content from BOE (Spanish official gazette) documents."""
        content_parts = []

        # BOE uses specific structure
        main_content = (soup.find('div', id='textoConsolidado') or
                       soup.find('div', id='texto') or
                       soup.find('div', class_='documento'))

        if main_content:
            # Extract articles and sections
            articles = main_content.find_all(['p', 'div'],
                                            class_=lambda x: x and ('articulo' in str(x).lower() or
                                                                   'titulo' in str(x).lower() or
                                                                   'capitulo' in str(x).lower()))

            if articles:
                for article in articles:
                    text = article.get_text(separator='\n', strip=True)
                    if text and len(text) > 20:
                        content_parts.append(text)
            else:
                content_parts.append(main_content.get_text(separator='\n', strip=True))
        else:
            # Fallback
            body = soup.find('body')
            if body:
                content_parts.append(body.get_text(separator='\n', strip=True))

        text = '\n\n'.join(content_parts)
        return self._clean_text(text)

    def _extract_bde_content(self, soup: BeautifulSoup) -> str:
        """Extract content from Bank of Spain documents."""
        content_parts = []

        # Bank of Spain structure
        main_content = (soup.find('div', class_='contenido') or
                       soup.find('div', id='contenido') or
                       soup.find('main') or
                       soup.find('article'))

        if main_content:
            content_parts.append(main_content.get_text(separator='\n', strip=True))
        else:
            body = soup.find('body')
            if body:
                content_parts.append(body.get_text(separator='\n', strip=True))

        text = '\n'.join(content_parts)
        return self._clean_text(text)

    def _extract_general_html(self, soup: BeautifulSoup) -> str:
        """General HTML content extraction."""
        # Try to find main content area
        main_content = (soup.find('main') or
                       soup.find('article') or
                       soup.find('div', class_='content') or
                       soup.find('div', id='content'))

        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        lines = []
        for line in text.splitlines():
            line = line.strip()
            # Skip very short lines (likely navigation/UI elements)
            if len(line) > 2:
                lines.append(line)

        text = '\n'.join(lines)

        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove common navigation/UI text patterns
        ui_patterns = [
            r'^(Menu|Menú|Navigation|Navegación|Skip to|Ir a|Print|Imprimir).*$',
            r'^(Cookie|Privacy|Legal notice|Aviso legal).*$',
            r'^(Home|Inicio|Back|Volver|Search|Buscar).*$'
        ]

        for pattern in ui_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Final cleanup
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def extract_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract PDF links from a webpage."""
        pdf_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.pdf') or 'pdf' in href.lower():
                full_url = urljoin(base_url, href)
                pdf_links.append(full_url)
        return pdf_links

    def scrape_page(self, url: str) -> List[Dict[str, any]]:
        """Scrape a single page and return documents found."""
        documents = []

        # Handle PDF URLs directly
        if url.endswith('.pdf'):
            text = self.extract_pdf_text(url)
            if text:
                doc = {
                    'url': url,
                    'source': self.get_source_name(url),
                    'type': 'pdf',
                    'title': os.path.basename(urlparse(url).path),
                    'text': text,
                    'scraped_at': datetime.now().isoformat(),
                    'keywords': self.extract_keywords(text)
                }
                documents.append(doc)
        else:
            # Handle HTML pages
            response = self.fetch_url(url)
            if not response:
                return documents

            extracted = self.extract_html_text(response.text, url)

            # Save main page content
            if extracted['text']:
                doc = {
                    'url': url,
                    'source': self.get_source_name(url),
                    'type': 'html',
                    'title': extracted['title'],
                    'text': extracted['text'],
                    'scraped_at': datetime.now().isoformat(),
                    'keywords': self.extract_keywords(extracted['text'])
                }
                documents.append(doc)

            # Process linked PDFs
            logger.info(f"Found {len(extracted['links'])} PDF links")
            for pdf_link in extracted['links'][:10]:  # Limit to first 10 PDFs per page
                text = self.extract_pdf_text(pdf_link)
                if text:
                    doc = {
                        'url': pdf_link,
                        'source': self.get_source_name(pdf_link),
                        'type': 'pdf',
                        'title': os.path.basename(urlparse(pdf_link).path),
                        'text': text,
                        'scraped_at': datetime.now().isoformat(),
                        'parent_url': url,
                        'keywords': self.extract_keywords(text)
                    }
                    documents.append(doc)

        return documents

    def get_source_name(self, url: str) -> str:
        """Extract source name from URL."""
        domain = urlparse(url).netloc
        if 'bde.es' in domain:
            return 'Bank of Spain'
        elif 'ecb.europa.eu' in domain or 'bankingsupervision.europa.eu' in domain:
            return 'ECB'
        elif 'bis.org' in domain:
            return 'Basel Committee'
        elif 'boe.es' in domain:
            return 'BOE (Spanish Official Gazette)'
        elif 'cnmv.es' in domain:
            return 'CNMV'
        elif 'eba.europa.eu' in domain:
            return 'EBA'
        elif 'eur-lex.europa.eu' in domain:
            return 'EUR-Lex (EU Law)'
        else:
            return domain

    def extract_keywords(self, text: str) -> List[str]:
        """Extract banking regulation keywords from text."""
        keywords = []
        patterns = [
            r'\b(?:credit risk|riesgo de crédito)\b',
            r'\b(?:PD|probability of default)\b',
            r'\b(?:LGD|loss given default)\b',
            r'\b(?:EAD|exposure at default)\b',
            r'\b(?:Basel III|Basel IV|Basilea)\b',
            r'\b(?:CRR|CRD|Capital Requirements)\b',
            r'\b(?:IRB|internal ratings-based)\b',
            r'\b(?:IFRS 9)\b',
            r'\b(?:provision|provisión)\b',
            r'\b(?:capital|capital requirement)\b',
            r'\b(?:retail|minorista)\b',
            r'\b(?:corporate|corporativo)\b',
            r'\b(?:SME|PYME)\b',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.extend(matches)

        return list(set(keywords))[:20]  # Return unique keywords, max 20

    def save_documents(self, documents: List[Dict[str, any]], filename: str):
        """Save scraped documents to JSON file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(documents)} documents to {output_file}")

    def process_local_pdfs(self, pdf_dir: str = "data/pdf") -> List[Dict[str, any]]:
        """Process all PDF files from a local directory."""
        pdf_path = Path(pdf_dir)

        if not pdf_path.exists():
            logger.warning(f"PDF directory not found: {pdf_dir}")
            return []

        # Find all PDF files
        pdf_files = list(pdf_path.glob("*.pdf")) + list(pdf_path.glob("*.PDF"))

        if not pdf_files:
            logger.info(f"No PDF files found in {pdf_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        documents = []
        for pdf_file in pdf_files:
            logger.info(f"\nProcessing: {pdf_file.name}")

            # Extract text from PDF
            text = self.extract_pdf_from_file(pdf_file)

            if text:
                # Determine source from filename or default to "Local PDF"
                source = self._guess_source_from_filename(pdf_file.name)

                doc = {
                    'url': f'file:///{pdf_file.absolute()}',
                    'source': source,
                    'type': 'pdf',
                    'title': pdf_file.stem,  # Filename without extension
                    'text': text,
                    'scraped_at': datetime.now().isoformat(),
                    'keywords': self.extract_keywords(text),
                    'local_file': True,
                    'filename': pdf_file.name
                }
                documents.append(doc)
                logger.info(f"  ✓ Successfully processed {pdf_file.name}")
            else:
                logger.warning(f"  ✗ Failed to extract text from {pdf_file.name}")

        return documents

    def _guess_source_from_filename(self, filename: str) -> str:
        """Guess the source from the PDF filename."""
        filename_lower = filename.lower()

        if 'bde' in filename_lower or 'banco' in filename_lower or 'spain' in filename_lower:
            return 'Bank of Spain (Local PDF)'
        elif 'eba' in filename_lower:
            return 'EBA (Local PDF)'
        elif 'ecb' in filename_lower:
            return 'ECB (Local PDF)'
        elif 'boe' in filename_lower:
            return 'BOE (Local PDF)'
        elif 'basel' in filename_lower or 'bis' in filename_lower:
            return 'Basel Committee (Local PDF)'
        elif 'cnmv' in filename_lower:
            return 'CNMV (Local PDF)'
        elif 'crr' in filename_lower or 'crd' in filename_lower or 'eur-lex' in filename_lower:
            return 'EUR-Lex (Local PDF)'
        else:
            return 'Local PDF'

    def run(self, url_file: str = "regurl.txt", include_local_pdfs: bool = True, pdf_dir: str = "data/pdf"):
        """Run the scraper on all URLs and optionally process local PDFs."""
        all_documents = []

        # Process URLs
        urls = self.load_urls(url_file)
        logger.info(f"\n{'='*70}")
        logger.info("STEP 1: Scraping from URLs")
        logger.info(f"{'='*70}\n")

        for i, url in enumerate(urls, 1):
            logger.info(f"Processing URL {i}/{len(urls)}: {url}")
            documents = self.scrape_page(url)
            all_documents.extend(documents)
            logger.info(f"Collected {len(documents)} documents from this URL")

        logger.info(f"\n✓ URL scraping complete! Collected {len(all_documents)} documents from URLs")

        # Process local PDFs
        if include_local_pdfs:
            logger.info(f"\n{'='*70}")
            logger.info("STEP 2: Processing local PDF files")
            logger.info(f"{'='*70}\n")

            local_docs = self.process_local_pdfs(pdf_dir)
            all_documents.extend(local_docs)

            logger.info(f"\n✓ Local PDF processing complete! Collected {len(local_docs)} documents from PDFs")

        # Save all documents
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_documents(all_documents, f"regulation_data_{timestamp}.json")

        logger.info(f"\n{'='*70}")
        logger.info(f"SCRAPING COMPLETE!")
        logger.info(f"{'='*70}")
        logger.info(f"Total documents collected: {len(all_documents)}")
        logger.info(f"  - From URLs: {len(all_documents) - len(local_docs) if include_local_pdfs else len(all_documents)}")
        if include_local_pdfs:
            logger.info(f"  - From local PDFs: {len(local_docs)}")
        logger.info(f"{'='*70}\n")

        return all_documents


if __name__ == "__main__":
    scraper = RegulationScraper()
    scraper.run()
