"""
Enhanced Web Scraper with support for JavaScript-heavy sites like LinkedIn.
Uses Selenium for dynamic content and includes rate limiting and retry logic.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Any
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Selenium imports (optional)
SELENIUM_AVAILABLE = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        SELENIUM_AVAILABLE = True
    except ImportError:
        logger.warning("webdriver-manager not installed. Selenium features limited.")
except ImportError:
    logger.warning("Selenium not installed. Dynamic scraping will not be available.")


class EnhancedScraper:
    """
    Enhanced scraper with support for JavaScript-heavy sites.
    Includes retry logic, rate limiting, and proxy support.
    """

    def __init__(self, output_dir: str = "data/raw",
                 use_selenium: bool = True,
                 headless: bool = True):
        """
        Initialize the enhanced scraper.

        Args:
            output_dir: Directory to save scraped data
            use_selenium: Whether to use Selenium for JS-heavy sites
            headless: Run browser in headless mode
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Request session for simple HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })

        # Rate limiting settings
        self.delay_range = (2, 5)  # Random delay between requests
        self.max_retries = 3
        self.backoff_factor = 2

        # Selenium driver
        self.driver = None
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.headless = headless

        # Sites that require Selenium
        self.selenium_sites = [
            'linkedin.com',
            'twitter.com',
            'x.com',
            'facebook.com',
        ]

    def _init_selenium(self):
        """Initialize Selenium WebDriver."""
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available. Install selenium and webdriver-manager.")
            return False

        if self.driver is not None:
            return True

        try:
            options = Options()
            if self.headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

            # Disable automation detection
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option('excludeSwitches', ['enable-automation'])
            options.add_experimental_option('useAutomationExtension', False)

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)

            # Execute script to hide webdriver
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            logger.info("Selenium WebDriver initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            self.driver = None
            return False

    def _close_selenium(self):
        """Close Selenium WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def _needs_selenium(self, url: str) -> bool:
        """Check if URL requires Selenium for scraping."""
        domain = urlparse(url).netloc.lower()
        return any(site in domain for site in self.selenium_sites)

    def _random_delay(self):
        """Add random delay for rate limiting."""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

    def fetch_with_retry(self, url: str, use_selenium: bool = False) -> Optional[str]:
        """
        Fetch URL with retry logic and exponential backoff.

        Args:
            url: URL to fetch
            use_selenium: Force use of Selenium

        Returns:
            HTML content or None
        """
        for attempt in range(self.max_retries):
            try:
                self._random_delay()

                if use_selenium or self._needs_selenium(url):
                    return self._fetch_with_selenium(url)
                else:
                    return self._fetch_with_requests(url)

            except Exception as e:
                wait_time = self.backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)

        logger.error(f"All {self.max_retries} attempts failed for {url}")
        return None

    def _fetch_with_requests(self, url: str) -> Optional[str]:
        """Fetch URL using requests library."""
        logger.info(f"Fetching (requests): {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def _fetch_with_selenium(self, url: str) -> Optional[str]:
        """Fetch URL using Selenium."""
        if not self._init_selenium():
            logger.warning("Selenium not available, falling back to requests")
            return self._fetch_with_requests(url)

        logger.info(f"Fetching (Selenium): {url}")

        try:
            self.driver.get(url)

            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Additional wait for dynamic content
            time.sleep(3)

            # Scroll to load lazy content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            return self.driver.page_source

        except TimeoutException:
            logger.warning(f"Timeout loading {url}")
            return self.driver.page_source if self.driver else None
        except WebDriverException as e:
            logger.error(f"Selenium error: {e}")
            return None

    def scrape_linkedin_article(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a LinkedIn article or post.

        Args:
            url: LinkedIn URL

        Returns:
            Dictionary with article data or None
        """
        if 'linkedin.com' not in url:
            logger.warning(f"Not a LinkedIn URL: {url}")
            return None

        html = self.fetch_with_retry(url, use_selenium=True)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        # Extract article content
        article_data = {
            'url': url,
            'source': 'LinkedIn',
            'type': 'article',
            'scraped_at': datetime.now().isoformat()
        }

        # Try to extract title
        title_selectors = [
            'h1.article-title',
            'h1[data-test-id="article-title"]',
            'h1.t-24',
            '.feed-shared-article__title',
            'h1'
        ]

        for selector in title_selectors:
            title = soup.select_one(selector)
            if title:
                article_data['title'] = title.get_text(strip=True)
                break

        if 'title' not in article_data:
            article_data['title'] = 'LinkedIn Content'

        # Extract main content
        content_selectors = [
            '.article-content',
            '.feed-shared-update-v2__commentary',
            '.feed-shared-article__description',
            '.article__content',
            '[data-test-id="article-content"]',
            '.break-words',
        ]

        text_parts = []
        for selector in content_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(separator='\n', strip=True)
                if text and len(text) > 50:
                    text_parts.append(text)

        # Fallback: extract from body if no specific content found
        if not text_parts:
            body = soup.find('body')
            if body:
                # Remove script, style, and nav elements
                for tag in body(['script', 'style', 'nav', 'footer', 'header']):
                    tag.decompose()
                text_parts.append(body.get_text(separator='\n', strip=True))

        article_data['text'] = '\n\n'.join(text_parts)

        # Extract author
        author_selectors = [
            '.author-info__name',
            '.feed-shared-actor__name',
            '[data-test-id="author-name"]',
        ]

        for selector in author_selectors:
            author = soup.select_one(selector)
            if author:
                article_data['author'] = author.get_text(strip=True)
                break

        # Extract keywords from text
        article_data['keywords'] = self._extract_keywords(article_data.get('text', ''))

        logger.info(f"Scraped LinkedIn article: {article_data.get('title', 'Unknown')[:50]}...")
        return article_data

    def scrape_linkedin_profile(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape public information from a LinkedIn profile.

        Note: This only works for public profiles and respects LinkedIn's ToS.

        Args:
            url: LinkedIn profile URL

        Returns:
            Dictionary with profile data or None
        """
        if 'linkedin.com/in/' not in url:
            logger.warning(f"Not a LinkedIn profile URL: {url}")
            return None

        html = self.fetch_with_retry(url, use_selenium=True)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        profile_data = {
            'url': url,
            'source': 'LinkedIn Profile',
            'type': 'profile',
            'scraped_at': datetime.now().isoformat()
        }

        # Extract name
        name_elem = soup.select_one('h1.text-heading-xlarge') or soup.select_one('h1')
        if name_elem:
            profile_data['name'] = name_elem.get_text(strip=True)

        # Extract headline
        headline_elem = soup.select_one('.text-body-medium') or soup.select_one('.headline')
        if headline_elem:
            profile_data['headline'] = headline_elem.get_text(strip=True)

        # Extract about section
        about_section = soup.select_one('#about') or soup.select_one('[data-section="summary"]')
        if about_section:
            about_text = about_section.find_next('div', class_='inline-show-more-text')
            if about_text:
                profile_data['about'] = about_text.get_text(strip=True)

        # Compile text for processing
        text_parts = [
            profile_data.get('name', ''),
            profile_data.get('headline', ''),
            profile_data.get('about', '')
        ]
        profile_data['text'] = '\n\n'.join(filter(None, text_parts))
        profile_data['title'] = profile_data.get('name', 'LinkedIn Profile')

        logger.info(f"Scraped LinkedIn profile: {profile_data.get('name', 'Unknown')}")
        return profile_data

    def scrape_generic_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a generic webpage.

        Args:
            url: URL to scrape

        Returns:
            Dictionary with page data or None
        """
        use_selenium = self._needs_selenium(url)
        html = self.fetch_with_retry(url, use_selenium=use_selenium)

        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            tag.decompose()

        page_data = {
            'url': url,
            'source': urlparse(url).netloc,
            'type': 'webpage',
            'scraped_at': datetime.now().isoformat()
        }

        # Extract title
        title = soup.find('title')
        page_data['title'] = title.get_text(strip=True) if title else urlparse(url).path

        # Extract main content
        main_content = (soup.find('main') or
                       soup.find('article') or
                       soup.find('div', class_='content') or
                       soup.find('div', id='content') or
                       soup.find('body'))

        if main_content:
            page_data['text'] = main_content.get_text(separator='\n', strip=True)
        else:
            page_data['text'] = soup.get_text(separator='\n', strip=True)

        # Clean text
        page_data['text'] = self._clean_text(page_data['text'])
        page_data['keywords'] = self._extract_keywords(page_data['text'])

        return page_data

    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape any URL, automatically detecting the type.

        Args:
            url: URL to scrape

        Returns:
            Dictionary with scraped data or None
        """
        url_lower = url.lower()

        if 'linkedin.com/pulse' in url_lower or 'linkedin.com/posts' in url_lower:
            return self.scrape_linkedin_article(url)
        elif 'linkedin.com/in/' in url_lower:
            return self.scrape_linkedin_profile(url)
        elif 'linkedin.com' in url_lower:
            return self.scrape_linkedin_article(url)  # Try as article
        else:
            return self.scrape_generic_page(url)

    def scrape_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of scraped data dictionaries
        """
        results = []

        for i, url in enumerate(urls, 1):
            logger.info(f"Processing {i}/{len(urls)}: {url}")

            try:
                data = self.scrape_url(url)
                if data:
                    results.append(data)
                    logger.info(f"  Success: {data.get('title', 'Unknown')[:50]}")
            except Exception as e:
                logger.error(f"  Error scraping {url}: {e}")

        self._close_selenium()
        return results

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if len(line) > 2:
                lines.append(line)

        text = '\n'.join(lines)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        keywords = []
        patterns = [
            r'\b(?:credit risk|riesgo de credito)\b',
            r'\b(?:PD|probability of default)\b',
            r'\b(?:LGD|loss given default)\b',
            r'\b(?:EAD|exposure at default)\b',
            r'\b(?:Basel III|Basel IV|Basilea)\b',
            r'\b(?:CRR|CRD|Capital Requirements)\b',
            r'\b(?:IRB|internal ratings-based)\b',
            r'\b(?:IFRS 9)\b',
            r'\b(?:provision|provision)\b',
            r'\b(?:capital|capital requirement)\b',
            r'\b(?:retail|minorista)\b',
            r'\b(?:corporate|corporativo)\b',
            r'\b(?:SME|PYME)\b',
            r'\b(?:fintech|banking|banca)\b',
            r'\b(?:regulation|regulacion)\b',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.extend(matches)

        return list(set(keywords))[:20]

    def save_results(self, data: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Save scraped data to JSON file.

        Args:
            data: List of scraped data
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scraped_data_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(data)} items to {filepath}")
        return str(filepath)

    def __del__(self):
        """Cleanup when object is destroyed."""
        self._close_selenium()


def scrape_linkedin_urls(urls: List[str], output_dir: str = "data/raw") -> List[Dict[str, Any]]:
    """
    Convenience function to scrape LinkedIn URLs.

    Args:
        urls: List of LinkedIn URLs
        output_dir: Output directory

    Returns:
        List of scraped data
    """
    scraper = EnhancedScraper(output_dir=output_dir)
    results = scraper.scrape_multiple(urls)
    scraper.save_results(results)
    return results


if __name__ == "__main__":
    # Example usage
    print("Enhanced Scraper Test")
    print("=" * 50)

    # Test URLs
    test_urls = [
        # Add your test URLs here
        # "https://www.linkedin.com/pulse/...",
    ]

    if test_urls:
        scraper = EnhancedScraper()
        results = scraper.scrape_multiple(test_urls)
        scraper.save_results(results)
        print(f"\nScraped {len(results)} items")
    else:
        print("No test URLs provided. Add URLs to test_urls list.")
