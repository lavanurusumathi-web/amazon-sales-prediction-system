"""
Amazon Product Scraper Module

Scrapes live Amazon product data using Selenium + Chrome (headless).
Uses webdriver-manager for automatic chromedriver setup.
"""
import time
import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from threading import Lock

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

REQUEST_DELAY = 2.0


class AmazonScraper:
    def __init__(self, delay: float = REQUEST_DELAY, use_selenium: bool = True):
        self.delay = delay
        self.last_request_time = 0.0
        self._driver = None
        self._lock = Lock()

    def _get_driver(self):
        if self._driver is None:
            with self._lock:
                if self._driver is None:
                    opts = Options()
                    opts.add_argument("--headless=new")
                    opts.add_argument("--no-sandbox")
                    opts.add_argument("--disable-dev-shm-usage")
                    opts.add_argument("--disable-blink-features=AutomationControlled")
                    opts.add_argument("--window-size=1920,1080")
                    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
                    opts.add_experimental_option("useAutomationExtension", False)
                    opts.add_argument(
                        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                    )
                    try:
                        service = Service(ChromeDriverManager().install())
                        self._driver = webdriver.Chrome(service=service, options=opts)
                        self._driver.execute_cdp_cmd(
                            "Page.addScriptToEvaluateOnNewDocument",
                            {"source": """
                                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
                                Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
                            """}
                        )
                        logger.info("Selenium Chrome driver started (headless)")
                    except Exception as e:
                        logger.error(f"Failed to start Chrome driver: {e}")
                        self._driver = None
        return self._driver

    def _polite_wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()

    def _parse_price(self, price_str: Optional[str]) -> Optional[float]:
        if not price_str:
            return None
        cleaned = re.sub(r'[^\d.,]', '', price_str)
        cleaned = cleaned.replace(',', '')
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _parse_rating(self, text: Optional[str]) -> Optional[float]:
        if not text:
            return None
        match = re.search(r'([\d.]+)\s*out\s*of\s*5', text)
        if match:
            return float(match.group(1))
        return None

    def _parse_review_count(self, text: Optional[str]) -> Optional[int]:
        if not text:
            return None
        cleaned = re.sub(r'[^\d]', '', text)
        try:
            return int(cleaned)
        except (ValueError, TypeError):
            return None

    def _scrape_page(self, url: str, wait_selector: str = "[data-asin]") -> Optional[str]:
        driver = self._get_driver()
        if driver is None:
            return None
        try:
            self._polite_wait()
            driver.get(url)
            try:
                WebDriverWait(driver, 8).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
                )
            except TimeoutException:
                pass
            time.sleep(1.5)
            return driver.page_source
        except WebDriverException as e:
            logger.error(f"Selenium error fetching {url}: {e}")
            return None

    def get_product(self, asin: str) -> Dict[str, Any]:
        url = f"https://www.amazon.com/dp/{asin}"
        logger.info(f"Fetching product: {url}")

        html = self._scrape_page(url, wait_selector="#productTitle")
        if html is None:
            return {"error": "Failed to load page (bot detection or network issue)", "asin": asin, "url": url}

        soup = BeautifulSoup(html, "lxml")
        product: Dict[str, Any] = {
            "asin": asin, "url": url,
            "scraped_at": datetime.now().isoformat(), "source": "amazon.com"
        }

        title_elem = soup.select_one("#productTitle") or soup.select_one("span#productTitle")
        if title_elem:
            product["title"] = title_elem.get_text(strip=True)

        price_elem = (
            soup.select_one(".a-price .a-offscreen")
            or soup.select_one("#priceblock_ourprice")
            or soup.select_one("#priceblock_dealprice")
        )
        if price_elem:
            product["price"] = self._parse_price(price_elem.get_text(strip=True))

        list_price_elem = (
            soup.select_one(".a-text-strike")
            or soup.select_one("span.a-text-price span.a-offscreen")
        )
        if list_price_elem:
            product["list_price"] = self._parse_price(list_price_elem.get_text(strip=True))

        if product.get("list_price") and product.get("price"):
            product["discount_pct"] = round(
                (product["list_price"] - product["price"]) / product["list_price"] * 100, 1
            )

        rating_elem = (
            soup.select_one("i.a-icon-star span.a-icon-alt")
            or soup.select_one("span.a-icon-alt")
        )
        if rating_elem:
            product["rating"] = self._parse_rating(rating_elem.get_text(strip=True))

        review_elem = (
            soup.select_one("#acrCustomerReviewText")
            or soup.select_one("span#acrCustomerReviewText")
        )
        if review_elem:
            product["review_count"] = self._parse_review_count(review_elem.get_text(strip=True))

        bsr_section = soup.find(text=re.compile(r"Best Sellers? Rank", re.IGNORECASE))
        if bsr_section:
            bsr_text = bsr_section.find_next().get_text(strip=True) if bsr_section.find_next() else str(bsr_section)
            match = re.search(r'#([\d,]+)\s+in', bsr_text)
            if match:
                product["best_seller_rank"] = int(match.group(1).replace(',', ''))

        breadcrumb = soup.select_one("#wayfinding-breadcrumbs_container")
        if breadcrumb:
            links = breadcrumb.select("a")
            categories = [a.get_text(strip=True) for a in links]
            product["category"] = categories[-1] if categories else None
            product["category_path"] = " > ".join(categories)

        brand_elem = soup.select_one("#bylineInfo") or soup.select_one("a#bylineInfo")
        if brand_elem:
            product["brand"] = brand_elem.get_text(strip=True).replace("Visit the ", "").replace(" Store", "")

        features = soup.select("#feature-bullets ul.a-unordered-list li span.a-list-item")
        if features:
            product["features"] = [f.get_text(strip=True) for f in features if f.get_text(strip=True)]

        avail_elem = soup.select_one("#availability span")
        if avail_elem:
            product["availability"] = avail_elem.get_text(strip=True)

        logger.info(f"Scraped: {product.get('title', 'N/A')} - ${product.get('price', 'N/A')}")
        return product

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        url = f"https://www.amazon.com/s?k={requests_quote(query)}"
        logger.info(f"Searching: {url}")

        html = self._scrape_page(url, wait_selector="[data-asin]")
        if html is None:
            return [{"error": "Failed to load search results (bot detection or network issue)"}]

        soup = BeautifulSoup(html, "lxml")
        results = []

        for item in soup.select('[data-asin]'):
            asin = item.get("data-asin", "").strip()
            if not asin:
                continue

            result = {"asin": asin, "url": f"https://www.amazon.com/dp/{asin}"}

            title_elem = item.select_one("h2 a span") or item.select_one("h2 span")
            if title_elem:
                result["title"] = title_elem.get_text(strip=True)

            price_elem = item.select_one(".a-price .a-offscreen") or item.select_one(".a-price-whole")
            if price_elem:
                result["price"] = self._parse_price(price_elem.get_text(strip=True))

            rating_elem = item.select_one("i.a-icon-star-small span.a-icon-alt") or item.select_one(".a-icon-alt")
            if rating_elem:
                result["rating"] = self._parse_rating(rating_elem.get_text(strip=True))

            review_elem = item.select_one("span.a-size-small span.a-size-base") or item.select_one(".a-size-base.s-underline-text")
            if review_elem:
                result["review_count"] = self._parse_review_count(review_elem.get_text(strip=True))

            img_elem = item.select_one("img.s-image")
            if img_elem:
                result["image_url"] = img_elem.get("src")

            results.append(result)
            if len(results) >= max_results:
                break

        logger.info(f"Found {len(results)} results for '{query}'")
        return results

    def get_top_sellers_in_category(self, category_id: str, max_results: int = 20) -> List[Dict[str, Any]]:
        url = f"https://www.amazon.com/gp/bestsellers/{category_id}"
        logger.info(f"Fetching best sellers: {url}")

        html = self._scrape_page(url, wait_selector="div.p13n-sc-uncoverable-card")
        if html is None:
            return [{"error": "Failed to load best sellers page"}]

        soup = BeautifulSoup(html, "lxml")
        results = []

        items = soup.select("div.p13n-sc-uncoverable-card")
        if not items:
            items = soup.select('[id*="p13n-asin"]')

        for item in items:
            link = item.select_one("a.a-link-normal")
            if not link:
                continue
            href = link.get("href", "")
            asin_match = re.search(r'/dp/([A-Z0-9]{10})', href)
            if not asin_match:
                continue

            result = {"asin": asin_match.group(1)}
            title_elem = item.select_one("div.p13n-sc-truncate") or item.select_one("span[role='text']")
            if title_elem:
                result["title"] = title_elem.get_text(strip=True)

            price_elem = item.select_one(".a-price .a-offscreen") or item.select_one(".a-price-whole")
            if price_elem:
                result["price"] = self._parse_price(price_elem.get_text(strip=True))

            rating_elem = item.select_one("i.a-icon-star-small span.a-icon-alt")
            if rating_elem:
                result["rating"] = self._parse_rating(rating_elem.get_text(strip=True))

            results.append(result)
            if len(results) >= max_results:
                break

        logger.info(f"Found {len(results)} best sellers")
        return results

    def close(self):
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None


def requests_quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s)
