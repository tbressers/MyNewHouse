#!/usr/bin/env python3
"""
Simplified house scraper v2: Scan provider URLs, extract all links, compare with houses.json
Uses pure regex patterns to detect property listings.
"""

import asyncio
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Set
from logging.handlers import RotatingFileHandler
from urllib.parse import urljoin, urlparse
import re

from playwright.async_api import async_playwright
from pushover_utils import send_info_notification, set_dry_run_mode
from price_extraction import parse_price_to_int, extract_price_from_text, extract_price_from_page

# Configure logging
LOG_FILE = Path(__file__).resolve().parent / "logs/main.log"
HOUSES_LOG_FILE = Path(__file__).resolve().parent / "logs/houses.json"

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000_000, backupCount=5, encoding="utf-8")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

root_logger.handlers.clear()
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Cities to analyze (for regex patterns and filtering)
CITIES = [
    "Nijmegen"
]

# Provider URLs to scan
PROVIDER_URLS = [
    "https://kamernet.nl/en/for-rent/student-housing-nijmegen",
    "https://mijn.sshn.nl/nl-NL/",
    "https://www.wibeco.nl/verhuur/studentenkamers-nijmegen/",
    "https://nijmegenstudentenstad.nl/kamers-in-nijmegen",
    "https://www.pararius.nl/huurwoningen/nijmegen/studentenhuisvesting",
    "https://www.kamernijmegen.com/",
    "https://nymveste.nl/studentenkamer-nijmegen-lingewaard",
    "https://kbsvastgoedbeheer.nl/aanbod/",
    "https://www.klikenhuur.nl/woning-overzicht?cityOrPostalcode=nijmegen&page=1&pagesize=12",
    "https://www.huurzone.nl/huurwoningen/nijmegen?utm_source=daisycon&utm_medium=affiliate&utm_campaign=daisycon_NijmegenStudentenstad.nl"
]

# Provider-specific configuration for filtering and extraction
PROVIDER_CONFIG = {
    "kamernet": {
        "strict_filtering": True,        # High quality listings, keep strict
        "wait_for_js": False,
        # Optional: follow subpages if needed in future
        "follow_subpages": False,
        "subpage_follow_limit": 0,
        "overview_patterns": [],
        "dom_link_href_allow": []
    },
    "mijn": {
        "strict_filtering": True,
        "wait_for_js": False,
        "follow_subpages": False,
        "subpage_follow_limit": 0,
        "overview_patterns": [],
        "dom_link_href_allow": []
    },
    "wibeco": {
        "strict_filtering": True,
        "wait_for_js": False,
        "follow_subpages": False,
        "subpage_follow_limit": 0,
        "overview_patterns": [],
        "dom_link_href_allow": []
    },
    "nijmegenstudentenstad": {
        "strict_filtering": False,       # Portal with minimal info, use relaxed
        "wait_for_js": False,
        "follow_subpages": False,
        "subpage_follow_limit": 0,
        "overview_patterns": [],
        "dom_link_href_allow": []
    },
    "pararius": {
        "strict_filtering": False,       # Client-side rendered, expect minimal text
        "wait_for_js": True,             # Needs JS to render
        # New: follow overview sub-pages and harvest detail links
        "follow_subpages": True,
        "subpage_follow_limit": 5,       # keep small to avoid crawl explosion
        # Treat these as overview pages, not individual listings
        "overview_patterns": [
            r'/huurwoningen/',                                               # general listing root
            r'/huurwoningen/[^/]+/(?:wijk-[^/]+/)?(?:appartement|kamer|studio|woning)/?$'
        ],
        # HREFs that look like actual detail pages (anchors often have no innerText)
        "dom_link_href_allow": [
            r'/[a-z\-]+-te-huur/',                                           # e.g. /appartement-te-huur/
        ]
    },
    "kamernijmegen": {
        "strict_filtering": False,       # News/nav heavy, try relaxed filtering
        "wait_for_js": False,
        "follow_subpages": False,
        "subpage_follow_limit": 0,
        "overview_patterns": [],
        "dom_link_href_allow": []
    },
    "nymveste": {
        "strict_filtering": False,
        "wait_for_js": False,
        "follow_subpages": False,
        "subpage_follow_limit": 0,
        "overview_patterns": [],
        "dom_link_href_allow": []
    },
    "kbsvastgoedbeheer": {
        "strict_filtering": True,
        "wait_for_js": False,
        "follow_subpages": False,
        "subpage_follow_limit": 0,
        "overview_patterns": [],
        "dom_link_href_allow": []
    },
    "klikenhuur": {
        "strict_filtering": True,
        "wait_for_js": False,
        "follow_subpages": False,
        "subpage_follow_limit": 0,
        "overview_patterns": [],
        "dom_link_href_allow": []
    },
    "huurzone": {
        "strict_filtering": True,
        "wait_for_js": False,
        "follow_subpages": False,
        "subpage_follow_limit": 0,
        "overview_patterns": [],
        "dom_link_href_allow": []
    },
}

def get_provider_name(url: str) -> str:
    """Extract provider name from URL"""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "").split(".")[0]
    return domain

def is_overview_subpage(url: str, provider_name: str) -> bool:
    """Detect provider-specific overview/listing pages that group multiple results."""
    cfg = PROVIDER_CONFIG.get(provider_name, {})
    patterns = cfg.get("overview_patterns", []) or []

    if any(re.search(p, url) for p in patterns):
        return True

    # Generic fallbacks for Dutch rental portals
    generic_overview = [
        r'/huurwoningen(?:/|$)',                         # category trees
        r'/for-rent(?:/|$)',
        r'/kamer(?:s)?(?:/|$)$',                         # ends with type
        r'/appartement(?:en)?(?:/|$)$',
        r'/studio(?:s)?(?:/|$)$',
        r'/woning(?:en)?(?:/|$)$'
    ]
    return any(re.search(p, url) for p in generic_overview)

def extract_listing_links(html: str, base_url: str) -> Dict[str, Dict]:
    """
    Extract all links from HTML.
    Returns dict mapping link -> {title, price, date} where available.
    """
    from html.parser import HTMLParser
    
    links_data = {}
    
    class LinkExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.current_text = []
            self.listing_data = {}
            self.in_anchor = False
        
        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            if tag == "a" and "href" in attrs_dict:
                href = attrs_dict["href"]
                if href and not href.startswith("javascript"):
                    full_url = urljoin(base_url, href)
                    self.current_text = []
                    self.in_anchor = True
                    # capture potential title fallbacks (many cards use aria-label/title)
                    fallback = attrs_dict.get("aria-label") or attrs_dict.get("title") or ""
                    self.listing_data = {"link": full_url, "fallback_title": fallback}
            # capture image alt inside an anchor as text
            if tag == "img" and self.in_anchor:
                alt_text = dict(attrs).get("alt")
                if alt_text:
                    self.current_text.append(alt_text.strip())
        
        def handle_data(self, data):
            text = data.strip()
            if self.in_anchor and text:
                self.current_text.append(text)
        
        def handle_endtag(self, tag):
            if tag == "a" and self.listing_data.get("link"):
                self.in_anchor = False
                title = " ".join(self.current_text).strip()
                if not title:
                    # fall back to stored aria-label/title
                    title = self.listing_data.get("fallback_title", "").strip()
                title = " ".join(title.split())

                if title:
                    self.listing_data["title"] = title
                    # Extract price from title
                    price = extract_price_from_text(title)
                    if price > 0:
                        self.listing_data["price"] = price
                    links_data[self.listing_data["link"]] = {k: v for k, v in self.listing_data.items() if k != "fallback_title"}
                
                self.listing_data = {}
                self.current_text = []
    
    try:
        parser = LinkExtractor()
        parser.feed(html)
        return links_data
    except Exception as e:
        logger.warning(f"Error parsing HTML: {e}")
        return {}

async def _extract_links_via_dom(page, base_url: str, provider_name: str) -> Dict[str, Dict]:
    """
    DOM-based extraction to catch anchors without visible text (e.g., listing cards).
    Uses provider-specific allowlist patterns for hrefs.
    """
    cfg = PROVIDER_CONFIG.get(provider_name, {})
    allow_patterns = [re.compile(p) for p in (cfg.get("dom_link_href_allow") or [])]

    try:
        anchors = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                href: a.href,
                text: (a.textContent || '').trim(),
                title: a.getAttribute('title') || '',
                aria: a.getAttribute('aria-label') || ''
            }))
        """)
    except Exception as e:
        logger.debug(f"{provider_name}: DOM link extraction failed: {e}")
        return {}

    base_host = urlparse(base_url).netloc
    out: Dict[str, Dict] = {}
    for a in anchors:
        href = a.get("href") or ""
        if not href:
            continue
        if urlparse(href).netloc and urlparse(href).netloc != base_host:
            continue  # stay on-site

        # If allowlist patterns are defined, keep only those that match
        if allow_patterns and not any(p.search(href) for p in allow_patterns):
            continue

        title = a.get("text") or a.get("aria") or a.get("title") or ""
        title = " ".join(title.split())
        data = {"link": href}
        if title:
            data["title"] = title
            price = extract_price_from_text(title)
            if price > 0:
                data["price"] = price
        out[href] = data

    return out

def is_property_listing(title: str, url: str, strict: bool = True) -> bool:
    """
    Detect if a link is a real property listing using pure regex patterns.
    """
    title = title.strip()
    url = url.lower()
    
    # Pattern 1: Dutch address anywhere in title OR in URL
    street_pattern_with_number = r'(?:new\s+)?[A-Za-z][A-Za-z\s\.\-]*?\s+\d+(?:\s*[-A-Za-z0-9]*)?'
    cities_pattern = '|'.join(CITIES)
    street_pattern_without_number = rf'(?:new\s+)?[A-Za-z][A-Za-z\s\.\-]+\s*-\s*(?:{cities_pattern})'
    
    has_street_in_title = bool(re.search(street_pattern_with_number, title, re.IGNORECASE)) or \
                          bool(re.search(street_pattern_without_number, title, re.IGNORECASE))
    
    has_street_in_url = bool(re.search(r'/kamers-in-nijmegen/[a-z\-]+/\d{4}-\d{2}-\d{2}', url))
    has_street = has_street_in_title or has_street_in_url
    
    # Pattern 2: Location indicator - city name, "te", "in", "bij", etc.
    location_cities = '|'.join(CITIES)
    location_pattern = rf'\b(?:{location_cities}|te|in|bij|at|city|stad)\b'
    has_location = bool(re.search(location_pattern, title, re.IGNORECASE)) or bool(re.search(location_pattern, url))
    
    # Pattern 3: Must have property-related content indicators
    property_indicators = [
        r'\d+\s*m²',
        r'€\s*[\d,\.]+',
        r'\d+\s*(?:/maand|/month|per maand|per month)',
        r'\b(?:room|kamer|studio|apartment|appartement|woning|huis|house|flat|unit|bekijk)\b',
        r'(?:furnished|unfurnished|gemeubileerd|ongemeubileerd)',
        r'(?:beschikbaar|available|available from)',
    ]
    has_property_indicator = any(re.search(pattern, title, re.IGNORECASE) for pattern in property_indicators)
    
    pattern_matches = sum([has_street, has_location, has_property_indicator])
    
    # Pattern 4: URL must NOT be pagination/filter/overview
    bad_url_patterns = [
        r'[\?&]page=',
        r'[\?&]start=',
        r'-overzicht(?:\?|/|$)',
        r'/(?:en|nl)/(?:for-rent|huurwoningen)\s*$',  # Just category, no specific property
        # New: common category endings such as Pararius types (avoid saving overview pages)
        r'/huurwoningen/.*/(?:appartement|kamer|studio|woning)/?$'
    ]
    
    if any(re.search(pattern, url) for pattern in bad_url_patterns):
        return False
    
    required_patterns = 3 if strict else 2
    return pattern_matches >= required_patterns

def filter_property_listings(links: List[Dict], strict: bool = True) -> List[Dict]:
    """
    Filter links to only keep real property listings.
    Uses pure regex patterns without LLM or keyword exclusions.
    
    Args:
        links: List of link dictionaries to filter
        strict: If True, requires all 3 patterns. If False, requires 2/3 patterns.
    """
    if not links:
        return []
    
    filtered_links = []
    for link in links:
        title = link.get("title", "").strip()
        url = link.get("link", "").strip()
        
        if is_property_listing(title, url, strict=strict):
            filtered_links.append(link)
    
    logger.info(f"Filtered to {len(filtered_links)} property listings (from {len(links)} total links)")
    return filtered_links

async def scan_provider_with_retry(browser, url: str, max_retries: int = 2, save_html_dir: str = None) -> tuple[str, List[Dict]]:
    """
    Scan a provider URL with automatic retry on failure.
    
    Args:
        browser: Playwright browser instance
        url: Provider URL to scan
        max_retries: Maximum number of retry attempts (default: 2)
        save_html_dir: Optional directory to save HTML output for debugging
    
    Returns:
        Tuple of (provider_name, list_of_links)
    """
    provider_name = get_provider_name(url)
    
    for attempt in range(max_retries):
        try:
            return await scan_provider(browser, url, save_html_dir=save_html_dir)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, then 2s
                logger.warning(f"{provider_name} attempt {attempt+1}/{max_retries} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"{provider_name}: failed after {max_retries} attempts: {e}")
                return provider_name, []
    
    return provider_name, []

async def scan_provider(browser, url: str, save_html_dir: str = None) -> tuple[str, List[Dict]]:
    """
    Scan a provider URL and extract housing listing links.
    Returns (provider_name, list_of_links_with_metadata)
    """
    provider_name = get_provider_name(url)
    config = PROVIDER_CONFIG.get(provider_name, {"strict_filtering": True, "wait_for_js": False})
    logger.info(f"Scanning {provider_name}: {url}")
    
    context = await browser.new_context()
    page = await context.new_page()
    
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        logger.info(f"{provider_name}: page loaded successfully")
        
        # Wait for dynamic content to render if configured
        if config.get("wait_for_js"):
            try:
                await asyncio.sleep(2)
                await page.wait_for_load_state("networkidle", timeout=3000)
                logger.debug(f"{provider_name}: waited for dynamic content")
            except Exception:
                logger.debug(f"{provider_name}: dynamic content load attempt completed")
                pass
        
        # Try to dismiss consent dialogs
        for consent_sel in [
            "button:has-text('Accept')",
            "button:has-text('Akkoord')",
            "button:has-text('Alles accepteren')",
            "button:has-text('Allow')"
        ]:
            try:
                btn = await page.query_selector(consent_sel)
                if btn:
                    await btn.click()
                    await asyncio.sleep(0.5)
                    logger.debug(f"{provider_name}: dismissed consent dialog")
                    break
            except Exception:
                pass
        
        # Progressive scrolling to load more content
        last_height = 0
        scroll_count = 0
        max_scrolls = 20
        
        while scroll_count < max_scrolls:
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            await asyncio.sleep(0.3)
            height = await page.evaluate("document.body.scrollHeight")
            
            if height == last_height:
                logger.debug(f"{provider_name}: reached end of page after {scroll_count} scrolls")
                break
            
            last_height = height
            scroll_count += 1
        
        # Get full HTML
        html = await page.content()
        logger.info(f"{provider_name}: extracted HTML ({len(html)} bytes)")
        
        # Save HTML for debugging if requested
        if save_html_dir:
            try:
                save_path = Path(save_html_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                filename = save_path / f"{provider_name}_page1.html"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(html)
                logger.info(f"{provider_name}: saved HTML to {filename}")
            except Exception as e:
                logger.error(f"{provider_name}: failed to save HTML: {e}")
        
        # Extract all links from first page (HTML parse + DOM-based for anchors without text)
        all_links = extract_listing_links(html, page.url)
        logger.info(f"{provider_name}: found {len(all_links)} total links on page 1 (HTML parse)")
        try:
            dom_links = await _extract_links_via_dom(page, page.url, provider_name)
            for link in dom_links.values():
                if link["link"] not in all_links:
                    all_links[link["link"]] = link
            logger.info(f"{provider_name}: merged DOM links, now {len(all_links)} links on page 1")
        except Exception as e:
            logger.debug(f"{provider_name}: DOM extraction skipped: {e}")
        
        # Try to follow pagination (up to 3 additional pages)
        pagination_limit = 3
        for page_num in range(2, pagination_limit + 2):
            try:
                # Look for next page link - try different patterns
                next_link = None
                selectors = [
                    f"a:has-text('{page_num}')",
                    f"a[href*='page={page_num}']",
                    "a:has-text('Next')",
                    "a[rel='next']",
                ]
                
                for selector in selectors:
                    next_link = await page.query_selector(selector)
                    if next_link:
                        break
                
                if not next_link:
                    logger.debug(f"{provider_name}: no pagination link found for page {page_num}")
                    break
                
                next_url = await next_link.evaluate("el => el.getAttribute('href')")
                if not next_url:
                    break
                
                next_url = urljoin(page.url, next_url)
                logger.debug(f"{provider_name}: following pagination to page {page_num}")
                
                await page.goto(next_url, wait_until="networkidle", timeout=15000)
                await asyncio.sleep(0.5)
                
                page_html = await page.content()
                
                if save_html_dir:
                    try:
                        filename = Path(save_html_dir) / f"{provider_name}_page{page_num}.html"
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(page_html)
                        logger.debug(f"{provider_name}: saved page {page_num} HTML")
                    except Exception as e:
                        logger.debug(f"{provider_name}: failed to save page {page_num} HTML: {e}")
                
                page_links = extract_listing_links(page_html, page.url)
                try:
                    dom_page_links = await _extract_links_via_dom(page, page.url, provider_name)
                except Exception:
                    dom_page_links = {}
                
                for link in {**page_links, **dom_page_links}.values():
                    if link["link"] not in all_links:
                        all_links[link["link"]] = link
                
            except Exception as e:
                logger.debug(f"{provider_name}: pagination stopped at page {page_num}: {e}")
                break
        
        logger.info(f"{provider_name}: total {len(all_links)} links after pagination")

        # NEW: Follow a small number of overview sub-pages and harvest their listing links
        if config.get("follow_subpages"):
            subpages = [u for u in list(all_links.keys()) if is_overview_subpage(u, provider_name)]
            limit = int(config.get("subpage_follow_limit", 0))
            visited = set()
            for idx, sub_url in enumerate(subpages[:limit]):
                if sub_url in visited:
                    continue
                visited.add(sub_url)
                try:
                    logger.info(f"{provider_name}: expanding overview sub-page {idx+1}/{limit}: {sub_url}")
                    await page.goto(sub_url, wait_until="networkidle", timeout=20000)
                    if config.get("wait_for_js"):
                        try:
                            await asyncio.sleep(1.0)
                            await page.wait_for_load_state("networkidle", timeout=3000)
                        except Exception:
                            pass
                    sub_html = await page.content()
                    sub_links_html = extract_listing_links(sub_html, page.url)
                    try:
                        sub_links_dom = await _extract_links_via_dom(page, page.url, provider_name)
                    except Exception:
                        sub_links_dom = {}
                    # merge
                    for link in {**sub_links_html, **sub_links_dom}.values():
                        if link["link"] not in all_links:
                            all_links[link["link"]] = link

                    # Optional: save subpage HTML
                    if save_html_dir:
                        try:
                            filename = Path(save_html_dir) / f"{provider_name}_sub_{idx+1}.html"
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(sub_html)
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"{provider_name}: failed to expand subpage {sub_url}: {e}")

            logger.info(f"{provider_name}: total {len(all_links)} links after sub-page expansion")

        # Remove overview pages from candidates before filtering
        candidate_links = [
            link for link in all_links.values()
            if not is_overview_subpage(link.get("link", ""), provider_name)
        ]

        # Filter links to property listings FIRST (before fetching prices from pages)
        strict_filtering = config.get("strict_filtering", True)
        filtered_links = filter_property_listings(candidate_links, strict=strict_filtering)
        logger.info(f"{provider_name}: filtered to {len(filtered_links)} property listings (strict={strict_filtering})")
        
        # Extract missing prices from FILTERED listing pages only
        links_needing_price = [link for link in filtered_links if not link.get("price")]
        if links_needing_price:
            logger.info(f"{provider_name}: extracting prices from {len(links_needing_price)} listing pages")
            for link_data in links_needing_price:
                price = await extract_price_from_page(browser, link_data.get("link", ""))
                if price > 0:
                    link_data["price"] = price
                await asyncio.sleep(0.2)  # Rate limiting
        
        return provider_name, filtered_links
        
    except Exception as e:
        logger.error(f"Error scanning {provider_name}: {e}")
        return provider_name, []
    
    finally:
        await context.close()

def load_existing_links() -> Set[str]:
    """
    Load previously saved listing links from houses.json to avoid duplicates.
    Returns a set of link URLs.
    """
    links: Set[str] = set()
    try:
        if HOUSES_LOG_FILE.exists():
            with HOUSES_LOG_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    url = item.get("link", "")
                    if url:
                        links.add(url)
        else:
            logger.info(f"No existing houses.json found at {HOUSES_LOG_FILE}, starting fresh")
    except Exception as e:
        logger.error(f"Failed to load existing links from houses.json: {e}")
    return links

async def main():
    parser = argparse.ArgumentParser(description="Scan housing providers for new listings (v2)")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending Pushover notifications")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--provider", type=str, help="Scan only a specific provider (by name or URL)")
    parser.add_argument("--save-html", type=str, help="Save HTML output to specified directory for debugging")
    args = parser.parse_args()
    
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.dry_run:
        set_dry_run_mode(True)
        logger.info("Running in DRY-RUN mode - no Pushover notifications will be sent")
    
    # Load existing links
    existing_links = load_existing_links()
    logger.info(f"Starting scan with {len(existing_links)} known links")
    
    # Filter providers if specific one requested
    urls_to_scan = PROVIDER_URLS
    if args.provider:
        provider_filter = args.provider.lower()
        urls_to_scan = [
            url for url in PROVIDER_URLS 
            if provider_filter in url.lower() or provider_filter == get_provider_name(url)
        ]
        if not urls_to_scan:
            logger.error(f"No providers matched filter: {args.provider}")
            return
        logger.info(f"Scanning {len(urls_to_scan)} provider(s) matching '{args.provider}'")
    
    # Scan all providers
    all_new_links = []
    provider_results = {}
    
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            
            for url in urls_to_scan:
                provider_name, links = await scan_provider_with_retry(browser, url, save_html_dir=args.save_html)
                
                # Deduplicate links from this provider
                unique_links = {}
                for link_data in links:
                    link_url = link_data.get("link")
                    if link_url and link_url not in unique_links:
                        unique_links[link_url] = link_data
                
                # Find new links
                new_links = [
                    link for link in unique_links.values()
                    if link.get("link") and link["link"] not in existing_links
                ]
                
                provider_results[provider_name] = {
                    "total": len(unique_links),
                    "new": len(new_links),
                    "new_links": new_links
                }
                
                logger.info(f"{provider_name}: {len(new_links)} new listings found (out of {len(unique_links)} total)")
                all_new_links.extend(new_links)
            
            await browser.close()
    
    except Exception as e:
        logger.error(f"Fatal error during scan: {e}")
        return
    
    # Deduplicate all new links globally
    unique_new_links = {}
    for link_data in all_new_links:
        link_url = link_data.get("link")
        if link_url and link_url not in unique_new_links:
            unique_new_links[link_url] = link_data
    
    all_new_links = list(unique_new_links.values())
    
    # Save new links to houses.json
    if all_new_links:
        try:
            HOUSES_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            houses_log = []
            
            if HOUSES_LOG_FILE.exists():
                with HOUSES_LOG_FILE.open("r", encoding="utf-8") as f:
                    houses_log = json.load(f)
            
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Add new links
            for link_data in all_new_links:
                # Find which provider this link came from
                provider_name = "unknown"
                for name, res in provider_results.items():
                    if link_data in res["new_links"]:
                        provider_name = name
                        break
                
                houses_log.append({
                    "timestamp": timestamp,
                    "site": provider_name,
                    "link": link_data.get("link", ""),
                    "title": link_data.get("title", ""),
                    "price": link_data.get("price", ""),
                    "date": link_data.get("date", "")
                })
            
            with HOUSES_LOG_FILE.open("w", encoding="utf-8") as f:
                json.dump(houses_log, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(all_new_links)} new links to houses.json")
        
        except Exception as e:
            logger.error(f"Failed to update houses.json: {e}")
    
    # Send notification
    message_parts = []
    total_new = 0
    
    for provider_name in sorted(provider_results.keys()):
        res = provider_results[provider_name]
        if res["new"] > 0:
            total_new += res["new"]
            message_parts.append(f"\n{provider_name}: {res['new']} new listings")
            for link_data in res["new_links"]:  # Show max 3 per provider
                title = link_data.get("title", "No title")[:60]
                link = link_data.get("link", "")
                message_parts.append(f"  • {title}\n    {link}")
            if res["new"] > 3:
                message_parts.append(f"  ... and {res['new'] - 3} more")
    
    if message_parts:
        message = "\n".join(message_parts)
        logger.info(f"Found {total_new} new listings total")
        send_info_notification(message)
    else:
        logger.info("No new listings found")

if __name__ == "__main__":
    asyncio.run(main())