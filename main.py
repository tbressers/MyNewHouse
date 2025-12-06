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

from playwright.async_api import async_playwright
from pushover_utils import send_info_notification, set_dry_run_mode

# Configure logging
LOG_FILE = Path(__file__).resolve().parent / "logs/main_v2.log"
HOUSES_LOG_FILE = Path(__file__).resolve().parent / "logs/houses.json"
HTML_OUTPUT_DIR = Path(__file__).resolve().parent / "logs/html_dumps"

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

def get_provider_name(url: str) -> str:
    """Extract provider name from URL"""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "").split(".")[0]
    return domain

def load_existing_links() -> Set[str]:
    """Load all existing house links from houses.json"""
    if not HOUSES_LOG_FILE.exists():
        return set()
    
    try:
        with HOUSES_LOG_FILE.open("r", encoding="utf-8") as f:
            houses = json.load(f)
        links = {h.get("link") for h in houses if h.get("link")}
        logger.info(f"Loaded {len(links)} existing links from houses.json")
        return links
    except Exception as e:
        logger.error(f"Failed to load houses.json: {e}")
        return set()

def is_property_listing(title: str, url: str) -> bool:
    """
    Detect if a link is a real property listing using pure regex patterns.
    A listing must have:
    1. Dutch address pattern with street name and number somewhere in title
    2. Location indicators (city name or "te", "in", etc.)
    3. Property indicators (size m², price €, property type)
    """
    import re
    
    title = title.strip()
    url = url.lower()
    
    # Pattern 1: Dutch address anywhere in title
    # Match: "StreetName 123", "StreetName 123-456", "StreetName 123 K5", etc.
    # Allow optional prefix like "New" before the street name
    street_pattern = r'(?:new\s+)?[A-Za-z][A-Za-z\s\.\-]*?\s+\d+(?:\s*[-A-Za-z0-9]*)?'
    
    if not re.search(street_pattern, title, re.IGNORECASE):
        return False
    
    # Pattern 2: Location indicator - city name, "te", "in", "bij", etc.
    # This ensures we're looking at an address, not random digits in title
    location_pattern = r'\b(?:nijmegen|arnhem|wageningen|gennep|oss|te|in|bij|at|city|stad)\b'
    
    if not re.search(location_pattern, title, re.IGNORECASE):
        return False
    
    # Pattern 3: Must have property-related content indicators
    property_indicators = [
        r'\d+\s*m²',                          # Size: "42 m²"
        r'€\s*[\d,\.]+',                      # Price: "€ 1200" or "€1,200.50"
        r'\d+\s*(?:/maand|/month|per maand|per month)', # Price per month
        r'\b(?:room|kamer|studio|apartment|appartement|woning|huis|house|flat|unit)\b',  # Property types
        r'(?:furnished|unfurnished|gemeubileerd|ongemeubileerd)',  # Furnishing
        r'(?:beschikbaar|available|available from)',  # Availability
    ]
    
    has_property_indicator = any(re.search(pattern, title, re.IGNORECASE) for pattern in property_indicators)
    
    if not has_property_indicator:
        return False
    
    # Pattern 4: URL must NOT be pagination/filter/overview
    bad_url_patterns = [
        r'[\?&]page=',           # Pagination: ?page=2
        r'[\?&]start=',          # Pagination: ?start=20
        r'-overzicht(?:\?|/|$)',  # Overview pages
        r'/(?:en|nl)/(?:for-rent|huurwoningen)\s*$',  # Just category, no specific property
    ]
    
    if any(re.search(pattern, url) for pattern in bad_url_patterns):
        return False
    
    return True

def filter_property_listings(links: List[Dict]) -> List[Dict]:
    """
    Filter links to only keep real property listings.
    Uses pure regex patterns without LLM or keyword exclusions.
    """
    if not links:
        return []
    
    filtered_links = []
    for link in links:
        title = link.get("title", "").strip()
        url = link.get("link", "").strip()
        
        if is_property_listing(title, url):
            filtered_links.append(link)
    
    logger.info(f"Filtered to {len(filtered_links)} property listings (from {len(links)} total links)")
    return filtered_links

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
        
        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            if tag == "a" and "href" in attrs_dict:
                href = attrs_dict["href"]
                if href and not href.startswith("javascript"):
                    full_url = urljoin(base_url, href)
                    self.current_text = []
                    self.listing_data = {"link": full_url}
        
        def handle_data(self, data):
            text = data.strip()
            if text and len(text) > 0:
                self.current_text.append(text)
        
        def handle_endtag(self, tag):
            if tag == "a" and self.listing_data.get("link"):
                title = " ".join(self.current_text)
                # Normalize title (remove extra spaces)
                title = " ".join(title.split())
                
                # Keep all links, let LLM decide
                if title:  # Only if there's some text
                    self.listing_data["title"] = title
                    links_data[self.listing_data["link"]] = self.listing_data
                
                self.listing_data = {}
                self.current_text = []
    
    try:
        parser = LinkExtractor()
        parser.feed(html)
        return links_data
    except Exception as e:
        logger.warning(f"Error parsing HTML: {e}")
        return {}

async def scan_provider(browser, url: str, save_html: bool = False) -> tuple[str, List[Dict]]:
    """
    Scan a provider URL and extract housing listing links.
    Returns (provider_name, list_of_links_with_metadata)
    
    Args:
        browser: Playwright browser instance
        url: Provider URL to scan
        save_html: If True, save HTML source to file for verification
    """
    provider_name = get_provider_name(url)
    logger.info(f"Scanning {provider_name}: {url}")
    
    context = await browser.new_context()
    page = await context.new_page()
    
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        logger.info(f"{provider_name}: page loaded successfully")
        
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
        
        # Save HTML to file if requested
        if save_html:
            HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            html_file = HTML_OUTPUT_DIR / f"{provider_name}_{timestamp}.html"
            with html_file.open("w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"{provider_name}: saved HTML to {html_file}")
        
        # Extract all links
        all_links = extract_listing_links(html, page.url)
        logger.info(f"{provider_name}: found {len(all_links)} total links")
        
        # Filter links to property listings only
        filtered_links = filter_property_listings(list(all_links.values()))
        logger.info(f"{provider_name}: filtered to {len(filtered_links)} property listings")
        
        return provider_name, filtered_links
        
    except Exception as e:
        logger.error(f"Error scanning {provider_name}: {e}")
        return provider_name, []
    
    finally:
        await context.close()

async def main():
    parser = argparse.ArgumentParser(description="Scan housing providers for new listings (v2)")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending Pushover notifications")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--provider", type=str, help="Scan only a specific provider (by name or URL)")
    parser.add_argument("--save-html", action="store_true", help="Save HTML source files for verification")
    args = parser.parse_args()
    
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.dry_run:
        set_dry_run_mode(True)
        logger.info("Running in DRY-RUN mode - no Pushover notifications will be sent")
    
    if args.save_html:
        logger.info("HTML output mode enabled - will save HTML source files")
    
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
                provider_name, links = await scan_provider(browser, url, save_html=args.save_html)
                
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
    
    # Send notification (skip if in HTML output mode)
    if not args.save_html:
        message_parts = []
        total_new = 0
        
        for provider_name in sorted(provider_results.keys()):
            res = provider_results[provider_name]
            if res["new"] > 0:
                total_new += res["new"]
                message_parts.append(f"\n{provider_name}: {res['new']} new listings")
                for link_data in res["new_links"]:  # Show max 3 per provider
                    title = link_data.get("title", "No title")[:60]
                    message_parts.append(f"  • {title}")
                if res["new"] > 3:
                    message_parts.append(f"  ... and {res['new'] - 3} more")
        
        if message_parts:
            message = "\n".join(message_parts)
            logger.info(f"Found {total_new} new listings total")
            send_info_notification(message, context="New house offers")
        else:
            logger.info("No new listings found")
    else:
        logger.info("HTML output mode - skipping Pushover notification")

if __name__ == "__main__":
    asyncio.run(main())