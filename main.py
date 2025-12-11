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
from intelligent_classifier import IntelligentClassifier

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

def get_provider_name(url: str) -> str:
    """Extract provider name from URL using the second-level domain.
    Examples:
    - mijn.sshn.nl -> sshn
    - www.pararius.nl -> pararius
    - kamernijmegen.com -> kamernijmegen
    """
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    parts = host.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else host

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
    No longer uses provider-specific allowlist patterns.
    """
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

def filter_property_listings(links: List[Dict], provider_name: str = "") -> List[Dict]:
    """
    Filter links to only keep real property listings using intelligent classification.
    No longer relies on hardcoded patterns or provider-specific configs.
    
    Args:
        links: List of link dictionaries to filter
        provider_name: Provider name for logging purposes
    """
    if not links:
        return []
    
    # Use intelligent classifier
    classifier = IntelligentClassifier()
    
    # Auto-tune threshold based on the link distribution
    # Expected: between 1 and 200 property listings per provider
    threshold = classifier.auto_tune_threshold(links, expected_min=1, expected_max=200)
    
    # Classify and filter
    filtered_links = classifier.batch_classify(links, threshold=threshold)
    
    logger.info(f"{provider_name}: filtered to {len(filtered_links)} property listings (from {len(links)} total links)")
    return filtered_links

async def scan_provider_with_retry(browser, url: str, save_html_dir: str = None, max_retries: int = 2, quicktest: bool = False) -> tuple[str, List[Dict]]:
    """
    Scan a provider URL with automatic retry on failure.
    
    Args:
        browser: Playwright browser instance
        url: Provider URL to scan
        exclude_patterns: Compiled regex patterns for URL exclusion
        save_html_dir: Optional directory to save HTML output for debugging
        max_retries: Maximum number of retry attempts (default: 2)
    
    Returns:
        Tuple of (provider_name, list_of_links)
    """
    provider_name = get_provider_name(url)
    
async def scan_provider_with_retry(browser, url: str, save_html_dir: str = None, max_retries: int = 2, quicktest: bool = False) -> tuple[str, List[Dict]]:
    """
    Scan a provider URL with automatic retry on failure.
    
    Args:
        browser: Playwright browser instance
        url: Provider URL to scan
        save_html_dir: Optional directory to save HTML output for debugging
        max_retries: Maximum number of retry attempts (default: 2)
    
    Returns:
        Tuple of (provider_name, list_of_links)
    """
    provider_name = get_provider_name(url)
    
    for attempt in range(max_retries):
        try:
            return await scan_provider(browser, url, save_html_dir=save_html_dir, quicktest=quicktest)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, then 2s
                logger.warning(f"{provider_name} attempt {attempt+1}/{max_retries} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"{provider_name}: failed after {max_retries} attempts: {e}")
                return provider_name, []
    
    return provider_name, []

async def scan_provider(browser, url: str, save_html_dir: str = None, quicktest: bool = False) -> tuple[str, List[Dict]]:
    """
    Scan a provider URL and extract housing listing links using intelligent classification.
    No longer uses provider-specific configs or exclusion patterns.
    Returns (provider_name, list_of_links_with_metadata)
    """
    provider_name = get_provider_name(url)
    logger.info(f"Scanning {provider_name}: {url}")
    
    context = await browser.new_context()
    page = await context.new_page()
    
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        logger.info(f"{provider_name}: page loaded successfully")
        
        # Always wait a bit for dynamic content - some sites need it
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

        # Convert to list for filtering
        candidate_links = list(all_links.values())

        # Filter links to property listings using intelligent classifier
        filtered_links = filter_property_listings(candidate_links, provider_name=provider_name)
        
        # In quicktest mode, stop after first listing and skip page price extraction
        if quicktest:
            filtered_links = filtered_links[:1]
            return provider_name, filtered_links
        
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
    parser.add_argument("--quicktest", action="store_true", help="Stop after the first house found per provider")
    args = parser.parse_args()
    
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.dry_run:
        set_dry_run_mode(True)
        logger.info("Running in DRY-RUN mode - no Pushover notifications will be sent")
    
    # Load existing links and records
    existing_links = load_existing_links()
    houses_log_data: List[Dict] = []
    if HOUSES_LOG_FILE.exists():
        try:
            with HOUSES_LOG_FILE.open("r", encoding="utf-8") as f:
                houses_log_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read houses.json: {e}")
            houses_log_data = []

    existing_link_map: Dict[str, Dict] = {
        item.get("link", ""): item for item in houses_log_data if item.get("link")
    }
    updated_existing = 0
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
                provider_name, links = await scan_provider_with_retry(browser, url, save_html_dir=args.save_html, quicktest=args.quicktest)

                # Redundant safety: ensure only a single listing in quicktest mode
                if args.quicktest and links:
                    links = links[:1]
                
                # Deduplicate links from this provider
                unique_links = {}
                for link_data in links:
                    link_url = link_data.get("link")
                    if link_url and link_url not in unique_links:
                        unique_links[link_url] = link_data

                    # If we already stored this link earlier but price was missing, backfill it
                    if link_url and link_url in existing_link_map:
                        existing_item = existing_link_map[link_url]
                        if (not existing_item.get("price")) and link_data.get("price"):
                            existing_item["price"] = link_data.get("price")
                            # Optionally refresh title/date if missing
                            if not existing_item.get("title") and link_data.get("title"):
                                existing_item["title"] = link_data.get("title")
                            if not existing_item.get("date") and link_data.get("date"):
                                existing_item["date"] = link_data.get("date")
                            updated_existing += 1
                
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
        logger.error(f"Fatal error during scan: {e}", exc_info=True)
        return
    
    # Deduplicate all new links globally
    unique_new_links = {}
    for link_data in all_new_links:
        link_url = link_data.get("link")
        if link_url and link_url not in unique_new_links:
            unique_new_links[link_url] = link_data
    
    all_new_links = list(unique_new_links.values())
    
    # Save updates to houses.json (new links or backfilled records)
    if all_new_links or updated_existing:
        try:
            HOUSES_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            houses_log = houses_log_data  # reuse and include any backfilled prices

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