#!/usr/bin/env python3
"""
House scraper v3: Uses LLM-guided website scanner to discover and extract listings.
Each provider website is analyzed once to determine extraction rules, then used for scraping.

IMPORTANT: This file may not contain provider-specific processing (should be in logs/provider_templates.json).
"""

import asyncio
import json
import logging
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
from logging.handlers import RotatingFileHandler
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright, Browser, Page
from pushover_utils import send_info_notification, set_dry_run_mode
from website_scanner import WebsiteScanner

# Configure logging
LOG_FILE = Path(__file__).resolve().parent / "logs/main.log"
HOUSES_LOG_FILE = Path(__file__).resolve().parent / "logs/houses.json"
PROVIDER_TEMPLATES_FILE = Path(__file__).resolve().parent / "logs/provider_templates.json"

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

# Cities to search for
CITIES = [
    "Nijmegen"
]

# Provider URLs to scan
PROVIDER_URLS = [
    "https://kamernet.nl/en/for-rent/student-housing-nijmegen",
    "https://mijn.sshn.nl/nl-NL/",
    "https://www.wibeco.nl/verhuur/studentenkamers-nijmegen/",
    "https://studentensteden.nl/nijmegen/kamers",
    "https://www.pararius.nl/huurwoningen/nijmegen/studentenhuisvesting",
    "https://www.kamernijmegen.com/",
    "https://kbsvastgoedbeheer.nl/aanbod/",
    "https://www.klikenhuur.nl/woning-overzicht?cityOrPostalcode=nijmegen&page=1&pagesize=12",
]


def normalize_url(url: str) -> str:
    """Normalize URL by removing fragments and sorting query parameters."""
    try:
        from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
        parsed = urlparse(url)
        # Remove fragment
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # No fragment
        ))
        return normalized.rstrip('/')
    except Exception:
        return url.rstrip('/')


def get_provider_name(url: str) -> str:
    """Extract provider name from URL using the second-level domain."""
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    parts = host.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else host


def get_base_url(url: str) -> str:
    """Extract base URL (scheme + netloc) from a full URL."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


async def get_or_create_scan_record(website_url: str, city: str) -> Optional[Dict[str, Any]]:
    """
    Get existing scan record from provider_templates.json or create a new one using website_scanner.
    
    Args:
        website_url: Base URL of the website
        city: City to search for
        
    Returns:
        Scan record with extraction_rules, or None if failed
    """
    base_url = get_base_url(website_url)
    city_lower = city.lower()
    
    # Load existing scan records from provider_templates.json
    if PROVIDER_TEMPLATES_FILE.exists():
        try:
            with PROVIDER_TEMPLATES_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Look for matching scan record
            for record in data:
                if (record.get("base_url") == base_url and 
                    record.get("city", "").lower() == city_lower and
                    record.get("extraction_rules")):
                    logger.info(f"Found existing scan record for {base_url} ({city})")
                    return record
                    
        except Exception as e:
            logger.error(f"Error reading provider_templates.json: {e}")
    
    # No existing record found, create new one
    logger.info(f"No scan record found for {base_url} ({city}), running website_scanner...")
    scanner = WebsiteScanner(website_url, city_lower)
    scan_record = await scanner.scan()
    
    return scan_record


async def extract_listings_from_page(page: Page, scan_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract house listings from a page using the scan record's extraction rules.
    
    Args:
        page: Playwright page object
        scan_record: Record with extraction_rules
        
    Returns:
        List of house dictionaries with title, url, price, address
    """
    rules = scan_record.get("extraction_rules", {})
    if not rules:
        logger.error("No extraction rules in scan record")
        return []
    
    try:
        evaluate_script = r"""
            (rules) => {
                try {
                    const containerSelector = rules.container_selector;
                    const containers = document.querySelectorAll(containerSelector);
                    const results = [];

                    for (const container of containers) {
                        const listing = {};

                        if (rules.title) {
                            if (rules.title.selector) {
                                const titleEl = container.querySelector(rules.title.selector);
                                if (titleEl) {
                                    const attr = rules.title.attribute || 'text';
                                    listing.title = attr === 'text' ? titleEl.innerText.trim() : titleEl.getAttribute(attr);
                                }
                            } else {
                                listing.title = container.innerText.split('\n').map(t => t.trim()).find(Boolean) || '';
                            }
                        }

                        if (rules.url) {
                            if (rules.url.selector) {
                                const urlEl = container.querySelector(rules.url.selector);
                                if (urlEl) {
                                    const attr = rules.url.attribute || 'href';
                                    listing.url = attr === 'href' ? urlEl.href : urlEl.getAttribute(attr);
                                }
                            } else {
                                const attr = rules.url.attribute || 'href';
                                listing.url = attr === 'href' ? container.href : container.getAttribute(attr);
                            }
                        }

                        if (rules.price) {
                            if (rules.price.selector) {
                                const priceEl = container.querySelector(rules.price.selector);
                                if (priceEl) {
                                    listing.price_text = priceEl.innerText.trim();
                                }
                            }

                            if (!listing.price_text) {
                                const text = container.innerText || '';
                                const patternStr = rules.price.pattern || "€?\\s*([0-9][0-9.,]+)";
                                let pattern;
                                try {
                                    pattern = new RegExp(patternStr, 'i');
                                } catch (e) {
                                    pattern = /€?\\s*([0-9][0-9.,]+)/i;
                                }
                                const m = text.match(pattern);
                                if (m) listing.price_text = m[0];
                            }
                        }

                        if (rules.address) {
                            if (rules.address.selector) {
                                const addressEl = container.querySelector(rules.address.selector);
                                if (addressEl) {
                                    const attr = rules.address.attribute || 'text';
                                    listing.address = attr === 'text' ? addressEl.innerText.trim() : addressEl.getAttribute(attr);
                                }
                            }
                            if (!listing.address) {
                                const lines = (container.innerText || '').split('\n').map(l => l.trim()).filter(Boolean);
                                if (lines.length) listing.address = lines[0];
                            }
                        }

                        if (listing.url) {
                            results.push(listing);
                        }
                    }

                    return results;
                } catch (e) {
                    return { __error: e.message || String(e) };
                }
            }
        """

        # Extract listings using the rules
        listings = await page.evaluate(evaluate_script, rules)
        
        if isinstance(listings, dict) and listings.get("__error"):
            logger.error(f"Error inside page.evaluate: {listings['__error']}")
            return []

        # Process listings
        processed_listings = []
        for listing in listings:
            # Parse price to integer
            price = 0
            price_text = listing.get("price_text", "")
            if price_text:
                price = extract_price_from_text(price_text)
            
            # Make URL absolute
            url = listing.get("url", "")
            if url:
                url = urljoin(page.url, url)
                url = normalize_url(url)
            
            # Extract address from URL if configured
            address = listing.get("address", "")
            if not address and rules.get("address", {}).get("extract_from_url"):
                url_instructions = rules["address"].get("url_instructions", "")
                if url_instructions and url:
                    # Parse URL to get path segments
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    path_segments = [s for s in parsed.path.split('/') if s]
                    
                    # Extract based on instructions
                    if path_segments:
                        # Common patterns from instructions
                        if "last path segment" in url_instructions.lower():
                            address = path_segments[-1]
                        elif "second-to-last" in url_instructions.lower() or "second to last" in url_instructions.lower():
                            address = path_segments[-2] if len(path_segments) >= 2 else ""
                        elif "third-to-last" in url_instructions.lower():
                            address = path_segments[-3] if len(path_segments) >= 3 else ""
                        elif "after" in url_instructions.lower() and "city" in url_instructions.lower():
                            # Look for city name in segments
                            city_lower = scan_record.get("city", "").lower()
                            try:
                                city_idx = next(i for i, seg in enumerate(path_segments) if city_lower in seg.lower())
                                if city_idx + 1 < len(path_segments):
                                    address = path_segments[city_idx + 1]
                            except StopIteration:
                                pass
                        elif "after" in url_instructions.lower():
                            # Generic "after" instruction - find specific marker
                            for marker in ["kamer-huren", "huur-", "verhuur", "woning"]:
                                for i, seg in enumerate(path_segments):
                                    if marker.lower() in seg.lower() and i + 1 < len(path_segments):
                                        address = path_segments[i + 1]
                                        break
                                if address:
                                    break
                        
                        if address:
                            # Clean up the address
                            address = address.replace('-', ' ').replace('_', ' ').strip()
                            # Capitalize properly
                            address = ' '.join(word.capitalize() for word in address.split())
            
            processed_listings.append({
                "title": listing.get("title", ""),
                "link": url,
                "price": price if price > 0 else None,
                "address": address
            })
        
        logger.info(f"Extracted {len(processed_listings)} listings from page")
        return processed_listings
        
    except Exception as e:
        logger.error(f"Error extracting listings: {e}", exc_info=True)
        return []


def extract_price_from_text(text: str) -> int:
    """Extract price as integer from text. Returns 0 if no price found."""
    if not text:
        return 0
    
    # Remove common non-numeric characters
    text = text.replace(',', '').replace('.', '')
    
    # Look for patterns like: €800, 800€, EUR 800, 800 per month, etc.
    patterns = [
        r'€\s*(\d+)',
        r'(\d+)\s*€',
        r'EUR\s*(\d+)',
        r'(\d+)\s*EUR',
        r'(\d+)\s*(?:per|p/m|pm)',
        r'(\d{3,})'  # Fallback: any 3+ digit number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                price = int(match.group(1))
                # Sanity check: price should be between 200 and 5000
                if 200 <= price <= 5000:
                    return price
            except (ValueError, IndexError):
                continue
    
    return 0


async def fetch_missing_from_link(browser: Browser, listing: Dict[str, Any], 
                                  scan_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch missing title and/or price information from an individual listing page.
    
    Args:
        browser: Playwright browser instance
        listing: Current listing dict with title, link, price, address
        scan_record: Record with extraction_rules and fallback_rules
        
    Returns:
        Updated listing dict with fetched information
    """
    updated_listing = listing.copy()
    link = listing.get("link", "")
    
    # Check if we have fallback rules
    fallback_rules = scan_record.get("fallback_rules")
    if not fallback_rules:
        return updated_listing
    
    # Check if we need to fetch anything
    needs_title = not listing.get("title")
    needs_price = listing.get("price") is None or listing.get("price") == 0
    
    if not (needs_title or needs_price):
        return updated_listing
    
    if not link:
        return updated_listing
    
    context = None
    page = None
    try:
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        await page.goto(link, wait_until='networkidle', timeout=30000)
        await asyncio.sleep(1)
        
        # Try to dismiss consent dialogs
        for consent_sel in [
            "button:has-text('Accept')",
            "button:has-text('Akkoord')",
            "button:has-text('Alles accepteren')",
            "button:has-text('Allow all')"
        ]:
            try:
                btn = await page.query_selector(consent_sel)
                if btn:
                    await btn.click()
                    await asyncio.sleep(0.5)
                    break
            except Exception:
                pass
        
        # Fetch title if missing
        if needs_title and fallback_rules.get("title"):
            title_rule = fallback_rules["title"]
            title_selector = title_rule.get("selector", "")
            title_attr = title_rule.get("attribute", "text")
            
            if title_selector:
                try:
                    if title_attr == "text":
                        fetched_title = await page.text_content(title_selector)
                    else:
                        fetched_title = await page.get_attribute(title_selector, title_attr)
                    
                    if fetched_title:
                        updated_listing["title"] = fetched_title.strip()
                        logger.debug(f"Fetched title from {link}: {updated_listing['title']}")
                except Exception as e:
                    logger.debug(f"Could not fetch title from {link}: {e}")
        
        # Fetch price if missing
        if needs_price and fallback_rules.get("price"):
            price_rule = fallback_rules["price"]
            price_selector = price_rule.get("selector", "")
            price_pattern = price_rule.get("pattern", r"€?\s*([0-9][0-9.,]+)")
            
            try:
                price_text = None
                if price_selector:
                    # Try to find element with selector
                    try:
                        price_text = await page.text_content(price_selector, timeout=5000)
                        if price_text:
                            logger.debug(f"Found price text with selector '{price_selector}': '{price_text.strip()[:50]}'")
                    except Exception as e:
                        logger.debug(f"Selector '{price_selector}' failed: {e}")
                
                if not price_text:
                    # Fallback: get all text content and search
                    price_text = await page.text_content("body")
                    logger.debug(f"Using body text fallback for price extraction")
                
                if price_text:
                    # Use the same extraction logic as extract_price_from_text
                    fetched_price = extract_price_from_text(price_text)
                    if fetched_price > 0:
                        updated_listing["price"] = fetched_price
                        logger.info(f"✓ Fetched price from {link}: €{fetched_price}")
                    else:
                        logger.debug(f"Could not extract valid price from text: '{price_text[:100]}'")
                else:
                    logger.debug(f"No price text found on page")
            except Exception as e:
                logger.warning(f"Could not fetch price from {link}: {e}")
        
        return updated_listing
        
    except Exception as e:
        logger.error(f"Error fetching missing info from {link}: {e}")
        return updated_listing
    finally:
        if page:
            await page.close()
        if context:
            await context.close()


async def scan_provider(browser: Browser, provider_url: str, city: str) -> tuple[str, List[Dict]]:
    """
    Scan a provider URL and extract all house listings using LLM-discovered extraction rules.
    
    Args:
        browser: Playwright browser instance
        provider_url: URL to scan
        city: City to search for
        
    Returns:
        Tuple of (provider_name, list_of_listings)
    """
    provider_name = get_provider_name(provider_url)
    logger.info(f"Scanning {provider_name}: {provider_url}")
    
    try:
        # Get or create scan record with extraction rules
        scan_record = await get_or_create_scan_record(provider_url, city)
        
        if not scan_record or not scan_record.get("extraction_rules"):
            logger.error(f"{provider_name}: No extraction rules available")
            return provider_name, []
        
        # Use the listing page URL from the scan record
        listing_page_url = scan_record.get("listing_page_url", provider_url)
        
        # Open page and extract listings
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        try:
            await page.goto(listing_page_url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(2)  # Let dynamic content load
            
            # Try to dismiss consent dialogs
            for consent_sel in [
                "button:has-text('Accept')",
                "button:has-text('Akkoord')",
                "button:has-text('Alles accepteren')",
                "button:has-text('Allow all')"
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
            
            # Extract listings
            listings = await extract_listings_from_page(page, scan_record)
            
            # Debug: if no listings found, log the container selector issue
            if not listings:
                rules = scan_record.get("extraction_rules", {})
                container_selector = rules.get("container_selector", "")
                logger.warning(f"{provider_name}: No listings extracted. Testing container selector: {container_selector}")
                try:
                    debug_info = await page.evaluate(f"""
                        (selector) => {{
                            const containers = Array.from(document.querySelectorAll(selector));
                            return {{
                                count: containers.length,
                                samples: containers.slice(0, 2).map(el => ({{
                                    html: el.outerHTML.substring(0, 500),
                                    text: el.innerText.substring(0, 200)
                                }}))
                            }};
                        }}
                    """, container_selector)
                    logger.warning(f"{provider_name}: Container selector matches {debug_info['count']} elements")
                    if debug_info['samples']:
                        logger.warning(f"{provider_name}: Sample container text: {debug_info['samples'][0]['text'][:150]}")
                except Exception as e:
                    logger.warning(f"{provider_name}: Error testing container selector: {e}")
            
            # Fetch missing information from individual listing pages if needed
            if scan_record.get("fallback_rules"):
                listings_with_fallback = []
                for listing in listings:
                    # Check if title or price is missing
                    if not listing.get("title") or listing.get("price") is None or listing.get("price") == 0:
                        logger.debug(f"Fetching missing info from {listing.get('link')}")
                        listing = await fetch_missing_from_link(browser, listing, scan_record)
                    listings_with_fallback.append(listing)
                listings = listings_with_fallback
            else:
                # Even without explicit fallback_rules, try to fetch missing prices from individual pages
                # This provides better coverage when fallback_rules haven't been defined yet
                listings_with_fallback = []
                for listing in listings:
                    if listing.get("price") is None or listing.get("price") == 0:
                        logger.debug(f"Attempting fallback price fetch from {listing.get('link')}")
                        # Create minimal fallback rules for price extraction
                        minimal_rules = {
                            "price": {
                                "selector": "[class*='price'], [class*='prijs'], [class*='rent'], [class*='cost']",
                                "pattern": r"€?\s*([0-9,]+)"
                            }
                        }
                        temp_record = dict(scan_record)
                        temp_record["fallback_rules"] = minimal_rules
                        listing = await fetch_missing_from_link(browser, listing, temp_record)
                    listings_with_fallback.append(listing)
                listings = listings_with_fallback
            
            # Tag with city
            for listing in listings:
                listing["city"] = city
            
            logger.info(f"{provider_name}: extracted {len(listings)} listings")
            return provider_name, listings
            
        finally:
            await context.close()
            
    except Exception as e:
        logger.error(f"Error scanning {provider_name}: {e}", exc_info=True)
        return provider_name, []



def load_existing_links() -> Set[str]:
    """Load previously saved listing links from houses.json to avoid duplicates."""
    links: Set[str] = set()
    try:
        if HOUSES_LOG_FILE.exists():
            with HOUSES_LOG_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    url = item.get("link", "")
                    if url:
                        links.add(normalize_url(url))
        else:
            logger.info(f"No existing houses.json found at {HOUSES_LOG_FILE}, starting fresh")
    except Exception as e:
        logger.error(f"Failed to load existing links from houses.json: {e}")
    return links


async def main():
    parser = argparse.ArgumentParser(description="Scan housing providers for new listings (v3)")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending Pushover notifications")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--provider", type=str, help="Scan only a specific provider (by name or URL)")
    parser.add_argument("--rescan", action="store_true", help="Force rescan of website structures (ignore existing scan records)")
    parser.add_argument("--test", action="store_true", help="Test mode: skip loading existing data, show extracted listings immediately")
    args = parser.parse_args()
    
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.dry_run:
        set_dry_run_mode(True)
        logger.info("Running in DRY-RUN mode - no Pushover notifications will be sent")
    
    # Clear scan records if rescan requested
    if args.rescan:
        logger.info("Rescan requested - will regenerate extraction rules for all sites")
        if PROVIDER_TEMPLATES_FILE.exists():
            try:
                PROVIDER_TEMPLATES_FILE.unlink()
                logger.info(f"Cleared all provider templates from {PROVIDER_TEMPLATES_FILE}")
            except Exception as e:
                logger.error(f"Error clearing provider templates: {e}")
    
    # Load existing links
    existing_links = load_existing_links() if not args.test else set()
    logger.info(f"Starting scan with {len(existing_links)} known links")
    
    # Load all existing data
    houses_log_data: List[Dict] = []
    if HOUSES_LOG_FILE.exists() and not args.test:
        try:
            with HOUSES_LOG_FILE.open("r", encoding="utf-8") as f:
                houses_log_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read houses.json: {e}")
            houses_log_data = []
    
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
    
    # Scan all providers for each city
    all_new_links = []
    provider_results = {}
    
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            
            for city in CITIES:
                logger.info(f"\n{'='*60}")
                logger.info(f"Searching for properties in: {city}")
                logger.info(f"{'='*60}\n")
                
                for url in urls_to_scan:
                    provider_name, listings = await scan_provider(browser, url, city)
                    
                    # In test mode, show extracted listings immediately
                    if args.test:
                        logger.info(f"\n{'='*60}")
                        logger.info(f"TEST MODE - {provider_name} extracted {len(listings)} listings:")
                        logger.info(f"{'='*60}")
                        for i, listing in enumerate(listings[:5], 1):  # Show first 5
                            logger.info(f"{i}. {listing.get('title', 'No title')[:60]}")
                            logger.info(f"   URL: {listing.get('link', 'No URL')}")
                            logger.info(f"   Price: €{listing.get('price', 'N/A')}")
                            logger.info(f"   Address: {listing.get('address', 'N/A')}")
                        if len(listings) > 5:
                            logger.info(f"... and {len(listings) - 5} more")
                        logger.info("")
                    
                    # Find new listings
                    new_listings = []
                    for listing in listings:
                        link = normalize_url(listing.get("link", ""))
                        if link and link not in existing_links:
                            listing["link"] = link  # Store normalized URL
                            new_listings.append(listing)
                    
                    provider_city_key = f"{provider_name} ({city})"
                    provider_results[provider_city_key] = {
                        "total": len(listings),
                        "new": len(new_listings),
                        "new_links": new_listings,
                        "city": city
                    }
                    
                    logger.info(f"{provider_name} ({city}): {len(new_listings)} new listings found (out of {len(listings)} total)")
                    all_new_links.extend(new_listings)
            
            await browser.close()
    
    except Exception as e:
        logger.error(f"Fatal error during scan: {e}", exc_info=True)
        return
    
    # Deduplicate all new links globally
    unique_new_links = {}
    for link_data in all_new_links:
        link_url = normalize_url(link_data.get("link", ""))
        if link_url and link_url not in unique_new_links:
            unique_new_links[link_url] = link_data
    
    all_new_links = list(unique_new_links.values())
    
    # Save updates to houses.json
    if all_new_links and not args.test:
        try:
            HOUSES_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Add new listings to existing data
            for link_data in all_new_links:
                # Find which provider this link came from
                provider_name = "unknown"
                city = link_data.get("city", "")
                for name, res in provider_results.items():
                    if link_data in res["new_links"]:
                        provider_name = name.split(" (")[0] if " (" in name else name
                        city = res.get("city", city)
                        break
                
                houses_log_data.append({
                    "timestamp": timestamp,
                    "site": provider_name,
                    "link": link_data.get("link", ""),
                    "title": link_data.get("title", ""),
                    "price": link_data.get("price"),
                    "address": link_data.get("address", ""),
                    "city": city
                })
            
            with HOUSES_LOG_FILE.open("w", encoding="utf-8") as f:
                json.dump(houses_log_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(all_new_links)} new links to houses.json")
        
        except Exception as e:
            logger.error(f"Failed to update houses.json: {e}")
    
    # Send notification
    message_parts = []
    total_new = 0
    
    for provider_city_key in sorted(provider_results.keys()):
        res = provider_results[provider_city_key]
        if res["new"] > 0:
            total_new += res["new"]
            message_parts.append(f"\n{provider_city_key}: {res['new']} new listings")
            for link_data in res["new_links"][:3]:  # Show max 3 per provider
                title = link_data.get("title", "No title")[:60]
                price = link_data.get("price")
                price_str = f"€{price}" if price else "No price"
                link = link_data.get("link", "")
                message_parts.append(f"  • {title} - {price_str}\n    {link}")
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
