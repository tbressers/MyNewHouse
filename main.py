#!/usr/bin/env python3
"""
House scraper v3: Uses LLM-guided website scanner to discover and extract listings.
Each provider website is analyzed once to determine extraction rules, then used for scraping.
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
    "https://nijmegenstudentenstad.nl/kamers-in-nijmegen",
    "https://www.pararius.nl/huurwoningen/nijmegen/studentenhuisvesting",
    "https://www.kamernijmegen.com/",
    "https://nymveste.nl/studentenkamer-nijmegen-lingewaard",
    "https://kbsvastgoedbeheer.nl/aanbod/",
    "https://www.klikenhuur.nl/woning-overzicht?cityOrPostalcode=nijmegen&page=1&pagesize=12",
    "https://www.huurzone.nl/huurwoningen/nijmegen?utm_source=daisycon&utm_medium=affiliate&utm_campaign=daisycon_NijmegenStudentenstad.nl"
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
        List of house dictionaries with title, url, price, street
    """
    rules = scan_record.get("extraction_rules", {})
    if not rules:
        logger.error("No extraction rules in scan record")
        return []
    
    try:
        # Extract listings using the rules
        listings = await page.evaluate("""
            (rules) => {
                const container_selector = rules.container_selector;
                const containers = document.querySelectorAll(container_selector);
                
                const results = [];
                
                for (const container of containers) {
                    const listing = {};
                    
                    // Extract title
                    if (rules.title && rules.title.selector) {
                        const titleEl = container.querySelector(rules.title.selector);
                        if (titleEl) {
                            const attr = rules.title.attribute || 'text';
                            listing.title = attr === 'text' ? titleEl.innerText.trim() : titleEl.getAttribute(attr);
                        }
                    }
                    
                    // Extract URL
                    if (rules.url && rules.url.selector) {
                        const urlEl = container.querySelector(rules.url.selector);
                        if (urlEl) {
                            const attr = rules.url.attribute || 'href';
                            listing.url = attr === 'href' ? urlEl.href : urlEl.getAttribute(attr);
                        }
                    }
                    
                    // Extract price
                    if (rules.price && rules.price.selector) {
                        const priceEl = container.querySelector(rules.price.selector);
                        if (priceEl) {
                            listing.price_text = priceEl.innerText.trim();
                        }
                    }
                    
                    // Extract street (optional)
                    if (rules.street && rules.street.selector) {
                        const streetEl = container.querySelector(rules.street.selector);
                        if (streetEl) {
                            const attr = rules.street.attribute || 'text';
                            listing.street = attr === 'text' ? streetEl.innerText.trim() : streetEl.getAttribute(attr);
                        }
                    }
                    
                    // Only add if we have at least a URL
                    if (listing.url) {
                        results.push(listing);
                    }
                }
                
                return results;
            }
        """, rules)
        
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
            
            processed_listings.append({
                "title": listing.get("title", ""),
                "link": url,
                "price": price if price > 0 else None,
                "street": listing.get("street", "")
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
    existing_links = load_existing_links()
    logger.info(f"Starting scan with {len(existing_links)} known links")
    
    # Load all existing data
    houses_log_data: List[Dict] = []
    if HOUSES_LOG_FILE.exists():
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
    if all_new_links:
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
                    "street": link_data.get("street", ""),
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
