#!/usr/bin/env python3
"""
Price extraction utilities for housing listings.
Handles parsing and extracting prices from various formats and page sources.
"""

import re
import logging
import asyncio

logger = logging.getLogger(__name__)


def parse_price_to_int(price_str: str) -> int:
    """
    Convert a price string (with € and separators) to an integer.
    Handles: €500, €500,-, €500.00, €1.200, €1,200.50, etc.
    
    Returns the integer price or 0 if parsing fails.
    """
    if not price_str:
        return 0
    
    # Remove € symbol and whitespace
    price_str = price_str.replace("€", "").strip()
    
    # Remove the dash after comma if present (e.g., "500,-" becomes "500")
    price_str = price_str.rstrip("-")
    
    # Determine if comma or period is the thousands separator vs decimal separator
    # Dutch format: 1.200,50 (period for thousands, comma for decimals)
    # Alternative format: 1,200.50 (comma for thousands, period for decimals)
    
    comma_count = price_str.count(",")
    period_count = price_str.count(".")
    
    if comma_count == 1 and period_count == 1:
        # Both present: determine which is which
        comma_pos = price_str.index(",")
        period_pos = price_str.index(".")
        
        if comma_pos > period_pos:
            # Dutch format: 1.200,50 - period is thousands, comma is decimal
            price_str = price_str.replace(".", "").replace(",", ".")
        else:
            # US format: 1,200.50 - comma is thousands, period is decimal
            price_str = price_str.replace(",", "")
    elif comma_count == 1:
        # Only comma: could be thousands or decimal
        # If more than 2 chars after comma, it's thousands separator
        after_comma = len(price_str.split(",")[1])
        if after_comma > 2:
            price_str = price_str.replace(",", "")
        else:
            # It's a decimal separator
            price_str = price_str.replace(",", ".")
    elif period_count == 1:
        # Only period: could be thousands or decimal
        after_period = len(price_str.split(".")[1])
        if after_period > 2:
            price_str = price_str.replace(".", "")
        else:
            # It's a decimal separator - keep it as is
            pass
    
    # Remove any remaining non-digit, non-period characters
    price_str = re.sub(r"[^\d.]", "", price_str)
    
    try:
        # Convert to float then to int (truncating decimals)
        price_float = float(price_str) if price_str else 0
        return int(price_float)
    except ValueError:
        return 0


def extract_price_from_text(text: str) -> int:
    """
    Extract price from text using regex patterns and convert to integer.
    Matches: €500, €500,-, €500.00, €1.200, €1,200.50, etc.
    
    Returns the price as an integer or 0 if not found.
    """
    # Pattern matches: € followed by digits (at least 2-3 digits for realistic prices)
    # with optional separators (. , or -) and decimals
    # Examples: €500, €500,-, €1.200, €1,200.50, €500.00
    # This avoids matching things like "€ 15" from option values
    price_pattern = r'€\s*(?:\d{2,}[\d.,\-]*)'
    match = re.search(price_pattern, text)
    if match:
        price_str = match.group(0).strip()
        return parse_price_to_int(price_str)
    return 0


async def extract_price_from_page(browser, url: str) -> int:
    """
    Open a URL and extract the listing price from the page.
    Extracts prices from listing-specific elements, avoiding filter/option prices.
    Prioritizes: page title -> structured data (JSON-LD) -> metadata -> content
    
    Args:
        browser: Playwright browser instance
        url: URL to open
    
    Returns:
        Price as integer if found, 0 otherwise
    """
    context = None
    try:
        context = await browser.new_context()
        page = await context.new_page()
        
        await page.goto(url, wait_until="networkidle", timeout=10000)
        
        # For dynamic sites, wait a bit for JavaScript to render content
        await asyncio.sleep(1)
        
        html = await page.content()
        
        # Priority 1: Extract price from page <title> tag
        # Example: <title>Huis te huur Veldstraat in Nijmegen voor €&nbsp;1.975</title>
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            title_text = title_match.group(1)
            price = extract_price_from_text(title_text)
            if price > 100:
                logger.debug(f"Extracted price from {url} title: €{price}")
                return price
        
        # Priority 2: Extract price from JSON-LD structured data
        # Look for "price" field in JSON-LD schema
        json_ld_match = re.search(r'"price"\s*:\s*"([\d.]+)"', html)
        if json_ld_match:
            price_str = json_ld_match.group(1)
            try:
                price = int(float(price_str))
                if price > 100:
                    logger.debug(f"Extracted price from {url} JSON-LD: €{price}")
                    return price
            except ValueError:
                pass
        
        # Priority 3: Extract from meta description
        meta_desc_match = re.search(r'<meta\s+name="description"\s+content="([^"]*€[^"]*)"', html, re.IGNORECASE)
        if meta_desc_match:
            desc_text = meta_desc_match.group(1)
            price = extract_price_from_text(desc_text)
            if price > 100:
                logger.debug(f"Extracted price from {url} meta description: €{price}")
                return price
        
        # Priority 4: Look for price patterns in common listing element contexts
        price_patterns = [
            r'(?:€|price|huur)\s*[\d.,\-]+\s*(?:pm|maand|month)',  # Price per month
            r'<[^>]*(?:price|amount|rental)[^>]*>.*?€\s*[\d.,\-]+',  # In price-specific elements
            r'(?:Huurprijs|Prijs|Price).*?€\s*[\d.,\-]+',  # Common labels
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                price_str = match.group(0)
                price = extract_price_from_text(price_str)
                if price > 100:  # Filter out small prices from filters/options
                    logger.debug(f"Extracted price from {url} pattern: €{price}")
                    return price
        
        # Priority 5: Extract any price larger than 100 (as fallback)
        # Prioritize larger prices to avoid picking up smaller component fees
        prices_found = []
        price_pattern = r'€\s*[\d.,\-]+'
        for match in re.finditer(price_pattern, html):
            price_str = match.group(0)
            price = extract_price_from_text(price_str)
            if price > 100:  # Assume prices under 100 are filter options
                prices_found.append(price)
        
        if prices_found:
            # Return the largest price found (likely the main listing price)
            price = max(prices_found)
            logger.debug(f"Extracted price from {url} content (largest of {len(prices_found)}): €{price}")
            return price
        
        return 0
    
    except Exception as e:
        logger.debug(f"Failed to extract price from page {url}: {e}")
        return 0
    
    finally:
        if context:
            await context.close()
