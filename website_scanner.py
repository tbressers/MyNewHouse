"""
Website scanner using Playwright and LLM to discover house listings.

This module uses an LLM to intelligently navigate a website to find house listing pages
for a given city, then extracts structured information about each house.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright, Page, Browser
from dotenv import load_dotenv
import openai

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

PROVIDER_TEMPLATES_FILE = Path(__file__).resolve().parent / "logs/provider_templates.json"


class WebsiteScanner:
    """Scans a website to find and extract house listings using LLM guidance."""
    
    def __init__(self, website_url: str, city: str = "nijmegen"):
        """
        Initialize the scanner.
        
        Args:
            website_url: The base URL of the website to scan
            city: The city to search for (default: "nijmegen")
        """
        self.website_url = website_url
        self.city = city
        self.domain = urlparse(website_url).netloc
        
    async def scan(self) -> Dict[str, Any]:
        """
        Main scanning method. Returns a record to be saved in houses.json.
        
        Returns:
            Dict with keys: timestamp, city, base_url, listing_page_url, extraction_rules
        """
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            try:
                # Step 1: Navigate to website and find listing page
                logger.info(f"Navigating to {self.website_url} to find listings for {self.city}")
                listing_page_url = await self._find_listing_page(page)
                
                if not listing_page_url:
                    logger.error(f"Could not find listing page for {self.city} on {self.website_url}")
                    return None
                
                logger.info(f"Found listing page: {listing_page_url}")
                
                # Step 2: Analyze listing page structure
                await page.goto(listing_page_url, wait_until='networkidle', timeout=30000)
                extraction_rules = await self._analyze_listing_structure(page)
                
                if not extraction_rules:
                    logger.error(f"Could not analyze listing structure on {listing_page_url}")
                    return None
                
                # Step 3: Create record
                parsed = urlparse(self.website_url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "city": self.city,
                    "base_url": base_url,
                    "listing_page_url": listing_page_url,
                    "extraction_rules": extraction_rules
                }
                
                # Step 4: Save to houses.json
                self._save_to_log(record)
                
                logger.info(f"Successfully scanned {self.website_url} for {self.city}")
                return record
                
            except Exception as e:
                logger.error(f"Error scanning {self.website_url}: {e}", exc_info=True)
                return None
            finally:
                await browser.close()
    
    async def _find_listing_page(self, page: Page) -> Optional[str]:
        """
        Use LLM to navigate the website and find the listing page for the city.
        
        Returns:
            URL of the listing page, or None if not found
        """
        try:
            # Load the homepage
            await page.goto(self.website_url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(2)  # Let any dynamic content load
            
            # Check if current page already shows listings (verify before asking LLM)
            is_already_listing_page = await self._verify_listing_page(page)
            if is_already_listing_page:
                logger.info(f"The starting URL already appears to be a listing page!")
                return page.url
            
            # Get page content and visible links
            html_content = await page.content()
            links = await self._get_visible_links(page)
            
            # Ask LLM how to navigate to listings
            navigation_prompt = f"""You are helping navigate a real estate website to find house listings.

Website: {self.website_url}
City: {self.city}

Here are the visible links on the homepage:
{self._format_links_for_prompt(links[:50])}  # Limit to first 50 links

Task: Identify which link(s) would lead to rental house listings for {self.city}. 
Look for links related to:
- Rentals ("huur", "rent", "te huur")
- The city name "{self.city}"
- House/apartment listings ("woningen", "kamers", "houses", "apartments")

Respond in JSON format:
{{
    "recommended_link": "the most promising link URL",
    "reason": "brief explanation",
    "alternative_method": "if no direct link exists, describe how to use search/filter boxes"
}}
"""
            
            response = await self._call_llm(navigation_prompt)
            
            if not response:
                return None
            
            # Try the recommended link
            recommended_url = response.get("recommended_link")
            if recommended_url:
                # Ensure it's an absolute URL
                absolute_url = urljoin(self.website_url, recommended_url)
                logger.info(f"LLM recommends: {absolute_url} - {response.get('reason')}")
                
                # Navigate to the recommended page
                await page.goto(absolute_url, wait_until='networkidle', timeout=30000)
                await asyncio.sleep(2)
                
                # Verify this looks like a listing page
                is_listing_page = await self._verify_listing_page(page)
                if is_listing_page:
                    return page.url
                
                # If not, check if we need to go deeper
                links = await self._get_visible_links(page)
                second_prompt = f"""We navigated to {absolute_url} but it doesn't appear to be a listing page yet.

Here are the links on this page:
{self._format_links_for_prompt(links[:30])}

Which link would show the actual list of houses for {self.city}?
Respond with JSON: {{"link": "URL", "reason": "explanation"}}
"""
                
                second_response = await self._call_llm(second_prompt)
                if second_response and second_response.get("link"):
                    second_url = urljoin(page.url, second_response["link"])
                    await page.goto(second_url, wait_until='networkidle', timeout=30000)
                    return page.url
            
            # Try alternative method (search/filter)
            alt_method = response.get("alternative_method")
            if alt_method:
                logger.info(f"Trying alternative method: {alt_method}")
                success = await self._use_search_or_filter(page, alt_method)
                if success:
                    return page.url
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding listing page: {e}", exc_info=True)
            return None
    
    async def _get_visible_links(self, page: Page) -> List[Dict[str, str]]:
        """Extract visible links from the page."""
        try:
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links
                        .filter(a => a.offsetParent !== null)  // visible
                        .map(a => ({
                            text: a.innerText.trim(),
                            href: a.href,
                            title: a.title || ''
                        }))
                        .filter(l => l.text || l.title);
                }
            """)
            return links
        except Exception as e:
            logger.error(f"Error getting links: {e}")
            return []
    
    def _format_links_for_prompt(self, links: List[Dict[str, str]]) -> str:
        """Format links for LLM prompt."""
        formatted = []
        for i, link in enumerate(links, 1):
            text = link.get('text', '')[:100]  # Limit length
            href = link.get('href', '')
            if text and href:
                formatted.append(f"{i}. [{text}] -> {href}")
        return "\n".join(formatted[:50])  # Limit total
    
    async def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call the LLM and parse JSON response."""
        try:
            client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)
            
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes websites to find real estate listings. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            return None
    
    async def _verify_listing_page(self, page: Page) -> bool:
        """Check if the current page appears to be a listing page."""
        try:
            # Look for multiple property listings on the page
            listing_count = await page.evaluate("""
                () => {
                    // Look for common property listing patterns
                    const selectors = [
                        '[class*="listing"]',
                        '[class*="property"]',
                        '[class*="house"]',
                        '[class*="woning"]',
                        '[class*="kamer"]',
                        '[data-testid*="listing"]',
                        'article',
                        '.result-item',
                        '.search-result'
                    ];
                    
                    let maxCount = 0;
                    for (const selector of selectors) {
                        const count = document.querySelectorAll(selector).length;
                        maxCount = Math.max(maxCount, count);
                    }
                    return maxCount;
                }
            """)
            
            # If we find 3+ potential listings, consider it a listing page
            return listing_count >= 3
            
        except Exception as e:
            logger.error(f"Error verifying listing page: {e}")
            return False
    
    async def _use_search_or_filter(self, page: Page, method_description: str) -> bool:
        """Attempt to use search/filter functionality based on LLM guidance."""
        try:
            # This is a simplified implementation
            # Could be enhanced with more sophisticated interaction
            
            # Look for search inputs
            search_input = await page.query_selector('input[type="search"], input[name*="search"], input[placeholder*="search"]')
            if search_input:
                await search_input.fill(self.city)
                await search_input.press('Enter')
                await page.wait_for_load_state('networkidle')
                return True
            
            # Look for city/location filters
            city_filter = await page.query_selector(f'text=/{self.city}/i')
            if city_filter:
                await city_filter.click()
                await page.wait_for_load_state('networkidle')
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error using search/filter: {e}")
            return False
    
    async def _analyze_listing_structure(self, page: Page) -> Optional[Dict[str, Any]]:
        """
        Analyze the listing page to determine how to extract house information.
        
        Returns:
            Dict with extraction rules: selectors/patterns for title, url, price, address
        """
        try:
            # Get page HTML structure with URLs
            html_sample = await page.evaluate("""
                () => {
                    // Find the first few listing items
                    const selectors = [
                        '[class*="listing"]',
                        '[class*="property"]',
                        '[class*="house"]',
                        '[class*="woning"]',
                        '[class*="kamer"]',
                        'article',
                        '.result-item'
                    ];
                    
                    for (const selector of selectors) {
                        const items = Array.from(document.querySelectorAll(selector));
                        if (items.length >= 3) {
                            // Get first 2 items as examples
                            return items.slice(0, 2).map(item => {
                                // Find the main link in this item
                                const link = item.querySelector('a[href]');
                                return {
                                    html: item.outerHTML.substring(0, 2000),
                                    text: item.innerText.substring(0, 500),
                                    url: link ? link.href : ''
                                };
                            });
                        }
                    }
                    return [];
                }
            """)
            
            if not html_sample:
                logger.error("Could not find listing items on page")
                return None
            
            # Ask LLM to analyze the structure
            analysis_prompt = f"""Analyze these house listing HTML samples to determine how to extract information.

Website: {self.website_url}
Listing Page: {page.url}

Sample listings (with their actual URLs):
{json.dumps(html_sample, indent=2)}

Task: Create CSS selectors or patterns to extract the following from each listing:
1. Title (house/apartment name or description)
2. URL (link to full listing)
3. Price (as integer, e.g., €800 per month)
4. Address/street name - IMPORTANT: Look at both the HTML content AND the sample URLs above

For the address field, you have multiple options:
a) If the address is visible in the HTML text: provide a CSS selector
b) If the address is in the URL (look at the sample URLs!): 
   - Set "extract_from_url" to true
   - Provide "url_instructions" explaining how to extract it (e.g., "last path segment", "second-to-last segment", "segment after city name")
   - Examples:
     * URL: /nijmegen/39b3bfca/heeskesacker => "last path segment is the address"
     * URL: /huis-te-huur/amsterdam/van-breestraat => "last path segment is the address"
c) If address is not reliably available: set optional to true and leave selector empty

Respond in JSON format:
{{
    "container_selector": "CSS selector for each listing item",
    "title": {{"selector": "CSS selector relative to container", "attribute": "text or attribute name"}},
    "url": {{"selector": "CSS selector for link", "attribute": "href"}},
    "price": {{"selector": "CSS selector for price", "pattern": "regex to extract number"}},
    "address": {{
        "selector": "CSS selector for address/street (if in HTML, otherwise empty string)", 
        "attribute": "text",
        "extract_from_url": true/false,
        "url_instructions": "human-readable description of how to extract from URL (if extract_from_url is true)",
        "optional": true
    }}
}}

Be specific and use the actual class names and structure you see in the HTML and URLs.
"""
            
            extraction_rules = await self._call_llm(analysis_prompt)
            
            if extraction_rules:
                logger.info(f"LLM generated extraction rules: {json.dumps(extraction_rules, indent=2)}")
                return extraction_rules
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing listing structure: {e}", exc_info=True)
            return None
    
    def _save_to_log(self, record: Dict[str, Any]) -> None:
        """Save the scan record to logs/provider_templates.json."""
        try:
            # Ensure logs directory exists
            PROVIDER_TEMPLATES_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing records
            records = []
            if PROVIDER_TEMPLATES_FILE.exists():
                with open(PROVIDER_TEMPLATES_FILE, 'r', encoding='utf-8') as f:
                    try:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = [records]
                    except json.JSONDecodeError:
                        records = []
            
            # Append new record
            records.append(record)
            
            # Save back
            with open(PROVIDER_TEMPLATES_FILE, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved scan record to {PROVIDER_TEMPLATES_FILE}")
            
        except Exception as e:
            logger.error(f"Error saving to log: {e}", exc_info=True)


async def scan_website(website_url: str, city: str = "nijmegen") -> Optional[Dict[str, Any]]:
    """
    Convenience function to scan a website.
    
    Args:
        website_url: The base URL of the website to scan
        city: The city to search for (default: "nijmegen")
    
    Returns:
        Dict with scan results, or None if failed
    """
    scanner = WebsiteScanner(website_url, city)
    return await scanner.scan()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    # Get URL from command line or use default
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.pararius.nl"
    city = sys.argv[2] if len(sys.argv) > 2 else "nijmegen"
    
    # Run the scanner
    result = asyncio.run(scan_website(url, city))
    
    if result:
        print(f"\nSuccess! Scan results:")
        print(json.dumps(result, indent=2))
    else:
        print(f"\nFailed to scan {url}")
        sys.exit(1)
