#!/usr/bin/env python3
"""Debug script to test wibeco price extraction"""

import asyncio
import re
from playwright.async_api import async_playwright

async def test_wibeco():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Test URL from houses.json
        test_url = "https://www.wibeco.nl/kamer-huren-nijmegen/graafseweg-41-kamer-01-nijmegen"
        
        print(f"Testing URL: {test_url}")
        print("=" * 60)
        
        await page.goto(test_url, wait_until='networkidle', timeout=30000)
        await asyncio.sleep(2)
        
        # Try to dismiss consent
        try:
            btn = await page.query_selector("button:has-text('Accept'), button:has-text('Akkoord')")
            if btn:
                await btn.click()
                await asyncio.sleep(1)
                print("✓ Dismissed consent dialog")
        except:
            pass
        
        # Test multiple price selectors
        selectors = [
            ".prijs-ruimte",
            "[class*='price']",
            "[class*='prijs']", 
            "[class*='rent']",
            "[class*='cost']",
            ".price",
            ".rent"
        ]
        
        print("\nTesting price selectors:")
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    print(f"  {selector}: '{text}'")
                else:
                    print(f"  {selector}: NOT FOUND")
            except Exception as e:
                print(f"  {selector}: ERROR - {e}")
        
        # Get all text content and search for prices
        print("\nSearching page text for price patterns:")
        body_text = await page.text_content("body")
        
        patterns = [
            (r'€\s*(\d+)', "€X"),
            (r'(\d+)\s*€', "X€"),
            (r'(\d{3,})\s*(?:per|p/m|pm|maand)', "X per month"),
            (r'huur[:\s]+€?\s*(\d+)', "huur: €X"),
            (r'prijs[:\s]+€?\s*(\d+)', "prijs: €X"),
        ]
        
        for pattern, desc in patterns:
            matches = re.findall(pattern, body_text, re.IGNORECASE)
            if matches:
                print(f"  {desc}: {matches[:5]}")  # Show first 5 matches
        
        # Check specific div/span classes
        print("\nChecking HTML structure:")
        html = await page.content()
        
        # Look for price-related class names
        price_classes = re.findall(r'class="([^"]*(?:price|prijs|rent|cost)[^"]*)"', html, re.IGNORECASE)
        if price_classes:
            print(f"  Found price-related classes: {set(price_classes)}")
        
        # Get a snippet of HTML around price mentions
        price_contexts = []
        for match in re.finditer(r'€\s*\d+', html):
            start = max(0, match.start() - 100)
            end = min(len(html), match.end() + 100)
            price_contexts.append(html[start:end])
        
        if price_contexts:
            print(f"\n  HTML context around prices (first match):")
            print(f"  {price_contexts[0][:200]}")
        
        await browser.close()
        
        print("\n" + "=" * 60)
        print("Debug complete!")

if __name__ == "__main__":
    asyncio.run(test_wibeco())
