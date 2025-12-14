#!/usr/bin/env python3
"""Test price extraction logic"""

import re

def extract_price_from_text(text: str) -> int:
    """Extract price as integer from text. Returns 0 if no price found."""
    if not text:
        return 0
    
    print(f"Input text: '{text}'")
    
    # Remove common non-numeric characters
    text_cleaned = text.replace(',', '').replace('.', '')
    print(f"After removing , and .: '{text_cleaned}'")
    
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
                print(f"Pattern '{pattern}' matched: {match.group(1)} -> {price}")
                # Sanity check: price should be between 200 and 5000
                if 200 <= price <= 5000:
                    print(f"✓ Valid price: {price}")
                    return price
                else:
                    print(f"✗ Price {price} outside range 200-5000")
            except (ValueError, IndexError) as e:
                print(f"✗ Pattern '{pattern}' error: {e}")
                continue
    
    print("✗ No valid price found")
    return 0

# Test cases from wibeco
test_cases = [
    "€ 550,00",
    "€ 595,00",
    "\n                                                € 595,00\n                                        ",
    "€550",
    "550€",
    "550 per month",
]

print("Testing price extraction:")
print("=" * 60)

for test in test_cases:
    print(f"\nTest: {repr(test)}")
    result = extract_price_from_text(test)
    print(f"Result: {result}")
    print("-" * 60)
