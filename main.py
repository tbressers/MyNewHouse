#!/usr/bin/env python3
import asyncio
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple

from playwright.async_api import async_playwright
from pushover_utils import send_info_notification, send_error_notification

STATE_DIR = Path(os.getenv("STATE_DIR", "./state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

SITES = [
    {
        "name": "kamernet",
        "url": "https://kamernet.nl/en/for-rent/student-housing-nijmegen",
        # CSS selector voor listing cards/titels (kan wijzigen; houd in de gaten)
        # Updated: Kamernet uses .SearchResultCard_root class for listing cards
        "selector": ".SearchResultCard_root__hSxn3, a[href*='/for-rent/room-'], a[href*='/for-rent/apartment-'], a[href*='/for-rent/studio-']",
        # Hoe we een listing "samenvatten" (title + link)
        "item_fields": [
            {"type": "text", "selector": ".MuiTypography-subtitle1, .SearchResultCard_contentRow__VZIJY"},
            {"type": "href", "selector": "a"}
        ],
        "wait_until": "networkidle"
    },
    {
        "name": "ssh&",
        "url": "https://www.sshn.nl/ik-zoek-een-kamer/studentenwoning-nijmegen",
        # SSH& laadt via JS; we pakken listing blokken / links in het aanbod
        "selector": "a[href*='/ik-zoek-een-kamer/']",
        "item_fields": [
            {"type": "text", "selector": "a[href*='/ik-zoek-een-kamer/']"}
        ],
        "wait_until": "networkidle"
    },
    {
        "name": "wibeco",
        "url": "https://www.wibeco.nl/verhuur/studentenkamers-nijmegen/",
        "selector": ".objecten .object, .woning-listing, .woning",
        "item_fields": [
            {"type": "text", "selector": ".object .title, .object h3, h2"},
            {"type": "text", "selector": ".object .price, .prijs"}
        ],
        "wait_until": "networkidle"
    },
    {
        "name": "nijmegen-studentenstad",
        "url": "https://nijmegenstudentenstad.nl/kamers-in-nijmegen",
        "selector": "a[href*='kamernet'], a[href*='huurzone'], a[href*='rentbird']",
        "item_fields": [
            {"type": "href", "selector": "a"}
        ],
        "wait_until": "networkidle"
    },
    {
        "name": "pararius",
        "url": "https://www.pararius.nl/huurwoningen/nijmegen/studentenhuisvesting",
        "selector": ".search-list__item, article, [data-testid='search-item']",
        "item_fields": [
            {"type": "text", "selector": ".search-list__item h2, h3 a"},
            {"type": "href", "selector": "a[href*='/kamer/']"},
            {"type": "text", "selector": ".search-list__item .price, .price"}
        ],
        "wait_until": "networkidle"
    },
    {
        "name": "kamernijmegen",
        "url": "https://www.kamernijmegen.com/",
        "selector": "a[href*='/kamer-nijmegen/']",
        "item_fields": [
            {"type": "text", "selector": "a"},
            {"type": "href", "selector": "a[href*='/kamer-nijmegen/']"},
            {"type": "text", "selector": ".price"}
        ],
        "wait_until": "networkidle"
    },
    {
        "name": "nymveste",
        "url": "https://nymveste.nl/studentenkamer-nijmegen-lingewaard",
        "selector": ".woning-item, .listing, article",
        "item_fields": [
            {"type": "text", "selector": "h2, .title"},
            {"type": "href", "selector": "a[href*='/woning/']"},
            {"type": "text", "selector": ".price"}
        ],
        "wait_until": "networkidle"
    },
    {
        "name": "kbsvastgoed",
        "url": "https://kbsvastgoedbeheer.nl/huurwoningen/nijmegen/",
        "selector": ".property-item, .woning, div[class*='listing']",
        "item_fields": [
            {"type": "text", "selector": ".title, h3"},
            {"type": "href", "selector": "a[href*='/huurwoning/']"},
            {"type": "text", "selector": ".price, .huur"}
        ],
        "wait_until": "networkidle"
    },
    {
        "name": "guesthousenijmegen",
        "url": "https://www.guesthousenijmegen.nl/rooms/",
        "selector": ".room-card, .accommodation-item, article",
        "item_fields": [
            {"type": "text", "selector": "h2, .room-title"},
            {"type": "href", "selector": "a[href*='/room/']"},
            {"type": "text", "selector": ".price, .rate"}
        ],
        "wait_until": "networkidle"
    }
]

def compute_hash(items: List[Dict]) -> str:
    # Sorteer deterministisch en hash als state-handtekening
    normalized = json.dumps(sorted(items, key=lambda x: json.dumps(x, sort_keys=True)), ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def load_last_state(site_name: str) -> Tuple[str, List[Dict]]:
    state_file = STATE_DIR / f"{site_name}.json"
    if state_file.exists():
        try:
            data = json.load(state_file.open("r", encoding="utf-8"))
            return data.get("hash", ""), data.get("items", [])
        except Exception:
            return "", []
    return "", []

def save_state(site_name: str, state_hash: str, items: List[Dict]):
    state_file = STATE_DIR / f"{site_name}.json"
    # Use timezone-aware UTC timestamps to avoid deprecation warnings in Python 3.12+
    json.dump({"hash": state_hash, "items": items, "ts": datetime.now(timezone.utc).isoformat()}, state_file.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

async def extract_items(page, site) -> List[Dict]:
    await page.goto(site["url"], wait_until=site.get("wait_until", "domcontentloaded"))
    # scroll om lazy-loaded content te triggeren
    for _ in range(5):
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight/5)")
        await asyncio.sleep(0.5)

    selector = site["selector"]
    elements = await page.query_selector_all(selector)
    items = []
    base_url = page.url  # Get the current page URL for resolving relative links

    for el in elements:
        item = {}
        for field in site["item_fields"]:
            sel = field["selector"]
            t = field["type"]
            target = await el.query_selector(sel)
            if not target:
                continue
            if t == "text":
                txt = (await target.text_content()) or ""
                txt = " ".join(txt.split())
                if txt:
                    # Gebruik veldnaam op basis van selector
                    key = f"text:{sel}"
                    item[key] = txt
            elif t == "href":
                href = await target.get_attribute("href")
                if href:
                    # Convert relative URLs to absolute URLs
                    from urllib.parse import urljoin
                    absolute_url = urljoin(base_url, href)
                    item[f"href:{sel}"] = absolute_url
        if item:
            items.append(item)

    # Fallback: als selector te strikt was, probeer brede anchors
    if not items:
        anchors = await page.query_selector_all("a")
        for a in anchors:
            href = await a.get_attribute("href")
            txt = (await a.text_content()) or ""
            txt = " ".join(txt.split())
            if href and txt and "nijmegen" in (txt.lower() + (href.lower() if href else "")):
                # Convert relative URLs to absolute URLs
                from urllib.parse import urljoin
                absolute_url = urljoin(base_url, href)
                items.append({"href:any": absolute_url, "text:any": txt})
    return items

async def check_site(browser, site) -> Dict:
    context = await browser.new_context()
    page = await context.new_page()

    try:
        items = await extract_items(page, site)
    except Exception as e:
        err = f"Error checking {site['name']}: {e}"
        try:
            send_error_notification(err, context=f"Error: {site['name']}")
        except Exception:
            print(f"[{datetime.now().isoformat()}] Failed to send error notification for {site['name']}")
        return {"site": site["name"], "status": "error", "error": str(e), "items": []}
    finally:
        await context.close()

    state_hash = compute_hash(items)
    last_hash, last_items = load_last_state(site["name"])
    has_change = (state_hash != last_hash) and ((len(items)-len(last_items)) > 0)

    if has_change:
        save_state(site["name"], state_hash, items)
        print(f"  {len(items)-len(last_items)} house(s) added!")

    # Een eenvoudige “diff” op basis van href-velden
    def hrefs(lst):
        hs = set()
        for it in lst:
            for k, v in it.items():
                if k.startswith("href:") or k == "href:any":
                    hs.add(v)
        return hs

    new_links = list(hrefs(items) - hrefs(last_items))

    return {
        "site": site["name"],
        "status": "ok",
        "count": len(items),
        "changed": has_change,
        "new_links": new_links[:10],  # toon max 10 nieuwe links
    }

async def main():
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            results = []
            print("Checking site:")
            for site in SITES:
                print(f"- {site['name']}")
                res = await check_site(browser, site)
                results.append(res)
            await browser.close()
    except Exception as e:
        err = f"Fatal runtime error: {e}"
        try:
            send_error_notification(err, context="Runtime Error")
        except Exception:
            print(f"[{datetime.now().isoformat()}] Failed to send fatal error notification: {e}")
        print(f"[{datetime.now().isoformat()}] Exiting due to fatal error.")
        return

    # Output voor cron/logs
    changed_sites = [r for r in results if r.get("changed")]
    error_sites = [r for r in results if r.get("status") == "error"]

    # Report site-level errors (if any) that weren't already reported
    if error_sites:
        err_lines = [f"{r['site']}: {r.get('error', 'unknown')}" for r in error_sites]
        err_msg = "Errors tijdens run:\n" + "\n".join(err_lines)
        try:
            send_error_notification(err_msg, context="Check Site Errors")
        except Exception:
            print(f"[{datetime.now().isoformat()}] Failed to send aggregated error notification.")

    if changed_sites:
        print(f"[{datetime.now().isoformat()}] Nieuw aanbod gedetecteerd op {len(changed_sites)} site(s):")
        for r in changed_sites:
            print(f"- {r['site']}: {len(r['new_links'])} nieuwe links")
            for link in r["new_links"]:
                print(f"  {link}")

        # Stuur Pushover notificatie met de nieuwe links (full URLs)
        msg_lines = []
        for r in changed_sites:
            msg_lines.append(f"\n{r['site']}: {len(r['new_links'])} nieuwe links")
            for link in r["new_links"]:
                msg_lines.append(link)
        message = "\n".join(msg_lines)
        try:
            send_info_notification(message, context="Nieuw aanbod")
        except Exception as e:
            err = f"Pushover info notify failed: {e}"
            print(err)
            try:
                send_error_notification(err, context="Notification Error")
            except Exception:
                print(f"[{datetime.now().isoformat()}] Failed to send error notification for Pushover failure")
    else:
        print(f"[{datetime.now().isoformat()}] Geen wijzigingen gedetecteerd.")

    # Overzicht (handig voor debugging)
    for r in results:
        print(f"{r['site']}: status={r['status']}, items={r.get('count', 0)}, changed={r.get('changed', False)}")

if __name__ == "__main__":
    asyncio.run(main())
