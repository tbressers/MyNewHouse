#!/usr/bin/env python3
import asyncio
import json
import os
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple
from logging.handlers import RotatingFileHandler

from playwright.async_api import async_playwright
from pushover_utils import send_info_notification, set_dry_run_mode

# Configure root logger to capture logs from all modules and write to file + console
LOG_FILE = Path(__file__).resolve().parent / "app.log"
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000_000, backupCount=5, encoding="utf-8")  # 1GB max
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()  # outputs to stderr
console_handler.setFormatter(formatter)

# Ensure a clean handler set (avoid duplicates on re-run)
root_logger.handlers.clear()
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

STATE_DIR = Path(os.getenv("STATE_DIR", "./state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

PROVIDERS_FILE = Path(__file__).resolve().parent / "providers.json"

# REMOVE static SITES; now loaded from JSON
def load_providers() -> List[Dict]:
    with PROVIDERS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_providers(providers: List[Dict]):
    # Persist updated preferred_method values
    with PROVIDERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(providers, f, ensure_ascii=False, indent=2)

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

async def perform_common_preparation(page, site):
    logger.info(f"Navigating {site['name']} -> {site['url']}")
    await page.goto(site["url"], wait_until=site.get("wait_until", "domcontentloaded"))
    # Consent dismissal
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
                logger.debug(f"{site['name']}: clicked consent button {consent_sel}")
                await asyncio.sleep(0.4)
                break
        except Exception:
            pass
    # Progressive scrolling
    last_height = 0
    for _ in range(12):
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        await asyncio.sleep(0.5)
        height = await page.evaluate("document.body.scrollHeight")
        if height == last_height:
            break
        last_height = height

async def method_primary(page, site) -> List[Dict]:
    selector = site["selector"]
    try:
        await page.wait_for_selector(selector, timeout=8000)
    except Exception:
        logger.warning(f"{site['name']}: primary selector timeout: {selector}")
    elements = await page.query_selector_all(selector)
    items = []
    base_url = page.url
    for el in elements:
        item = {}
        for field in site["item_fields"]:
            sel = field["selector"]
            t = field["type"]
            if t == "text":
                target = await el.query_selector(sel)
                if target:
                    txt = (await target.text_content()) or ""
                    txt = " ".join(txt.split())
                    if txt:
                        item[f"text:{sel}"] = txt
            elif t == "href":
                anchors = await el.query_selector_all(sel)
                for a in anchors:
                    href = await a.get_attribute("href")
                    if href:
                        from urllib.parse import urljoin
                        item[f"href:{href}"] = urljoin(base_url, href)
        if item:
            items.append(item)
    logger.info(f"{site['name']}: primary extracted {len(items)} items")
    return items

async def method_broad_anchor(page, site) -> List[Dict]:
    # Only meaningful for kamernet; for others return empty
    if site["name"] != "kamernet":
        return []
    base_url = page.url
    anchors = await page.query_selector_all("a[href*='/en/for-rent/'], a[href*='/kamer/'], a[href*='/room/']")
    items = []
    for a in anchors:
        href = await a.get_attribute("href")
        txt = (await a.text_content()) or ""
        txt = " ".join(txt.split())
        if href and txt and len(txt) > 3:
            from urllib.parse import urljoin
            items.append({"text:any": txt, f"href:{href}": urljoin(base_url, href)})
    logger.info(f"{site['name']}: broad_anchor extracted {len(items)} items")
    return items

async def method_generic_anchor(page, site) -> List[Dict]:
    base_url = page.url
    anchors = await page.query_selector_all("a[href]")
    items = []
    for a in anchors:
        href = await a.get_attribute("href")
        txt = (await a.text_content()) or ""
        txt = " ".join(txt.split())
        if href and txt and "nijmegen" in (txt.lower() + href.lower()):
            from urllib.parse import urljoin
            items.append({"href:any": urljoin(base_url, href), "text:any": txt})
    logger.info(f"{site['name']}: generic_anchor extracted {len(items)} items")
    return items

METHOD_SEQUENCE = ["primary", "broad_anchor", "generic_anchor"]

async def extract_items(page, site) -> Tuple[List[Dict], str]:
    await perform_common_preparation(page, site)
    preferred = site.get("preferred_method", "primary")
    logger.debug(f"{site['name']}: preferred_method={preferred}")

    async def run(method_name):
        if method_name == "primary":
            return await method_primary(page, site)
        if method_name == "broad_anchor":
            return await method_broad_anchor(page, site)
        if method_name == "generic_anchor":
            return await method_generic_anchor(page, site)
        return []

    # First try preferred
    if preferred in METHOD_SEQUENCE:
        items = await run(preferred)
        if items:
            return items, preferred

    # Preferred failed; try full sequence
    for m in METHOD_SEQUENCE:
        if m == preferred:
            continue
        items = await run(m)
        if items:
            logger.info(f"{site['name']}: switching preferred_method -> {m}")
            return items, m

    # All failed
    if site["name"] == "kamernet":
        # Debug snippet
        html_snip = await page.evaluate("document.querySelector('body').innerHTML.slice(0, 2000)")
        logger.debug(f"kamernet: debug HTML snippet (truncated 2000): {html_snip}")
    logger.error(f"{site['name']}: all methods failed")
    return [], preferred

async def check_site(browser, site) -> Dict:
    context = await browser.new_context()
    page = await context.new_page()
    try:
        items, used_method = await extract_items(page, site)
    except Exception as e:
        logger.error(f"Error checking {site['name']}: {e}")
        return {"site": site["name"], "status": "error", "error": str(e), "items": [], "method": site.get("preferred_method","primary")}
    finally:
        await context.close()

    state_hash = compute_hash(items)
    last_hash, last_items = load_last_state(site["name"])
    has_change = (state_hash != last_hash) and ((len(items)-len(last_items)) > 0)

    if has_change and items:
        save_state(site["name"], state_hash, items)

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
        "new_links": new_links[:10],
        "method": used_method
    }

async def main():
    parser = argparse.ArgumentParser(description="Check housing websites for new listings")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending Pushover notifications")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    if args.dry_run:
        set_dry_run_mode(True)
        logger.info("Running in DRY-RUN mode - no Pushover notifications will be sent")

    try:
        providers = load_providers()
    except Exception as e:
        logger.error(f"Failed to load providers.json: {e}")
        return

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            results = []
            for site in providers:
                res = await check_site(browser, site)
                results.append(res)
                # Update preferred_method if success and method differs
                if res["status"] == "ok" and res["count"] > 0:
                    if site.get("preferred_method") != res["method"]:
                        site["preferred_method"] = res["method"]
            await browser.close()
    except Exception as e:
        logger.error(f"Failed during site checks: {e}")
        return

    # Persist any preferred_method updates
    try:
        save_providers(providers)
    except Exception as e:
        logger.error(f"Failed saving providers.json: {e}")

    changed_sites = [r for r in results if r.get("changed")]
    error_sites = [r for r in results if r.get("status") == "error"]

    if error_sites:
        err_lines = [f"{r['site']}: {r.get('error', 'unknown')}" for r in error_sites]
        logger.error("Errors during run:\n" + "\n".join(err_lines))

    if changed_sites:
        for r in changed_sites:
            for link in r["new_links"]:
                logger.info(f"  {link}")
        msg_lines = []
        for r in changed_sites:
            msg_lines.append(f"\n{r['site']} ({r['method']}): {len(r['new_links'])} new links")
            for link in r["new_links"]:
                msg_lines.append(link)
        message = "\n".join(msg_lines)
        send_info_notification(message, context="New offers")
    else:
        logger.info("No new offers found.")

# Add this entry point:
if __name__ == "__main__":
    asyncio.run(main())
