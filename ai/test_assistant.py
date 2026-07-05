"""
End-to-end test suite for the showroom AI assistant.

Runs a broad battery of realistic customer questions against the live
/ask endpoint and checks the answers against ground truth pulled directly
from products_clean.json — covering every field (price, special price,
stock, specs, description, brand, category) and every question style
(brand lookup, category lookup, price, stock, specs, SKU, comparisons,
out-of-scope brands, typos, Arabic) a customer is likely to ask.

Usage:
    python3 test_assistant.py [server_url]

Requires the server to already be running (see 3_start_server.sh/.bat).
"""
import json
import re
import sys
import time

import requests

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

with open("products_clean.json", "r", encoding="utf-8") as f:
    PRODUCTS = [p for p in json.load(f) if not p.get("status", {}).get("is_test")]

BY_SKU = {p["sku"]: p for p in PRODUCTS}
BRANDS = sorted({p["brand"] for p in PRODUCTS if p.get("brand")})

def fmt(v):
    return f"{float(v):.3f}"

# ---------------------------------------------------------------------------
# Pick real products to anchor field-specific questions against ground truth.
# ---------------------------------------------------------------------------
OUT_OF_STOCK = next(p for p in PRODUCTS if not p["status"]["in_stock"] and p["brand"])
COMING_SOON = next(p for p in PRODUCTS if p["status"]["coming_soon"] and p["brand"])
SPECIAL_OFFER = next(p for p in PRODUCTS if p["special_price"] and 0 < p["special_price"] < p["price"] and p["brand"])
SPECS_RICH = next(p for p in PRODUCTS if len(p.get("specs") or {}) >= 4 and p["brand"])
# Excludes coming_soon because some source rows set both in_stock and coming_soon
# true at once (e.g. TFB3302GB) — an unambiguous "available now" pick needs both.
IN_STOCK = next(p for p in PRODUCTS
                if p["status"]["in_stock"] and p["status"]["qty"] and not p["status"]["coming_soon"] and p["brand"])

# ---------------------------------------------------------------------------
# Test cases: (label, question, checks) — checks is a list of callables
# that take (answer_text, sources_list) and return (passed: bool, note: str).
# ---------------------------------------------------------------------------
def contains(*substrings, ci=True):
    def check(answer, sources):
        hay = answer.lower() if ci else answer
        for s in substrings:
            needle = s.lower() if ci else s
            if needle not in hay:
                return False, f"missing expected text: {s!r}"
        return True, "ok"
    return check

def not_contains(*substrings, ci=True):
    def check(answer, sources):
        hay = answer.lower() if ci else answer
        for s in substrings:
            needle = s.lower() if ci else s
            if needle in hay:
                return False, f"contains forbidden text: {s!r}"
        return True, "ok"
    return check

def has_arabic():
    def check(answer, sources):
        arabic_chars = sum(1 for ch in answer if '؀' <= ch <= 'ۿ')
        letters = sum(1 for ch in answer if ch.isalpha())
        if letters > 5 and arabic_chars < letters * 0.3:
            return False, "answer is not predominantly Arabic"
        return True, "ok"
    return check

def no_comma_price():
    def check(answer, sources):
        if re.search(r"\d{1,3},\d{3}\s*BHD", answer):
            return False, "price has thousands-comma formatting bug"
        return True, "ok"
    return check

def no_bracket_leak():
    def check(answer, sources):
        if re.search(r"\[[A-Z]{2,}\]", answer):
            return False, "leaked internal [BRAND] bracket tag"
        return True, "ok"
    return check

def expect_source(*brands, ci=True):
    """Asserts the /ask response's `sources` list names (only) the expected
    brand(s) — a direct regression check for brand mis-attribution bugs."""
    def check(answer, sources):
        want = {b.lower() if ci else b for b in brands}
        got = {s.lower() if ci else s for s in sources}
        if not want & got:
            return False, f"expected source {sorted(want)!r}, got {sources!r}"
        return True, "ok"
    return check

def no_invented_price():
    """PDF-fallback brands (Nolte/Express/Cosentino/Poggenpohl) have no live price
    data — the model must direct the customer in-store instead of inventing a
    number, per the system prompt's explicit instruction."""
    def check(answer, sources):
        if re.search(r"\d+\.\d{3}\s*BHD", answer):
            return False, "invented a price for a brand with no live pricing data"
        return True, "ok"
    return check

# A sample of the untranslated German compound words actually present in the Nolte
# catalogue text — regression check for the system prompt's instruction that PDF
# excerpt language (German/Italian/French) must be translated, not copied verbatim.
GERMAN_LEAK_WORDS = ["landhausstil", "softmatt", "wohnliche", "buffetschrank", "küchenplanung"]

def no_untranslated_german():
    def check(answer, sources):
        hay = answer.lower()
        for w in GERMAN_LEAK_WORDS:
            if w in hay:
                return False, f"leaked untranslated German text: {w!r}"
        return True, "ok"
    return check

BASELINE_CHECKS = [no_comma_price(), no_bracket_leak()]

TESTS = [
    # --- Category questions, per brand (English) ---
    ("category:bosch:dishwasher", "What Bosch dishwashers do you have?", [contains("bosch")]),
    ("category:beko:refrigerator", "What Beko refrigerators are available?", [contains("beko")]),
    ("category:elba:oven", "What ovens does Elba make?", [contains("elba")]),
    ("category:elica:hood", "What kitchen hoods does Elica offer?", [contains("elica")]),
    ("category:ukinox:sink", "What sinks does UKINOX have?", [contains("ukinox")]),
    ("category:turboair:vacuum", "Does TurboAir make vacuum cleaners?", []),

    # --- Category across brands (no brand named) ---
    ("category:any:washing-machine", "What washing machines do you sell?", []),
    ("category:any:coffee-machine", "Do you have any coffee machines?", []),
    ("category:any:kettle", "What kettles are in stock?", []),

    # --- Price questions ---
    # The actually-payable price is special_price when one exists — the model
    # quoting only that (not also the pre-discount price) is correct, not a bug.
    ("price:special-offer", f"How much does the {SPECIAL_OFFER['name']} cost?",
     [contains(fmt(SPECIAL_OFFER["special_price"]))]),
    ("price:in-stock", f"What is the price of {IN_STOCK['name']}?", [contains(fmt(IN_STOCK["price"]))]),
    ("price:cheapest-dishwasher", "What's your cheapest Bosch dishwasher?", [contains("bosch")]),
    ("price:whats-on-sale", "What products are currently on special offer?", []),

    # --- Stock / availability questions ---
    ("stock:out-of-stock", f"Is the {OUT_OF_STOCK['name']} in stock?", [not_contains("in stock")]),
    ("stock:coming-soon", f"Can I buy the {COMING_SOON['name']} right now?", [not_contains("in stock right now", "available now")]),
    ("stock:in-stock-qty", f"How many {IN_STOCK['name']} do you have in stock?", []),

    # --- Spec questions ---
    ("specs:capacity-color", f"What color and capacity does the {SPECS_RICH['name']} come in?",
     [contains(SPECS_RICH["specs"].get("colour", ""))] if SPECS_RICH["specs"].get("colour") else []),
    ("specs:energy-rating", "Which Bosch appliances have an A+ energy rating?", []),

    # --- SKU lookup ---
    ("sku:exact", f"Do you have SKU {SPECS_RICH['sku']} and what does it cost?",
     [contains(fmt(SPECS_RICH["special_price"] or SPECS_RICH["price"]))]),
    ("sku:lowercase", f"do you have sku {SPECS_RICH['sku'].lower()}?",
     [contains(fmt(SPECS_RICH["special_price"] or SPECS_RICH["price"]))]),

    # --- Combined price + stock in one question ---
    # Effective price is special_price when one exists (same rule as the other price
    # tests above) — quoting only that, not the pre-discount price too, is correct.
    ("combo:price-and-stock", f"Is the {IN_STOCK['name']} in stock and how much does it cost?",
     [contains(fmt(IN_STOCK["special_price"] or IN_STOCK["price"]))]),

    # --- Out-of-scope brands (must not hallucinate) ---
    # Nolte/Cosentino/Poggenpohl used to be the out-of-scope example brands here,
    # but they're now covered by the PDF fallback (see the pdf: tests below) — these
    # use brands genuinely absent from both products_clean.json and pdfs/.
    ("oos:miele", "What kitchen cabinets does Miele offer?",
     [not_contains("miele offers", "miele has", "miele sells")]),
    ("oos:samsung", "What Samsung refrigerator colors do you have?",
     [not_contains("samsung offers", "samsung has")]),
    ("oos:whirlpool", "Show me Whirlpool kitchen designs.", []),

    # --- Typo / fuzzy brand matching ---
    ("typo:bosh", "What bosh dishwashers do you have?", [contains("bosch")]),
    ("typo:beco", "beco fridge price", []),
    ("typo:all-caps", "BOSCH DISHWASHER PRICE", [contains("bosch")]),

    # --- Off-topic / general ---
    ("general:greeting", "Hello, what products do you sell?", []),
    ("general:comparison", "What's the difference between your Bosch and Beko dishwashers?", []),

    # --- Arabic equivalents ---
    ("ar:beko-fridge-price", "ما هي أسعار ثلاجات بيكو المتوفرة؟", [has_arabic(), contains("بيكو")]),
    ("ar:bosch-alias", "بوش عندكم غسالات صحون؟", [has_arabic()]),
    ("ar:oos-nolte", "هل تبيعون منتجات نولته؟", [has_arabic()]),
    ("ar:stock-question", f"هل يوجد {IN_STOCK['name']} متوفر؟", [has_arabic()]),

    # --- PDF-fallback brands (Nolte/Express/Cosentino/Poggenpohl — not in
    # products_clean.json, answered from ai/pdf_kb.json instead) ---
    ("pdf:cosentino-silestone", "What Silestone colors do you have?",
     [expect_source("Cosentino")]),
    ("pdf:nolte-cabinets", "What kitchen cabinet styles does Nolte offer?",
     [contains("nolte"), expect_source("Nolte"), no_untranslated_german()]),
    ("pdf:poggenpohl-en", "Tell me about Poggenpohl kitchen designs.",
     [contains("poggenpohl"), expect_source("Poggenpohl")]),
    # Express has no direct test elsewhere in this file — it's the one PDF brand
    # never exercised outside the "must not leak" negative check below.
    ("pdf:express-kuchen", "What kitchen designs does Express offer?",
     [contains("express"), expect_source("Express")]),
    # Sub-brand aliases (SUB_BRAND_ALIASES in server.py) — these product-line names
    # share no characters with "COSENTINO" so difflib alone would never match them.
    ("pdf:cosentino-dekton", "What Dekton finishes are available?",
     [expect_source("Cosentino")]),
    ("pdf:cosentino-sensa", "Tell me about Sensa countertops.",
     [expect_source("Cosentino")]),
    # PDF-fallback brands carry no live price data — the model must direct the
    # customer in-store rather than invent a number (see system prompt in server.py).
    ("pdf:nolte-no-invented-price", "How much does a Nolte kitchen cost?",
     [expect_source("Nolte"), no_invented_price()]),
    # Brand name glued directly to "؟" (no space) — the natural way to write Arabic,
    # and a regression check for a real bug where this broke ARABIC_BRAND_ALIASES
    # lookup, silently misattributing the answer to an unrelated appliance brand.
    ("pdf:nolte-ar-no-space", "ما هي أنماط خزائن المطبخ التي تقدمها نولته؟",
     [has_arabic(), contains("نولته"), expect_source("Nolte")]),
    ("pdf:express-ar-alias", "ما هي تصاميم مطابخ اكسبرس؟",
     [has_arabic(), expect_source("Express")]),
    ("pdf:cosentino-silestone-ar-alias", "ما هي ألوان سيليستون المتوفرة؟",
     [has_arabic(), expect_source("Cosentino")]),
    # PDF brands must not leak into an ordinary appliance question, and vice versa.
    ("pdf:no-leak-into-appliance", "What Bosch dishwashers do you have?",
     [not_contains("nolte", "cosentino", "poggenpohl", "express küchen")]),

    # --- Edge cases ---
    ("edge:empty-question", "", []),
]


def ask(question: str):
    t0 = time.time()
    try:
        r = requests.post(f"{SERVER}/ask", json={"question": question}, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("answer", ""), data.get("sources", []), time.time() - t0, None
    except Exception as e:
        return "", [], time.time() - t0, str(e)


def main():
    print(f"Testing against {SERVER} — {len(TESTS)} cases, {len(PRODUCTS)} products loaded\n")
    health = requests.get(f"{SERVER}/health", timeout=10).json()
    print(f"Server: {health}\n")

    passed, failed = 0, 0
    for label, question, checks in TESTS:
        answer, sources, elapsed, error = ask(question)
        if error:
            print(f"[ERROR] {label}: {question!r} -> {error}")
            failed += 1
            continue

        all_checks = BASELINE_CHECKS + checks
        issues = []
        for check in all_checks:
            ok, note = check(answer, sources)
            if not ok:
                issues.append(note)

        status = "PASS" if not issues else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {label} ({elapsed:.1f}s) — {question}")
        if issues:
            for issue in issues:
                print(f"         ! {issue}")
        preview = answer.replace("\n", " ")[:180]
        print(f"         -> {preview}{'...' if len(answer) > 180 else ''}")
        print(f"         sources: {sources}")
        print()

    print(f"\n{passed}/{len(TESTS)} passed, {failed} failed")


if __name__ == "__main__":
    main()
