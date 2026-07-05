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

def _eff(p):
    """Effective (actually payable) price — special_price when a genuine discount exists."""
    sp = p.get("special_price") or 0
    return sp if 0 < sp < p["price"] else p["price"]

def _cat(p, sub):
    return sub in (p["name"] + " " + " ".join(p.get("categories", []))).lower()

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
CHEAPEST_BOSCH_DW = min(
    (p for p in PRODUCTS if p["brand"].lower() == "bosch" and _cat(p, "dishwash")),
    key=_eff)

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

def only_products_matching(pred, min_count=1):
    """Scans the answer for SKU-looking tokens that resolve to a real product and
    asserts every one found satisfies `pred`, with at least `min_count` matches —
    a black-box ground-truth check that doesn't depend on which specific product
    the semantic ranker happened to surface (e.g. for "cheapest"/"on offer"/
    "A++ rating" questions where dozens of products could legitimately qualify)."""
    def check(answer, sources):
        found = [BY_SKU[tok] for tok in re.findall(r"[A-Za-z0-9\-]{4,}", answer.upper()) if tok in BY_SKU]
        if len(found) < min_count:
            return False, f"expected at least {min_count} named product(s) with a recognizable SKU, found {len(found)}"
        bad = [p["sku"] for p in found if not pred(p)]
        if bad:
            return False, f"named product(s) that don't match the expected ground truth: {bad}"
        return True, "ok"
    return check

def no_price_above(limit):
    def check(answer, sources):
        for m in re.finditer(r"(\d+(?:\.\d{3})?)\s*BHD", answer):
            if float(m.group(1)) > limit:
                return False, f"quoted a price above {limit} BHD: {m.group(1)}"
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
    # Several Bosch dishwashers tie for cheapest at this effective price, so we assert
    # the true minimum price appears rather than pinning one specific SKU/name.
    ("price:cheapest-dishwasher", "What's your cheapest Bosch dishwasher?",
     [contains("bosch"), contains(fmt(_eff(CHEAPEST_BOSCH_DW)))]),
    ("price:whats-on-sale", "What products are currently on special offer?",
     [only_products_matching(lambda p: (p.get("special_price") or 0) > 0 and p["special_price"] < p["price"])]),
    ("price:under-ceiling", "Show me Bosch ovens under 300 BHD", [contains("bosch"), no_price_above(300)]),

    # --- Stock / availability questions ---
    ("stock:out-of-stock", f"Is the {OUT_OF_STOCK['name']} in stock?", [not_contains("in stock")]),
    ("stock:coming-soon", f"Can I buy the {COMING_SOON['name']} right now?", [not_contains("in stock right now", "available now")]),
    ("stock:in-stock-qty", f"How many {IN_STOCK['name']} do you have in stock?", []),

    # --- Spec questions ---
    ("specs:capacity-color", f"What color and capacity does the {SPECS_RICH['name']} come in?",
     [contains(SPECS_RICH["specs"].get("colour", ""))] if SPECS_RICH["specs"].get("colour") else []),
    ("specs:energy-rating", "Which Bosch appliances have an A++ energy rating?",
     [only_products_matching(lambda p: "A++" in str((p.get("specs") or {}).get("rating", "")).upper())]),

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

# ---------------------------------------------------------------------------
# Multi-turn (conversation memory) cases: (label, first_question, follow_up,
# checks-for-the-follow-up-answer). The first question is asked for real, its
# actual answer becomes history, then the follow-up is asked with that history
# attached — this exercises contextualize() and generate_answer()'s history
# param exactly the way ask.html does, not a hand-built fake history.
# ---------------------------------------------------------------------------
MULTITURN_TESTS = [
    ("multiturn:cheapest-followup",
     "What Bosch dishwashers do you have?", "which is the cheapest?",
     [contains(fmt(_eff(CHEAPEST_BOSCH_DW)))]),
    ("multiturn:cheapest-followup-ar",
     "What Bosch dishwashers do you have?", "كم سعر أرخص واحد؟",
     [has_arabic()]),
    ("multiturn:topic-switch",
     "What Bosch dishwashers do you have?", "What Beko refrigerators are in stock?",
     [contains("beko"), not_contains("dishwash")]),
]


def ask(question: str, history=None):
    t0 = time.time()
    body = {"question": question}
    if history is not None:
        body["history"] = history
    try:
        r = requests.post(f"{SERVER}/ask", json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("answer", ""), data.get("sources", []), time.time() - t0, None
    except Exception as e:
        return "", [], time.time() - t0, str(e)


def run_case(label, question, checks, elapsed, answer, sources, error):
    if error:
        print(f"[ERROR] {label}: {question!r} -> {error}")
        return False

    issues = []
    for check in BASELINE_CHECKS + checks:
        ok, note = check(answer, sources)
        if not ok:
            issues.append(note)

    status = "PASS" if not issues else "FAIL"
    print(f"[{status}] {label} ({elapsed:.1f}s) — {question}")
    if issues:
        for issue in issues:
            print(f"         ! {issue}")
    preview = answer.replace("\n", " ")[:180]
    print(f"         -> {preview}{'...' if len(answer) > 180 else ''}")
    print(f"         sources: {sources}")
    print()
    return not issues


def main():
    total = len(TESTS) + len(MULTITURN_TESTS)
    print(f"Testing against {SERVER} — {total} cases, {len(PRODUCTS)} products loaded\n")
    health = requests.get(f"{SERVER}/health", timeout=10).json()
    print(f"Server: {health}\n")

    passed, failed = 0, 0
    for label, question, checks in TESTS:
        answer, sources, elapsed, error = ask(question)
        if run_case(label, question, checks, elapsed, answer, sources, error):
            passed += 1
        else:
            failed += 1

    for label, first_q, follow_up, checks in MULTITURN_TESTS:
        first_answer, _, _, first_error = ask(first_q)
        if first_error:
            print(f"[ERROR] {label}: first turn {first_q!r} -> {first_error}")
            failed += 1
            continue
        history = [{"role": "user", "content": first_q}, {"role": "assistant", "content": first_answer}]
        answer, sources, elapsed, error = ask(follow_up, history=history)
        if run_case(label, follow_up, checks, elapsed, answer, sources, error):
            passed += 1
        else:
            failed += 1

    print(f"\n{passed}/{total} passed, {failed} failed")


if __name__ == "__main__":
    main()
