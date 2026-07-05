"""
Fast pre-check for the structured pre-filter/sort logic in server.py — no server,
no embedding model download. Mirrors effective_price / _name_and_cats /
CATEGORY_MAP / _detect_category / structured_prefilter against the real catalogue
and asserts the documented cases in AI_ASSISTANT_IMPROVEMENTS.md.

Usage: python3 _prefilter_check.py
"""
import json
import re

with open("products_clean.json", "r", encoding="utf-8") as f:
    products = [p for p in json.load(f) if not p.get("status", {}).get("is_test")]

# --- mirrors server.py --------------------------------------------------------

def effective_price(p: dict) -> float:
    sp = p.get("special_price") or 0
    return sp if 0 < sp < p["price"] else p["price"]

def _name_and_cats(p: dict) -> str:
    return (p.get("name", "") + " " + " ".join(p.get("categories", []))).lower()

CATEGORY_MAP = [
    (r"dishwash|غسالة\s*صحون|جلاي", "dishwash"),
    (r"\boven|فرن", "oven"),
    (r"\bhob\b|cooktop|موقد", "hob"),
    (r"fridge|refrigerat|ثلاج", "refriger"),
    (r"washing\s*machine|washer|غسالة\s*ملابس", "laundry"),
    (r"\bdryer\b|نشاف|مجفف", "dryer"),
    (r"hood|chimney|extractor|شفاط|مدخنة|هود", "chimney"),
    (r"microwave|مايكروويف|ميكروويف", "microwave"),
    (r"kettle|غلاي", "kettle"),
    (r"coffee|قهوة|اسبريسو|إسبريسو", "coffee"),
    (r"\bsink\b|حوض|مغسلة", "sink"),
    (r"freezer|فريزر|مجمد", "freezer"),
    (r"toaster|محمصة|توستر", "toaster"),
    (r"blender|خلاط", "blend"),
    (r"vacuum|مكنسة", "vacuum"),
    (r"\biron\b|مكواة", "irons"),
    (r"\bcooker\b|\bstove\b|بوتاجاز", "cooker"),
]

def _detect_category(q: str):
    for trigger, sub in CATEGORY_MAP:
        if re.search(trigger, q):
            return sub
    return None

def structured_prefilter(question: str, base: list):
    q = question.lower()
    pool = list(base)

    cat_requested = _detect_category(q)
    cat_matched = True
    if cat_requested:
        cat_pool = [i for i in pool if cat_requested in _name_and_cats(products[i])]
        if cat_pool:
            pool = cat_pool
        else:
            cat_matched = False

    def soft(pred):
        nonlocal pool
        filtered = [i for i in pool if pred(products[i])]
        if filtered:
            pool = filtered

    if re.search(r"in stock|available|متوفر|بالمخزون", q):
        soft(lambda p: p["status"].get("in_stock") and (p["status"].get("qty") or 0) > 0
                       and not p["status"].get("coming_soon"))
    if re.search(r"\boffers?\b|\bsale\b|discount|deal|عرض|عروض|تخفيض|خصم", q):
        soft(lambda p: (p.get("special_price") or 0) > 0 and p["special_price"] < p["price"])
    cap = re.search(r"(?:under|below|less than|cheaper than|max(?:imum)?|أقل من|تحت|حتى)\s*(\d+(?:\.\d+)?)", q)
    if cap:
        limit = float(cap.group(1)); soft(lambda p: effective_price(p) <= limit)
    rating = re.search(r"\bA\+{1,3}", question) or re.search(r"\bA\b(?=\s*(?:energy|rating|class))", question)
    if rating:
        tok = rating.group(0).strip().upper()
        soft(lambda p: tok in str((p.get("specs") or {}).get("rating", "")).upper())

    sort_dir = None
    if re.search(r"cheapest|lowest\s*price|most affordable|least expensive|أرخص|ارخص|أقل سعر", q):
        sort_dir = "asc"
    elif re.search(r"most expensive|priciest|highest\s*price|dearest|أغلى|أعلى سعر", q):
        sort_dir = "desc"
    if sort_dir and not cat_matched:
        sort_dir = None
    return pool, sort_dir

# Simplified brand mirror (real correct_brand_typos does difflib fuzzy-matching +
# Arabic aliases) — good enough here since every contextualize case below turns
# only on "is a brand or category named at all", not on fuzzy/typo resolution.
BRANDS_SIMPLE = sorted({p["brand"] for p in products if p.get("brand")})

def _detect_brand_simple(q: str):
    upper = q.upper()
    for b in BRANDS_SIMPLE:
        if re.search(rf"\b{re.escape(b.upper())}\b", upper):
            return b
    return None

def contextualize(question: str, history: list) -> str:
    brand = _detect_brand_simple(question)
    if brand or _detect_category(question.lower()):
        return question
    for turn in reversed(history):
        if turn["role"] == "user":
            return turn["content"] + " — " + question
    return question

# --- assertions ----------------------------------------------------------------

def brand_pool(brand):
    return [i for i, p in enumerate(products) if (p.get("brand") or "").lower() == brand.lower()]

failures = []

def check(label, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        failures.append(label)

# 1. Cheapest Bosch dishwasher
pool, sort_dir = structured_prefilter("What's your cheapest Bosch dishwasher?", brand_pool("bosch"))
top = min(pool, key=lambda i: effective_price(products[i]))
check("cheapest bosch dishwasher: sort_dir asc", sort_dir == "asc")
check("cheapest bosch dishwasher: top price == 199.0", effective_price(products[top]) == 199.0)
check("cheapest bosch dishwasher: top is a dishwasher", "dishwash" in _name_and_cats(products[top]))

# 2. Plain category, no superlative
pool, sort_dir = structured_prefilter("What Bosch dishwashers do you have?", brand_pool("bosch"))
check("bosch dishwashers: sort_dir is None", sort_dir is None)
check("bosch dishwashers: all pooled are dishwashers", all("dishwash" in _name_and_cats(products[i]) for i in pool))
check("bosch dishwashers: count == 27", len(pool) == 27)

# 3. Brand + category + stock filter
pool, sort_dir = structured_prefilter("Which Beko refrigerators are in stock?", brand_pool("beko"))
check("beko fridges in stock: all in_stock", all(products[i]["status"].get("in_stock") for i in pool))
check("beko fridges in stock: all refrigerators", all("refriger" in _name_and_cats(products[i]) for i in pool))

# 4. Price ceiling
pool, sort_dir = structured_prefilter("Show me Bosch ovens under 300", brand_pool("bosch"))
check("bosch ovens under 300: all <= 300", all(effective_price(products[i]) <= 300 for i in pool))

# 5. Rating filter
pool, sort_dir = structured_prefilter("Which Bosch appliances have an A++ rating?", brand_pool("bosch"))
check("bosch A++ rating: all specs.rating contains A++",
      all("A++" in str((products[i].get("specs") or {}).get("rating", "")).upper() for i in pool))

# 6. Superlative suppressed when category doesn't match brand
pool, sort_dir = structured_prefilter("What's the cheapest TurboAir dishwasher?", brand_pool("turboair"))
check("turboair cheapest dishwasher: sort_dir suppressed (no dishwashers)", sort_dir is None)

# 7. Arabic superlative + category
pool, sort_dir = structured_prefilter("ما هي أرخص ثلاجة بوش؟", brand_pool("bosch"))
check("arabic cheapest bosch fridge: sort_dir asc", sort_dir == "asc")
check("arabic cheapest bosch fridge: all pooled are refrigerators",
      all("refriger" in _name_and_cats(products[i]) for i in pool))

# 8-12. contextualize (Task 4) — history is [user: "What Bosch dishwashers do you have?"]
H = [{"role": "user", "content": "What Bosch dishwashers do you have?"},
     {"role": "assistant", "content": "..."}]

r = contextualize("which is the cheapest?", H)
check("contextualize: borrows brand+category on bare follow-up",
      "bosch dishwashers" in r.lower() and "cheapest" in r.lower())

r = contextualize("كم سعرها؟", H)
check("contextualize: borrows on Arabic follow-up", "bosch dishwashers" in r.lower())

r = contextualize("What Beko fridges are in stock?", H)
check("contextualize: no borrow on topic switch (brand named)", r == "What Beko fridges are in stock?")

r = contextualize("Do you have kitchen hoods?", H)
check("contextualize: no borrow when category named", r == "Do you have kitchen hoods?")

r = contextualize("which is cheapest?", [])
check("contextualize: passthrough on empty history", r == "which is cheapest?")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    raise SystemExit(1)
print("All checks passed.")
