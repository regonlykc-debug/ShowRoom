import difflib
import json
import os
import re
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# The site (index.html, assets/, pdfs/) lives one directory up from ai/.
SITE_ROOT = ".."

KB_FILE = "product_kb.json"
PDF_KB_FILE = "pdf_kb.json"

# NVIDIA's OpenAI-compatible API (free tier). meta/llama-3.1-8b-instruct is used
# instead of the larger deepseek-v4 / 70B+ models on this same endpoint — those
# were measured timing out (>45s, likely oversubscribed on the free tier) while
# this one answers reliably in ~3-5s with correct English AND Arabic output.
# Checked qwen/qwen2.5-7b-instruct as a stronger-Arabic alternative (2026-07-05):
# not served on this endpoint — the only Qwen models available are 80B/122B/397B
# MoE models, well outside kiosk-latency budget. Kept llama; see Arabic-language
# rule in generate_answer() instead.
MODEL = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Reads NVIDIA_API_KEY from the environment — set it on the kiosk PC before
# starting this server (see README.md). Get a free key at build.nvidia.com.
llm_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=os.environ.get("NVIDIA_API_KEY"), timeout=30, max_retries=1)

# MUST match index_products.py. e5 needs "query: " on questions and
# "passage: " on documents (products were already prefixed at index time).
EMBED_MODEL = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "

# Products are compact, structured summaries (not page-length catalogue
# text), so we can afford to surface more of them per answer than the old
# PDF-chunk pipeline did.
TOP_K = 10
BRAND_MATCH_CUTOFF = 0.75    # high enough to avoid unrelated words matching a brand name

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading product knowledge base...")
with open(KB_FILE, "r", encoding="utf-8") as f:
    products = json.load(f)

embeddings_matrix = np.array([p["embedding"] for p in products], dtype=np.float32)
# Vectors are stored normalized by the indexer, but re-normalize defensively in case
# an older product_kb.json (built with a different model) is loaded.
norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
embeddings_matrix = embeddings_matrix / np.clip(norms, 1e-10, None)

BRANDS = sorted({p["brand"].upper() for p in products if p.get("brand")})
BRAND_INDICES = {b: [i for i, p in enumerate(products) if p.get("brand", "").upper() == b] for b in BRANDS}
SKU_INDEX = {p["sku"].upper(): i for i, p in enumerate(products) if p.get("sku")}

# Cabinetry/surface brands (Nolte, Express, Cosentino, Poggenpohl) have no rows in
# products_clean.json — they only exist as PDF catalogues. Loaded as a secondary,
# brand-scoped fallback: only pulled in when one of these specific brands is named,
# never mixed into general appliance searches (mixing PDF prose into unrelated
# product queries is exactly the quality problem the JSON export was built to fix).
pdf_chunks = []
if os.path.exists(PDF_KB_FILE):
    print("Loading PDF knowledge base (Nolte/Express/Cosentino/Poggenpohl)...")
    with open(PDF_KB_FILE, "r", encoding="utf-8") as f:
        pdf_chunks = json.load(f)

if pdf_chunks:
    pdf_embeddings_matrix = np.array([c["embedding"] for c in pdf_chunks], dtype=np.float32)
    pdf_norms = np.linalg.norm(pdf_embeddings_matrix, axis=1, keepdims=True)
    pdf_embeddings_matrix = pdf_embeddings_matrix / np.clip(pdf_norms, 1e-10, None)
else:
    print("No pdf_kb.json found — Nolte/Express/Cosentino/Poggenpohl won't be answerable. Run index_pdf_brands.py to enable.")
    pdf_embeddings_matrix = np.zeros((0, embeddings_matrix.shape[1]), dtype=np.float32)

PDF_BRANDS = sorted({c["brand"] for c in pdf_chunks})
PDF_BRAND_INDICES = {b: [i for i, c in enumerate(pdf_chunks) if c["brand"] == b] for b in PDF_BRANDS}
PDF_TOP_K = 6

# Used only for typo/fuzzy brand-name matching — includes PDF-only brands so a
# customer naming "Poggenphol" or "Nolta" still resolves correctly.
ALL_BRANDS = sorted(set(BRANDS) | set(PDF_BRANDS))

# Sub-brand / product-line names customers actually say, which don't fuzzy-match
# their parent brand string at all (e.g. "Silestone" vs "COSENTINO" share no
# characters in common, so difflib would never catch it).
SUB_BRAND_ALIASES = {"SILESTONE": "COSENTINO", "DEKTON": "COSENTINO", "SENSA": "COSENTINO"}

print(f"Loading embedding model: {EMBED_MODEL}")
embed_model = SentenceTransformer(EMBED_MODEL)
print("Server ready!")

class Turn(BaseModel):
    role: str      # "user" or "assistant"
    content: str

class Question(BaseModel):
    question: str
    language: str = "auto"
    history: list[Turn] = []

# Arabic transliterations of brand names — the Latin-script typo correction below can't
# see these at all, which was silently sending Arabic brand questions to a full, mixed
# search instead of the named brand's own products.
ARABIC_BRAND_ALIASES = {
    "بوش": "BOSCH", "بوشية": "BOSCH",
    "بيكو": "BEKO",
    "البا": "ELBA", "إلبا": "ELBA",
    "اليكا": "ELICA", "إليكا": "ELICA", "الاليكا": "ELICA",
    "يوكينوكس": "UKINOX", "اوكينوكس": "UKINOX",
    "توربو اير": "TURBOAIR", "توربوير": "TURBOAIR",
    "نولته": "NOLTE", "نولت": "NOLTE",
    "اكسبرس": "EXPRESS", "إكسبرس": "EXPRESS",
    "كوزنتينو": "COSENTINO", "كوسنتينو": "COSENTINO",
    "سيليستون": "COSENTINO", "سيلستون": "COSENTINO", "ديكتون": "COSENTINO",
    "بوجنبول": "POGGENPOHL", "بوغنبول": "POGGENPOHL",
}

def correct_brand_typos(question: str):
    """Replace a misspelled brand name with its correct spelling (e.g. 'boash' -> 'BOSCH'),
    so a lone typo'd brand name still embeds close to that brand's products, and return
    which brand (if any) was mentioned."""
    detected_brand = []  # list so the closure below can assign into it

    def repl(m):
        word = m.group(0)
        if len(word) < 3:
            return word
        upper = word.upper()
        if upper in SUB_BRAND_ALIASES:
            detected_brand.append(SUB_BRAND_ALIASES[upper])
            return word
        match = difflib.get_close_matches(upper, ALL_BRANDS, n=1, cutoff=BRAND_MATCH_CUTOFF)
        if match:
            detected_brand.append(match[0])
            return match[0]
        return word

    corrected = re.sub(r"[A-Za-z]+", repl, question)

    # Arabic letters only (ء-ي) — the broader ؀-ۿ block used elsewhere for
    # language detection also contains punctuation like ؟ (U+061F), which glues onto
    # the preceding word with no space (the normal way to write Arabic) and silently
    # broke the alias lookup, e.g. "نولته؟" never matching the dict key "نولته".
    for word in re.findall(r"[ء-ي]+", question):
        if word in ARABIC_BRAND_ALIASES:
            detected_brand.append(ARABIC_BRAND_ALIASES[word])

    return corrected, (detected_brand[0] if detected_brand else None)

def find_sku_match(question: str):
    """A staff member typing (or a barcode scanner entering) an exact SKU should always
    surface that product, regardless of how it scores semantically."""
    for word in re.findall(r"[A-Za-z0-9\-]{4,}", question.upper()):
        if word in SKU_INDEX:
            return SKU_INDEX[word]
    return None

# ---------------------------------------------------------------------------
# Structured pre-filter / sort. The product KB has clean fields (price,
# special_price, stock, specs.rating, categories), but a pure semantic top-K
# can't answer "cheapest", "in stock", "on offer", "under 300 BHD" or "A++
# rating" — the right rows simply may not be in the semantic top-K. So for
# those intents we filter/sort on the real fields first, then let the dense
# search rank whatever remains.
# ---------------------------------------------------------------------------
SUPERLATIVE_K = 5

def effective_price(p: dict) -> float:
    sp = p.get("special_price") or 0
    return sp if 0 < sp < p["price"] else p["price"]

def _name_and_cats(p: dict) -> str:
    return (p.get("name", "") + " " + " ".join(p.get("categories", []))).lower()

# (regex that may appear in the QUESTION) -> (substring present in the catalogue's
# name/category taxonomy). Arabic triggers map to the English catalogue term.
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
    # "irons" (plural, the category-tree term), not "iron" — the singular also
    # matches "Cast Iron" hob/grate descriptions, which aren't clothes irons.
    (r"\biron\b|مكواة", "irons"),
    (r"\bcooker\b|\bstove\b|بوتاجاز", "cooker"),
]

def _detect_category(q: str):
    for trigger, sub in CATEGORY_MAP:
        if re.search(trigger, q):
            return sub
    return None

def structured_prefilter(question: str, base: list):
    """Return (narrowed_pool, sort_dir). Category is a SOFT filter (never claim a
    brand lacks a product just because the taxonomy names it differently), but if a
    named category matches nothing we suppress superlative sorting so we don't hand
    the model 'the cheapest hood' as the answer to 'the cheapest dishwasher'."""
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

def contextualize(question: str, history: list) -> str:
    """Follow-ups drop the brand/category ('which is cheapest?'). If the current
    question carries neither, borrow the most recent user question, so brand
    filtering, category prefiltering, and the dense embedding all see the topic.
    Only retrieval uses this — the model still sees the real question.

    Known limitation: this only reaches one question back. A 3-hop chain
    ("Show me Bosch ovens" -> "under 300?" -> "and the black one?") can lose the
    thread on the third turn. The robust fix is an LLM query-rewrite step, but
    that's a second model call (~3-5s) per question — not acceptable kiosk
    latency, so this deterministic single-hop borrow is what ships."""
    _, brand = correct_brand_typos(question)
    if brand or _detect_category(question.lower()):
        return question
    for turn in reversed(history):
        if turn.role == "user":
            return turn.content + " — " + question
    return question

def find_relevant_products(question: str, top_k: int = TOP_K):
    # A named brand (even misspelled) should only pull products from that brand —
    # otherwise a global search gets drowned out by whichever brand has the most
    # products (Bosch has 317 of 531). A brand that's PDF-only (e.g. Cosentino)
    # has no entry in BRAND_INDICES, so this correctly yields zero products.
    corrected_question, brand = correct_brand_typos(question)
    base = list(BRAND_INDICES.get(brand, [])) if brand else list(range(len(products)))
    if not base:
        return [], brand

    pool, sort_dir = structured_prefilter(question, base)
    if not pool:
        return [], brand

    sku_idx = find_sku_match(question)

    # Superlative ("cheapest" / "most expensive"): answer from the true sorted
    # extreme, not a semantic guess.
    if sort_dir:
        ranked = sorted(pool, key=lambda i: effective_price(products[i]),
                        reverse=(sort_dir == "desc"))[:SUPERLATIVE_K]
        if sku_idx is not None and sku_idx not in ranked:
            ranked = [sku_idx] + ranked[:SUPERLATIVE_K - 1]
        return [{"product": products[i], "score": effective_price(products[i])} for i in ranked], brand

    # Otherwise dense-rank WITHIN the filtered pool, so category/stock/offer/price
    # filters sharpen the semantic search instead of competing with it.
    q_emb = embed_model.encode(
        [QUERY_PREFIX + corrected_question], normalize_embeddings=True
    )[0].astype(np.float32)
    scores = embeddings_matrix @ q_emb
    ranked = sorted(pool, key=lambda i: -scores[i])[:top_k]
    if sku_idx is not None and sku_idx not in ranked:
        ranked = [sku_idx] + ranked[:top_k - 1]
    return [{"product": products[i], "score": float(scores[i])} for i in ranked], brand

def find_relevant_pdf_chunks(question: str, top_k: int = PDF_TOP_K):
    """Only searches the PDF fallback brands (Nolte, Express, Cosentino, Poggenpohl),
    and only when one of them is explicitly named — never mixed into a general
    appliance search, which is exactly the failure mode the JSON export replaced."""
    if not pdf_chunks:
        return [], None

    corrected_question, brand = correct_brand_typos(question)
    if not brand or brand not in PDF_BRAND_INDICES:
        return [], brand
    candidate_indices = PDF_BRAND_INDICES[brand]

    q_emb = embed_model.encode(
        [QUERY_PREFIX + corrected_question], normalize_embeddings=True
    )[0].astype(np.float32)
    scores = pdf_embeddings_matrix @ q_emb

    ranked = sorted(candidate_indices, key=lambda i: -scores[i])[:top_k]

    results = []
    seen = set()
    for i in ranked:
        chunk = pdf_chunks[i]
        key = chunk["text"][:80]
        if key not in seen:
            seen.add(key)
            results.append({
                "brand": chunk["brand"],
                "file": chunk["file"],
                "page": chunk.get("page"),
                "text": chunk["text"],
                "score": float(scores[i]),
            })
    return results, brand

def fmt_price(value) -> str:
    return f"{float(value):.3f}"

def format_category(path: str) -> str:
    return path.replace("/", " > ")

def format_context_block(p: dict) -> str:
    brand = p.get("brand") or "Unknown brand"
    lines = [f"[{brand.upper()}] {p['name']} (SKU: {p['sku']})"]

    if p.get("categories"):
        lines.append("Category: " + "; ".join(format_category(c) for c in p["categories"]))

    price_line = f"Price: {fmt_price(p['price'])} {p['currency']}"
    special = p.get("special_price")
    if special and special > 0 and special < p["price"]:
        price_line += f" (special offer: {fmt_price(special)} {p['currency']})"
    lines.append(price_line)

    status = p.get("status", {})
    if status.get("coming_soon"):
        availability = "Coming soon (not yet available in the showroom)"
    elif status.get("in_stock") and status.get("qty", 0) > 0:
        availability = f"In stock (qty: {int(status['qty'])})"
    else:
        availability = "Out of stock"
    lines.append("Availability: " + availability)

    specs = p.get("specs") or {}
    spec_bits = [f"{k}: {v}" for k, v in specs.items() if v and k != "brand"]
    if spec_bits:
        lines.append("Specs: " + "; ".join(spec_bits))

    if p.get("description"):
        lines.append("Description: " + p["description"])

    return "\n".join(lines)

def format_pdf_chunk(chunk: dict) -> str:
    header = f"[{chunk['brand']}] Catalogue excerpt ({chunk['file']}, p.{chunk.get('page', '?')}) — no live price/stock data"
    return f"{header}\n{chunk['text']}"

# Characters that only show up in other Latin-script languages, never in normal
# English or Arabic prose — used to catch the model slipping into another language.
NON_TARGET_CHARS = re.compile(r"[äöüßàèìòùâêîôûçñ]", re.IGNORECASE)
# qwen2.5 occasionally falls back to a completely different script when confused —
# caught it leaking Chinese once and Cyrillic another time, so rather than chase scripts
# one at a time, block every script we don't support at all (we only support
# English/Arabic): CJK, Hangul, Cyrillic, Hebrew, Devanagari, Thai.
FOREIGN_SCRIPT_CHARS = re.compile(
    r"[一-鿿぀-ヿ가-힣"  # CJK / Hangul
    r"Ѐ-ӿ"  # Cyrillic
    r"֐-׿"  # Hebrew
    r"ऀ-ॿ"  # Devanagari
    r"฀-๿]"  # Thai
)

def is_wrong_language(answer: str, answer_language: str) -> bool:
    if FOREIGN_SCRIPT_CHARS.search(answer):
        return True
    if len(NON_TARGET_CHARS.findall(answer)) >= 3:
        return True
    if answer_language == "Arabic":
        arabic_chars = sum(1 for ch in answer if '؀' <= ch <= 'ۿ')
        letters = sum(1 for ch in answer if ch.isalpha())
        return letters > 10 and arabic_chars < letters * 0.3
    return False

def generate_answer(question: str, context: str, answer_language: str, history: list = [], retry: bool = False) -> str:
    if answer_language == "Arabic":
        # Models follow same-language instructions better. Pinned details: keep
        # product names/SKUs in Latin (transliterating them is unusable to
        # customers and staff at the till), and keep the "199.000 BHD" price
        # format (fix_price_formatting below is the belt-and-suspenders backup).
        rule = ("أجب باللغة العربية الفصحى فقط، بأسلوب سلس وواضح ولغة رسمية مهذبة. "
                "اترك أسماء المنتجات والعلامات التجارية وأرقام SKU كما هي بالأحرف اللاتينية، "
                "واكتب الأسعار بالأرقام اللاتينية مثل 199.000 BHD. لا تخلط جملاً إنجليزية في الإجابة.")
    else:
        rule = f"Respond ONLY in {answer_language}."
    if retry:
        wrong_language_notice = "إجابتك السابقة كانت بلغة خاطئة. " if answer_language == "Arabic" \
            else "Your previous answer was written in the wrong language. "
        rule = wrong_language_notice + rule

    system_prompt = f"""You are a helpful showroom assistant for Khalaifat Co, a home appliances and kitchen retailer in Bahrain. You answer customer questions using ONLY the information given with each question, which comes from one or both of:
1. Live product database entries (brand, SKU, exact price, stock, specs) for appliance brands.
2. "Catalogue excerpt" entries — text pulled from PDF brochures for Nolte, Express, Cosentino (Silestone/Dekton/Sensa) and Poggenpohl. These describe designs, materials, colours and finishes but carry NO live price or stock data. That catalogue text is often in German, Italian or French, INCLUDING compound technical/marketing words that don't look obviously foreign because they contain no accents (e.g. German "landhausstil", "Manganbronze", "acrylglas", "edel Holztöne" are NOT English or Arabic words). Only proper names of product lines/collections (e.g. "RAVENNA LACK", "neoFLAT") should stay as-is — every other noun, adjective and material name must be translated into plain {answer_language}. Example: German "Front in Manganbronze mit edel Holztöne" must become English "front panel in manganese bronze with an elegant wood-tone finish" — NOT left in German. If you are not fully sure of a translation, describe it in simple {answer_language} words rather than copying the foreign term. If asked about price or stock for one of these four brands, say that live pricing/availability isn't in the system and the customer should ask in-store.

IMPORTANT RULE: {rule}

Be concise and helpful. Use the exact price, availability and spec values given — never guess, round, or invent them. Copy every price EXACTLY character-for-character, e.g. "199.000 BHD" — it uses a decimal point with three digits (thousandths of a dinar), never a comma, and is never a thousands-grouped number. Do not reformat, round, or re-punctuate it. The "[BRAND] ..." tag at the start of each entry is internal formatting for you only — never copy those brackets into your answer; refer to products and brands in plain natural sentences instead. Plain text only — no markdown, no asterisks, no underscores for emphasis, no headings. If the information given doesn't actually answer the question, say so honestly instead of guessing or inventing a product."""

    user_content = f"Product Information:\n{context}\n\nCustomer Question: {question}\n\nReminder: {rule}"

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": user_content})

    completion = llm_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return (completion.choices[0].message.content or "").strip()

# Belt-and-suspenders: the "no markdown" instruction above isn't 100% reliable, and a
# customer seeing literal "_ovens_" or "**Bosch**" in a chat bubble reads as broken —
# strip markdown emphasis/heading/code markers regardless of whether the model listened.
MARKDOWN_MARKERS = re.compile(r"(\*\*|\*|__|_|`|^#+\s*)", re.MULTILINE)

def strip_markdown(text: str) -> str:
    return MARKDOWN_MARKERS.sub("", text)

# Belt-and-suspenders: smaller local models occasionally localize "199.000 BHD" into
# "199,000 BHD" (comma decimal separator, common in many locales) — a 1000x price error.
# Every real price in this catalogue is under 1000 BHD, so any comma-grouped "N,NNN BHD"
# the model emits is a mis-rendering, never a genuine thousands amount — safe to fix.
MISFORMATTED_PRICE = re.compile(r"\b(\d{1,3}),(\d{3})\s*BHD\b")

def fix_price_formatting(text: str) -> str:
    return MISFORMATTED_PRICE.sub(r"\1.\2 BHD", text)

# Belt-and-suspenders: the prompt tells the model not to copy the internal
# "[BRAND]" context tag into its answer, but on longer multi-brand answers it
# occasionally does anyway (e.g. "[BEKO] BEKO Washer Dryer..."). Strip any
# all-caps bracket tag a customer shouldn't see, plus the trailing space it leaves.
BRACKET_TAG = re.compile(r"\[[A-Z][A-Z0-9 &-]{1,30}\]\s*")

def strip_bracket_tags(text: str) -> str:
    return BRACKET_TAG.sub("", text)

@app.post("/ask")
async def ask(q: Question):
    retrieval_q = contextualize(q.question, q.history)
    product_results, _ = find_relevant_products(retrieval_q)
    pdf_results, _ = find_relevant_pdf_chunks(retrieval_q)

    if not product_results and not pdf_results:
        return {"answer": "لم أجد معلومات كافية عن المنتجات للإجابة على هذا السؤال. / I couldn't find enough product information to answer this question.", "sources": []}

    context_parts = []
    if product_results:
        context_parts.append("\n\n".join(format_context_block(r["product"]) for r in product_results))
    if pdf_results:
        context_parts.append("\n\n".join(format_pdf_chunk(r) for r in pdf_results))
    context = "\n\n".join(context_parts)

    sources = list({r["product"]["brand"] for r in product_results if r["product"].get("brand")}
                    | {r["brand"].title() for r in pdf_results})

    is_arabic = any('؀' <= ch <= 'ۿ' for ch in q.question)
    answer_language = "Arabic" if is_arabic else "English"

    try:
        answer = generate_answer(q.question, context, answer_language, q.history)
        # The model occasionally slips into another language or script instead of the
        # customer's. Catch that and force a corrective rewrite rather than silently
        # returning the wrong language. Two retries: a single retry occasionally lands
        # on another bad sample too (seen once with a mixed English/Arabic question
        # producing Chinese twice).
        for _ in range(2):
            if not is_wrong_language(answer, answer_language):
                break
            answer = generate_answer(q.question, context, answer_language, q.history, retry=True)
        answer = strip_markdown(answer)
        answer = fix_price_formatting(answer)
        answer = strip_bracket_tags(answer)
    except Exception as e:
        answer = f"Error connecting to AI model: {e}"

    return {"answer": answer, "sources": sources}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "products": len(products),
        "pdf_chunks": len(pdf_chunks),
        "pdf_brands": PDF_BRANDS,
    }

@app.get("/", response_class=HTMLResponse)
async def site():
    with open(f"{SITE_ROOT}/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    with open("ask.html", "r", encoding="utf-8") as f:
        return f.read()

# One port for everything: the catalogue site's own asset/PDF folders, mounted
# under the same paths index.html already references ("assets/...", "pdfs/...").
# Previously the site (port 8080) and the AI (port 8000) were two separate
# servers — customers/the tablet only need one URL and one port now.
app.mount("/assets", StaticFiles(directory=f"{SITE_ROOT}/assets"), name="assets")
app.mount("/pdfs", StaticFiles(directory=f"{SITE_ROOT}/pdfs"), name="pdfs")
app.mount("/css", StaticFiles(directory=f"{SITE_ROOT}/css"), name="css")
