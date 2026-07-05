import json

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PRODUCTS_FILE = "products_clean.json"
OUTPUT_FILE = "product_kb.json"

# MUST match server.py.
EMBED_MODEL = "intfloat/multilingual-e5-base"

# e5 requires every passage to start with "passage: " (and every query with
# "query: " on the server side). Skipping this silently degrades retrieval.
PASSAGE_PREFIX = "passage: "


def format_category(path):
    return path.replace("/", " > ")


def build_search_text(p):
    """One short paragraph per product, combining every field a customer
    might ask about, so a single embedding captures name + brand + category +
    specs + description. This is what gets matched against the customer's
    question."""
    parts = [p["name"]]
    if p.get("brand"):
        parts.append(f"Brand: {p['brand']}")
    if p.get("categories"):
        parts.append("Category: " + "; ".join(format_category(c) for c in p["categories"]))
    specs = p.get("specs") or {}
    # "brand" is duplicated into specs by the export; skip it, it's already above.
    spec_bits = [f"{k}: {v}" for k, v in specs.items() if v and k != "brand"]
    if spec_bits:
        parts.append("Specs: " + "; ".join(spec_bits))
    if p.get("description"):
        parts.append(p["description"])
    return ". ".join(str(x) for x in parts if x)


print(f"Loading {PRODUCTS_FILE}...")
with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
    raw_products = json.load(f)

# Test/placeholder catalogue entries (e.g. "Magneto Test Product") should
# never show up as a real answer to a customer.
products = [p for p in raw_products if not p.get("status", {}).get("is_test")]
print(f"{len(products)} real products ({len(raw_products) - len(products)} test entries excluded)")

for p in products:
    p["search_text"] = build_search_text(p)

print(f"\nLoading embedding model: {EMBED_MODEL}")
print("(first run downloads it — one-time)")
model = SentenceTransformer(EMBED_MODEL, device="cpu")

print(f"\nGenerating embeddings for {len(products)} products...")
texts = [PASSAGE_PREFIX + p["search_text"] for p in products]
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=16,
    normalize_embeddings=True,  # store unit vectors so the server can skip re-normalizing
    device="cpu",
)

for i, p in enumerate(products):
    p["embedding"] = embeddings[i].tolist()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False)

print(f"\nDone! Product knowledge base saved to {OUTPUT_FILE}")
print(f"Total products: {len(products)}")
print(f"Embedding model: {EMBED_MODEL}")
print("\nNow run: 3_start_server.sh / 3_start_server.bat (server.py must use the SAME model)")
