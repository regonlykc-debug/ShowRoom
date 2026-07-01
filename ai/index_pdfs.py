import gc
import os
import re
import json
import warnings

from sentence_transformers import SentenceTransformer
import pdfplumber

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDFS_DIR = "../pdfs"
OUTPUT_FILE = "knowledge_base.json"

# multilingual-e5-base gives much stronger Arabic/English -> Italian/German
# cross-lingual retrieval than the old MiniLM model, at a size this machine's
# 7.8GB RAM (no GPU) can handle safely. e5-large is stronger still but was too
# risky to load alongside PDF extraction on this hardware — if you move this
# to a bigger machine, e5-large is worth revisiting. The server MUST use
# whatever model is set here.
EMBED_MODEL = "intfloat/multilingual-e5-base"

# e5 requires every passage to start with "passage: " (and every query with
# "query: " on the server side). Skipping this silently degrades retrieval.
PASSAGE_PREFIX = "passage: "

# Chunk by characters on sentence/line boundaries rather than a blind word count,
# so a product's specs don't get sliced across two chunks.
CHUNK_SIZE = 1100        # ~180-220 words
CHUNK_OVERLAP = 250      # carry context across the boundary
EMBED_BATCH = 16


# ---------------------------------------------------------------------------
# Extraction (PDF text/tables only — no embedding model loaded during this
# phase, so its memory never competes with pdfplumber's per-page analysis).
# ---------------------------------------------------------------------------
def extract_page(page):
    """Return (prose_text, table_text) for one page.

    Tables are extracted on every page, not just sparse ones — spec tables
    (dimensions, capacity, energy class) are exactly what customers ask about
    and were being dropped whenever a page also had a paragraph of prose.
    They're kept separate and labelled [SPECS] so a later chunk boundary can't
    split a spec row in half.
    """
    prose = ""
    try:
        t = page.extract_text()
        if t:
            prose = t.strip()
    except Exception:
        pass

    table_text = ""
    try:
        for table in page.extract_tables():
            rows = []
            for row in table:
                cells = [c.strip() for c in row if c and c.strip()]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                table_text += "[SPECS] " + "  //  ".join(rows) + "\n"
    except Exception:
        pass

    return prose, table_text


def extract_document(pdf_path):
    """Extract the whole PDF as a list of (page_number, text) so chunking can
    stay aware of page boundaries and never merge two unrelated products
    across a page."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                if i % 25 == 0:
                    print(f"    page {i + 1}/{total}...")
                prose, tables = extract_page(page)
                combined = "\n".join(p for p in (prose, tables) if p).strip()
                if combined:
                    pages.append((i + 1, combined))
                # Release pdfplumber's per-page cached layout objects (chars,
                # lines, rects) immediately — letting these accumulate across
                # a ~250-page catalogue is what OOM-killed an earlier attempt
                # at extracting tables on every page.
                page.flush_cache()
    except Exception as e:
        print(f"  Warning: could not read {pdf_path}: {e}")
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n+")


def split_units(text):
    """Break text into small units (sentences / lines) we can pack into chunks
    without cutting through the middle of one."""
    units = []
    for part in _SENT_SPLIT.split(text):
        part = part.strip()
        if part:
            units.append(part)
    return units


def chunk_pages(pages):
    """Pack page text into ~CHUNK_SIZE-char chunks on unit boundaries, with
    overlap, resetting at every page so products from different pages never
    merge. Each chunk records the page it came from."""
    chunks = []
    for page_no, text in pages:
        units = split_units(text)
        buf = ""
        for unit in units:
            if len(buf) + len(unit) + 1 > CHUNK_SIZE and len(buf) > 50:
                chunks.append((page_no, buf.strip()))
                # start next chunk with a tail of the previous one for continuity
                buf = (buf[-CHUNK_OVERLAP:] + " " + unit).strip()
            else:
                buf = (buf + " " + unit).strip()
        if len(buf.strip()) > 50:
            chunks.append((page_no, buf.strip()))
    return chunks


def get_brand_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "pdfs" and i + 1 < len(parts):
            return parts[i + 1].upper()
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Phase 1: extract every PDF into raw text chunks. No embedding model loaded
# yet — keeps peak memory during the heaviest part (full-page table
# extraction across ~900 pages) as low as possible.
# ---------------------------------------------------------------------------
all_chunks = []

print(f"Scanning PDFs in {PDFS_DIR}...")
for root, dirs, files in os.walk(PDFS_DIR):
    for fname in sorted(files):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(root, fname)
        brand = get_brand_from_path(path)
        print(f"  Processing [{brand}] {fname}...")
        pages = extract_document(path)
        if not pages:
            print("    Skipped (no text extracted)")
            continue
        chunks = chunk_pages(pages)
        for page_no, chunk in chunks:
            all_chunks.append({
                "brand": brand,
                "file": fname,
                "page": page_no,
                "text": chunk,
            })
        print(f"    {len(chunks)} chunks from {len(pages)} pages")
        gc.collect()

# ---------------------------------------------------------------------------
# Phase 2: embeddings. All pdfplumber objects are out of scope by now — only
# the embedding model needs to fit in memory for this phase.
# ---------------------------------------------------------------------------
print(f"\nLoading embedding model: {EMBED_MODEL}")
print("(first run downloads it — one-time)")
model = SentenceTransformer(EMBED_MODEL, device="cpu")

print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
texts = [PASSAGE_PREFIX + c["text"] for c in all_chunks]
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=EMBED_BATCH,
    normalize_embeddings=True,   # store unit vectors so the server can skip re-normalizing
    device="cpu",
)

for i, chunk in enumerate(all_chunks):
    chunk["embedding"] = embeddings[i].tolist()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False)

print(f"\nDone! Knowledge base saved to {OUTPUT_FILE}")
print(f"Total chunks: {len(all_chunks)}")
print(f"Embedding model: {EMBED_MODEL}")
print("\nNow run: 3_start_server.sh / 3_start_server.bat (server.py must use the SAME model)")
