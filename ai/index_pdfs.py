import os
import json
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np

PDFS_DIR = "../pdfs"
OUTPUT_FILE = "knowledge_base.json"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

print("Loading embedding model (multilingual - supports Arabic/English)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def extract_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"  Warning: could not read {pdf_path}: {e}")
    return text.strip()

def chunk_text(text, source):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def get_brand_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "pdfs" and i + 1 < len(parts):
            return parts[i + 1].upper()
    return "UNKNOWN"

all_chunks = []

print(f"\nScanning PDFs in {PDFS_DIR}...")
for root, dirs, files in os.walk(PDFS_DIR):
    for fname in files:
        if fname.lower().endswith(".pdf"):
            path = os.path.join(root, fname)
            brand = get_brand_from_path(path)
            print(f"  Processing [{brand}] {fname}...")
            text = extract_text(path)
            if not text:
                print(f"    Skipped (no text extracted)")
                continue
            chunks = chunk_text(text, path)
            for chunk in chunks:
                all_chunks.append({
                    "brand": brand,
                    "file": fname,
                    "text": chunk
                })
            print(f"    {len(chunks)} chunks created")

print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

for i, chunk in enumerate(all_chunks):
    chunk["embedding"] = embeddings[i].tolist()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False)

print(f"\nDone! Knowledge base saved to {OUTPUT_FILE}")
print(f"Total chunks: {len(all_chunks)}")
print("\nNow run: 3_start_server.bat")
