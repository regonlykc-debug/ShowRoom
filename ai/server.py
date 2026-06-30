import json
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

KB_FILE = "knowledge_base.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
TOP_K = 5

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading knowledge base...")
with open(KB_FILE, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

embeddings_matrix = np.array([c["embedding"] for c in knowledge_base], dtype=np.float32)
norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
embeddings_matrix = embeddings_matrix / np.clip(norms, 1e-10, None)

print("Loading embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Server ready!")

class Question(BaseModel):
    question: str
    language: str = "auto"

def find_relevant_chunks(question: str, top_k: int = TOP_K):
    q_emb = embed_model.encode([question])[0].astype(np.float32)
    q_emb = q_emb / np.clip(np.linalg.norm(q_emb), 1e-10, None)
    scores = embeddings_matrix @ q_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    seen = set()
    for i in top_indices:
        chunk = knowledge_base[i]
        key = chunk["text"][:80]
        if key not in seen:
            seen.add(key)
            results.append({"brand": chunk["brand"], "file": chunk["file"], "text": chunk["text"], "score": float(scores[i])})
    return results

@app.post("/ask")
async def ask(q: Question):
    chunks = find_relevant_chunks(q.question)

    if not chunks or chunks[0]["score"] < 0.2:
        return {"answer": "لم أجد معلومات كافية في الكتالوجات للإجابة على هذا السؤال. / I couldn't find enough information in the catalogues to answer this question.", "sources": []}

    context = "\n\n".join([f"[{c['brand']} - {c['file']}]\n{c['text']}" for c in chunks])
    sources = list({c["brand"] for c in chunks})

    prompt = f"""You are a helpful showroom assistant for Khalaifat Co. You answer questions about products based only on the catalogue information provided below.

Answer in the same language the customer used. If they wrote in Arabic, answer in Arabic. If English, answer in English.
Be concise, helpful, and mention the brand name when relevant.

Catalogue Information:
{context}

Customer Question: {q.question}

Answer:"""

    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=60)
        answer = response.json().get("response", "").strip()
    except Exception as e:
        answer = f"Error connecting to AI model: {e}"

    return {"answer": answer, "sources": sources}

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "chunks": len(knowledge_base)}

@app.get("/", response_class=HTMLResponse)
async def ui():
    with open("ask.html", "r", encoding="utf-8") as f:
        return f.read()
