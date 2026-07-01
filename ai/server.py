import difflib
import json
import re
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
BRAND_MATCH_CUTOFF = 0.75  # high enough to avoid "electric"->ELICA, "expensive"->EXPRESS false positives

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading knowledge base...")
with open(KB_FILE, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

embeddings_matrix = np.array([c["embedding"] for c in knowledge_base], dtype=np.float32)
norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
embeddings_matrix = embeddings_matrix / np.clip(norms, 1e-10, None)

BRANDS = sorted({c["brand"] for c in knowledge_base})
BRAND_INDICES = {b: [i for i, c in enumerate(knowledge_base) if c["brand"] == b] for b in BRANDS}

# What each brand actually sells — injected into the prompt so the model can't imply
# e.g. Nolte (cabinetry) sells ovens just because an oven question retrieved a Nolte chunk.
# Kept in both languages: handing the model an English scope line inside an Arabic
# answer made it copy the English words verbatim (sometimes even with markdown emphasis
# markers), so we give it the scope already in the language it needs to answer in.
BRAND_SCOPE = {
    "BEKO": "home appliances only (ovens, fridges, dishwashers, washing machines)",
    "BOSCH": "home appliances only (ovens, dishwashers, cooking ranges, fridges)",
    "ELBA": "built-in cooking appliances only (ovens, hobs, hoods)",
    "ELICA": "kitchen hoods / ventilation only — does NOT sell ovens, fridges or other appliances",
    "NOLTE": "kitchen cabinetry and furniture only — does NOT sell appliances like ovens or fridges",
    "EXPRESS": "kitchen cabinetry and furniture only (Nolte Group) — does NOT sell appliances",
    "COSENTINO": "countertop/surface materials only (Silestone, Dekton, Sensa) — does NOT sell appliances",
    "POGGENPOHL": "luxury kitchen cabinetry and furniture only — does NOT sell appliances",
}
BRAND_SCOPE_AR = {
    "BEKO": "أجهزة منزلية فقط (أفران، ثلاجات، غسالات صحون، غسالات ملابس)",
    "BOSCH": "أجهزة منزلية فقط (أفران، غسالات صحون، مواقد طهي، ثلاجات)",
    "ELBA": "أجهزة طهي مدمجة فقط (أفران، مواقد، شفاطات)",
    "ELICA": "شفاطات ووسائل تهوية المطبخ فقط — لا تبيع أفران أو ثلاجات أو أجهزة أخرى",
    "NOLTE": "خزائن وأثاث مطبخ فقط — لا تبيع أجهزة مثل الأفران أو الثلاجات",
    "EXPRESS": "خزائن وأثاث مطبخ فقط (مجموعة نولته) — لا تبيع أجهزة",
    "COSENTINO": "مواد أسطح فقط (سيليستون، ديكتون، سينسا) — لا تبيع أجهزة",
    "POGGENPOHL": "خزائن وأثاث مطبخ فاخر فقط — لا تبيع أجهزة",
}

# Telling the model "Elica doesn't sell ovens" in the prompt isn't reliable enough at
# this model size — it still asserted Elica sold ovens after retrieving an Elica chunk
# that happened to mention an oven-adjacent word. So for appliance questions, exclude
# non-appliance brands from retrieval entirely rather than trusting the model to ignore
# their content.
APPLIANCE_BRANDS = {"BEKO", "BOSCH", "ELBA"}
APPLIANCE_KEYWORDS = re.compile(
    r"oven|fridge|refrigerator|freezer|dishwasher|washing\s*machine|dryer|cooker|stove|hob|"
    r"فرن|ثلاج|فريزر|غسال|مجفف|طباخ|بوتاجاز",
    re.IGNORECASE,
)
APPLIANCE_KEYWORD_LIST = ["OVEN", "FRIDGE", "REFRIGERATOR", "FREEZER", "DISHWASHER", "DRYER", "COOKER", "STOVE", "HOB"]

def is_appliance_question(question: str) -> bool:
    if APPLIANCE_KEYWORDS.search(question):
        return True
    # tolerate typos like "diswaser" the same way brand names are typo-corrected —
    # otherwise a misspelled appliance word falls through to the full, ELICA-dominated
    # search instead of being restricted to the brands that actually sell appliances.
    for word in re.findall(r"[A-Za-z]+", question.upper()):
        if len(word) >= 5 and difflib.get_close_matches(word, APPLIANCE_KEYWORD_LIST, n=1, cutoff=0.75):
            return True
    return False

print("Loading embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Server ready!")

class Question(BaseModel):
    question: str
    language: str = "auto"

# Arabic transliterations of brand names — the Latin-script typo correction below can't
# see these at all, which was silently sending Arabic brand questions to a full, ELICA-
# dominated search instead of the named brand's own content.
ARABIC_BRAND_ALIASES = {
    "بوش": "BOSCH", "بوشية": "BOSCH",
    "بيكو": "BEKO",
    "نولته": "NOLTE", "نولت": "NOLTE",
    "اكسبرس": "EXPRESS", "إكسبرس": "EXPRESS",
    "الاليكا": "ELICA", "إليكا": "ELICA", "اليكا": "ELICA",
    "كوزنتينو": "COSENTINO", "كوسنتينو": "COSENTINO",
    "بوجنبول": "POGGENPOHL", "بوغنبول": "POGGENPOHL",
    "البا": "ELBA", "إلبا": "ELBA",
}

def correct_brand_typos(question: str):
    """Replace a misspelled brand name with its correct spelling (e.g. 'boash' -> 'BOSCH'),
    so a lone typo'd brand name still embeds close to that brand's content, and return
    which brand (if any) was mentioned."""
    detected_brand = []  # list so the closure below can assign into it

    def repl(m):
        word = m.group(0)
        if len(word) < 3:
            return word
        match = difflib.get_close_matches(word.upper(), BRANDS, n=1, cutoff=BRAND_MATCH_CUTOFF)
        if match:
            detected_brand.append(match[0])
            return match[0]
        return word

    corrected = re.sub(r"[A-Za-z]+", repl, question)

    for word in re.findall(r"[؀-ۿ]+", question):
        if word in ARABIC_BRAND_ALIASES:
            detected_brand.append(ARABIC_BRAND_ALIASES[word])

    return corrected, (detected_brand[0] if detected_brand else None)

def find_relevant_chunks(question: str, top_k: int = TOP_K):
    # A named brand (even misspelled) should only pull chunks from that brand —
    # otherwise a global search gets drowned out by whichever brand has the most
    # chunks (ELICA has 262 of 477, over half the knowledge base).
    corrected_question, brand = correct_brand_typos(question)
    if brand:
        candidate_indices = BRAND_INDICES[brand]
    elif is_appliance_question(question):
        candidate_indices = [i for b in APPLIANCE_BRANDS for i in BRAND_INDICES[b]]
    else:
        candidate_indices = range(len(knowledge_base))

    q_emb = embed_model.encode([corrected_question])[0].astype(np.float32)
    q_emb = q_emb / np.clip(np.linalg.norm(q_emb), 1e-10, None)
    scores = embeddings_matrix @ q_emb
    ranked = sorted(candidate_indices, key=lambda i: -scores[i])[:top_k]

    results = []
    seen = set()
    for i in ranked:
        chunk = knowledge_base[i]
        key = chunk["text"][:80]
        if key not in seen:
            seen.add(key)
            results.append({"brand": chunk["brand"], "file": chunk["file"], "text": chunk["text"], "score": float(scores[i])})
    return results, brand

# Characters that only show up in the catalogues' source languages (German/Italian/
# French), never in normal English or Arabic prose — used to catch the model slipping
# into the source language instead of the customer's.
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

def generate_answer(question: str, context: str, answer_language: str, sources: list, retry: bool = False) -> str:
    rule = (
        f"Respond ONLY in {answer_language}. The catalogue information below may be written in "
        f"German, Italian, French or another language — translate anything you use from it. "
        f"Never write any German, Italian or French words in your answer, only {answer_language}."
    )
    if retry:
        rule = "Your previous answer was written in the wrong language. " + rule

    scope_dict = BRAND_SCOPE_AR if answer_language == "Arabic" else BRAND_SCOPE
    scope_lines = "\n".join(f"- {b}: {scope_dict[b]}" for b in sources if b in scope_dict)
    scope_block = (
        f"\nWhat each brand actually sells (do not contradict this even if the catalogue "
        f"text below suggests otherwise):\n{scope_lines}\n"
        if scope_lines else ""
    )

    prompt = f"""You are a helpful showroom assistant for Khalaifat Co. You answer questions about products based only on the catalogue information provided below.

IMPORTANT RULE: {rule}
{scope_block}
Be concise and helpful. Plain text only — no markdown, no asterisks, no underscores for emphasis. If the catalogue information below doesn't actually answer the question, say so honestly instead of guessing — do not invent or assume products a brand might sell.

Catalogue Information:
{context}

Customer Question: {question}

Reminder: {rule}

Answer (in {answer_language} only, plain text, no markdown):"""

    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=60)
    return response.json().get("response", "").strip()

# Belt-and-suspenders: the "no markdown" instruction above isn't 100% reliable, and a
# customer seeing literal "_ovens_" or "**Bosch**" in a chat bubble reads as broken —
# strip markdown emphasis/heading/code markers regardless of whether the model listened.
MARKDOWN_MARKERS = re.compile(r"(\*\*|\*|__|_|`|^#+\s*)", re.MULTILINE)

def strip_markdown(text: str) -> str:
    return MARKDOWN_MARKERS.sub("", text)

@app.post("/ask")
async def ask(q: Question):
    chunks, brand_named = find_relevant_chunks(q.question)

    # A bare brand name ("bosch") scores low against prose catalogue text purely
    # because a single word never embeds close to a paragraph — but we already know
    # exactly what brand it's about, so don't reject it for a weak similarity score.
    min_score = 0.05 if brand_named else 0.2
    if not chunks or chunks[0]["score"] < min_score:
        return {"answer": "لم أجد معلومات كافية في الكتالوجات للإجابة على هذا السؤال. / I couldn't find enough information in the catalogues to answer this question.", "sources": []}

    context = "\n\n".join([f"[{c['brand']} - {c['file']}]\n{c['text']}" for c in chunks])
    sources = list({c["brand"] for c in chunks})

    is_arabic = any('؀' <= ch <= 'ۿ' for ch in q.question)
    answer_language = "Arabic" if is_arabic else "English"

    try:
        answer = generate_answer(q.question, context, answer_language, sources)
        # The catalogues themselves are German/Italian/French — the model sometimes
        # slips into the source language instead of the customer's. Catch that and
        # force one corrective rewrite rather than silently returning the wrong language.
        if is_wrong_language(answer, answer_language):
            answer = generate_answer(q.question, context, answer_language, sources, retry=True)
        answer = strip_markdown(answer)
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
