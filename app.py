# app.py  — Chinese + English Contract AI Assistant (resilient boot)
# Boots even if ML libs (faiss / torch / transformers) are missing.
# Static files + /health always work. ML endpoints are guarded with clear messages.

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid
import json

# =========================
# Safe / optional imports
# =========================
# These try/except blocks make the server START even if you don't have the heavy libs installed yet.
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # guarded at use time

try:
    import faiss  # or 'faiss_cpu' if you use that build
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# We lazy-import transformers.PIPELINE models only when needed
SUM_EN = None
SUM_ZH = None

# =========================
# App setup
# =========================
BASE_DIR: Path = Path(__file__).parent.resolve()
STATIC_DIR: Path = BASE_DIR / "static"
STORE: Path = BASE_DIR / "storage"
(STORE / "indexes").mkdir(parents=True, exist_ok=True)
(STORE / "texts").mkdir(parents=True, exist_ok=True)
LAST_DOC_FILE: Path = STORE / "last_doc_id.txt"

APP = FastAPI(title="Chinese + English Contract AI Assistant")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve /static/*
APP.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@APP.get("/")
def root():
    """
    Redirect root → demo page. Change 'testing.html' to your actual filename if needed.
    """
    return RedirectResponse(url="/static/testing.html")

@APP.get("/health")
def health():
    return {"status": "ok"}

print(f"[BOOT] static dir: {STATIC_DIR}")
print(f"[BOOT] store dir : {STORE}")

# =========================
# Helpers: language heuristics
# =========================
def has_cjk_ratio(text: str) -> float:
    total = 0
    cjk = 0
    for ch in text or "":
        if ch.isspace():
            continue
        total += 1
        if "\u4e00" <= ch <= "\u9fff":  # basic CJK range
            cjk += 1
    return (cjk / total) if total else 0.0

def choose_embed_lang(sample: str) -> str:
    # Lightweight heuristic first
    if has_cjk_ratio(sample) >= 0.02:
        return "zh"
    # Fallback to langdetect if available
    try:
        from langdetect import detect  # optional dep
        lang = detect((sample or "")[:200])
        return "zh" if str(lang).startswith("zh") else "en"
    except Exception:
        return "en"

def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    chunks: List[str] = []
    i, n = 0, len(text or "")
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

# =========================
# Embeddings (lazy init)
# =========================
EMB_EN = None
EMB_MULTI = None

def get_embedder_model(lang: str):
    """
    Lazy-init SentenceTransformer only when needed.
    Raises clean errors if not available (e.g., torch not installed).
    """
    global EMB_EN, EMB_MULTI

    if SentenceTransformer is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Embeddings unavailable: 'sentence-transformers' (and Torch) not installed.\n"
                "Fix: python -m pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu"
            ),
        )

    if lang == "zh":
        if EMB_MULTI is None:
            EMB_MULTI = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return EMB_MULTI
    else:
        if EMB_EN is None:
            EMB_EN = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return EMB_EN

def encode_texts(texts: List[str], lang: str):
    model = get_embedder_model(lang)  # may raise if SentenceTransformer/Torch missing
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # If FAISS exists, we normalize to use inner product as cosine
    if faiss is not None:
        faiss.normalize_L2(vecs)
    return vecs

# =========================
# Summarizers (lazy load)
# =========================
def get_sum_en():
    global SUM_EN
    if SUM_EN is None:
        try:
            from transformers import pipeline  # heavy; optional
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=(
                    "English summarizer unavailable: 'transformers' not installed.\n"
                    "Fix: python -m pip install transformers"
                ),
            )
        SUM_EN = pipeline("summarization", model="facebook/bart-large-cnn")
    return SUM_EN

def get_sum_zh():
    global SUM_ZH
    if SUM_ZH is None:
        try:
            from transformers import pipeline  # heavy; optional
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Chinese summarizer unavailable: 'transformers' not installed.\n"
                    "Fix: python -m pip install transformers"
                ),
            )
        # Small Chinese summarizer (adjust if you prefer another)
        SUM_ZH = pipeline("summarization", model="IDEA-CCNL/Randeng-BART-139M-Chinese")
    return SUM_ZH

def summarize_text(text: str, lang: str) -> str:
    """
    Best-effort summarize. If summarizers are missing, return a trimmed snippet.
    """
    snippet = (text or "").strip().replace("\n", " ")
    if not snippet:
        return ""
    try:
        if lang == "zh":
            return get_sum_zh()(snippet, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
        else:
            return get_sum_en()(snippet, max_length=160, min_length=60, do_sample=False)[0]["summary_text"]
    except Exception:
        return snippet[:180] + ("…" if len(snippet) > 180 else "")

# =========================
# API: Upload → parse → index (guarded)
# =========================
@APP.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    if PdfReader is None:
        raise HTTPException(
            status_code=500,
            detail="PDF parsing unavailable: install pypdf (python -m pip install pypdf)",
        )
    if faiss is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Vector indexing unavailable: FAISS not installed on this system.\n"
                "Options:\n"
                "  • Install faiss-cpu (Linux/macOS easier)\n"
                "  • On Windows, consider using Annoy (pip install annoy) or scikit-learn NearestNeighbors\n"
                "  • Or move vector search to an external DB (Qdrant/Pinecone)"
            ),
        )

    # Save temp PDF
    tmp_pdf = STORE / f"tmp_{uuid.uuid4()}.pdf"
    tmp_pdf.write_bytes(await file.read())

    # Extract text
    reader = PdfReader(str(tmp_pdf))
    pages = [p.extract_text() or "" for p in reader.pages]
    doc_text = "\n".join([f"[Page {i+1}]\n{t}" for i, t in enumerate(pages)])

    # Detect doc language; embed with matching model
    doc_lang = choose_embed_lang(doc_text)
    chunks = chunk_text(doc_text)
    vecs = encode_texts(chunks, lang=doc_lang)  # may raise if sentence-transformers missing
    dim = int(vecs.shape[1])

    # Build FAISS index (inner product == cosine after L2 norm)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    nvec = int(vecs.shape[0])

    # Persist
    doc_id = str(uuid.uuid4())
    faiss.write_index(index, str(STORE / "indexes" / f"{doc_id}.faiss"))
    (STORE / "texts" / f"{doc_id}.json").write_text(
        json.dumps({"chunks": chunks, "lang": doc_lang}, ensure_ascii=False),
        encoding="utf-8",
    )
    LAST_DOC_FILE.write_text(doc_id, encoding="utf-8")
    try:
        tmp_pdf.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "vectors": nvec,
        "lang": doc_lang,
    }

# =========================
# API: Ask (guarded)
# =========================
class AskIn(BaseModel):
    doc_id: Optional[str] = None
    question: str
    top_k: int = 4

@APP.post("/ask")
def ask(inp: AskIn) -> Dict[str, Any]:
    if faiss is None:
        raise HTTPException(
            status_code=500,
            detail="Vector search unavailable: FAISS not installed.",
        )

    # Use last uploaded doc if none provided
    doc_id = inp.doc_id or (LAST_DOC_FILE.read_text(encoding="utf-8").strip() if LAST_DOC_FILE.exists() else None)
    if not doc_id:
        raise HTTPException(status_code=400, detail="No doc_id provided and no last_doc_id saved.")

    meta_path = STORE / "texts" / f"{doc_id}.json"
    idx_path = STORE / "indexes" / f"{doc_id}.faiss"
    if not meta_path.exists() or not idx_path.exists():
        raise HTTPException(status_code=404, detail=f"doc_id {doc_id} not found (missing index and/or metadata).")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks: List[str] = meta["chunks"]
    doc_lang = meta.get("lang", "en")
    index = faiss.read_index(str(idx_path))

    # Encode the query in the same language model used for the doc
    q_vecs = encode_texts([inp.question], lang=doc_lang)
    D, I = index.search(q_vecs, int(inp.top_k))

    hits = []
    for rank, (score, idx) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1):
        if idx == -1:
            continue
        text = chunks[idx]
        page_hint = None
        if text.startswith("[Page "):
            end = text.find("]\n")
            if end != -1:
                page_hint = text[1:end]  # e.g., "Page 3"
        hits.append(
            {
                "rank": rank,
                "score": round(float(score), 4),
                "page_hint": page_hint,
                "excerpt": text[:400],
            }
        )

    joined = "\n\n".join(h["excerpt"] for h in hits) if hits else "No relevant excerpts found."
    summary = summarize_text(joined, lang=doc_lang)

    return {
        "question": inp.question,
        "doc_id_used": doc_id,
        "language": doc_lang,
        "answer": summary,
        "citations": hits,
    }

# =========================
# API: Utilities
# =========================
@APP.get("/last_doc")
def get_last_doc() -> Dict[str, Optional[str]]:
    return {"last_doc_id": LAST_DOC_FILE.read_text(encoding="utf-8").strip()} if LAST_DOC_FILE.exists() else {"last_doc_id": None}

@APP.get("/deps")
def deps_status() -> Dict[str, Any]:
    return {
        "static_dir_exists": STATIC_DIR.exists(),
        "store_dir": str(STORE),
        "embeddings": {
            "impl": "sentence-transformers (lazy)",
            "en_model": "sentence-transformers/all-MiniLM-L6-v2",
            "multi_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "available": SentenceTransformer is not None,
        },
        "faiss_available": faiss is not None,
        "summarizers_loaded": {"en": SUM_EN is not None, "zh": SUM_ZH is not None},
    }

# Optional: local run helper (not used by Render)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:APP", host="0.0.0.0", port=8000, reload=True)
