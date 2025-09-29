# app.py — Insurance RAG (FastAPI + FAISS + Transformers + fallback Annoy)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid
import json
import os

# =========================
# Safe / optional imports
# =========================
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from annoy import AnnoyIndex
except Exception:
    AnnoyIndex = None

# Summarizers (lazy loaded)
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

APP.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@APP.get("/")
def root():
    return RedirectResponse(url="/static/testing.html")

@APP.get("/health")
def health():
    return {"status": "ok"}

print(f"[BOOT] static dir: {STATIC_DIR}")
print(f"[BOOT] store dir : {STORE}")

# =========================
# Helpers
# =========================
def has_cjk_ratio(text: str) -> float:
    total, cjk = 0, 0
    for ch in text or "":
        if ch.isspace():
            continue
        total += 1
        if "\u4e00" <= ch <= "\u9fff":
            cjk += 1
    return (cjk / total) if total else 0.0

def choose_embed_lang(sample: str) -> str:
    if has_cjk_ratio(sample) >= 0.02:
        return "zh"
    try:
        from langdetect import detect
        lang = detect((sample or "")[:200])
        return "zh" if str(lang).startswith("zh") else "en"
    except Exception:
        return "en"

def chunk_text(text: str, max_chars=800, overlap=120) -> List[str]:
    chunks = []
    i, n = 0, len(text or "")
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

# =========================
# Embeddings
# =========================
EMB_EN = None
EMB_MULTI = None

def get_embedder_model(lang: str):
    global EMB_EN, EMB_MULTI
    if SentenceTransformer is None:
        raise HTTPException(500, "SentenceTransformer not installed (pip install sentence-transformers)")
    if lang == "zh":
        if EMB_MULTI is None:
            EMB_MULTI = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return EMB_MULTI
    else:
        if EMB_EN is None:
            EMB_EN = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return EMB_EN

def encode_texts(texts: List[str], lang: str):
    model = get_embedder_model(lang)
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if faiss is not None:
        faiss.normalize_L2(vecs)
    return vecs

# =========================
# Summarizers
# =========================
def get_sum_en():
    global SUM_EN
    if SUM_EN is None:
        from transformers import pipeline
        # lighter than bart-large-cnn → less RAM
        SUM_EN = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return SUM_EN

def get_sum_zh():
    global SUM_ZH
    if SUM_ZH is None:
        from transformers import pipeline
        SUM_ZH = pipeline("summarization", model="IDEA-CCNL/Randeng-BART-139M-Chinese")
    return SUM_ZH

def summarize_text(text: str, lang: str) -> str:
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
# Upload
# =========================
@APP.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    if PdfReader is None:
        raise HTTPException(500, "PDF parsing unavailable (pip install pypdf)")
    # Save temp
    tmp_pdf = STORE / f"tmp_{uuid.uuid4()}.pdf"
    tmp_pdf.write_bytes(await file.read())
    reader = PdfReader(str(tmp_pdf))
    pages = [p.extract_text() or "" for p in reader.pages]
    doc_text = "\n".join([f"[Page {i+1}]\n{t}" for i, t in enumerate(pages)])
    doc_lang = choose_embed_lang(doc_text)
    chunks = chunk_text(doc_text)
    vecs = encode_texts(chunks, doc_lang)
    dim, nvec = int(vecs.shape[1]), int(vecs.shape[0])
    doc_id = str(uuid.uuid4())
    idx_dir = STORE / "indexes"; idx_dir.mkdir(parents=True, exist_ok=True)
    index_kind = None
    if faiss is not None:
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        faiss.write_index(index, str(idx_dir / f"{doc_id}.faiss"))
        index_kind = "faiss"
    elif AnnoyIndex is not None:
        index = AnnoyIndex(dim, metric="angular")
        for i, v in enumerate(vecs):
            index.add_item(i, v.tolist())
        index.build(10)
        index.save(str(idx_dir / f"{doc_id}.ann"))
        index_kind = "annoy"
    else:
        raise HTTPException(500, "No vector index backend available (install faiss-cpu or annoy)")
    (STORE / "texts" / f"{doc_id}.json").write_text(
        json.dumps({"chunks": chunks, "lang": doc_lang, "index_kind": index_kind}, ensure_ascii=False),
        encoding="utf-8",
    )
    LAST_DOC_FILE.write_text(doc_id, encoding="utf-8")
    tmp_pdf.unlink(missing_ok=True)
    return {"doc_id": doc_id, "filename": file.filename, "pages": len(pages), "chunks": len(chunks), "vectors": nvec, "lang": doc_lang}

# =========================
# Ask
# =========================
class AskIn(BaseModel):
    doc_id: Optional[str] = None
    question: str
    top_k: int = 4

@APP.post("/ask")
def ask(inp: AskIn) -> Dict[str, Any]:
    doc_id = inp.doc_id or (LAST_DOC_FILE.read_text(encoding="utf-8").strip() if LAST_DOC_FILE.exists() else None)
    if not doc_id:
        raise HTTPException(400, "No doc_id provided and no last_doc_id saved")
    meta_path = STORE / "texts" / f"{doc_id}.json"
    if not meta_path.exists():
        raise HTTPException(404, f"doc_id {doc_id} not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks, doc_lang, index_kind = meta["chunks"], meta.get("lang", "en"), meta.get("index_kind", "faiss")
    q_vecs = encode_texts([inp.question], lang=doc_lang)
    hits = []
    scores, idxs = [], []
    if index_kind == "faiss":
        idx_path = STORE / "indexes" / f"{doc_id}.faiss"
        index = faiss.read_index(str(idx_path))
        D, I = index.search(q_vecs, int(inp.top_k))
        scores, idxs = D[0].tolist(), I[0].tolist()
    else:
        aidx_path = STORE / "indexes" / f"{doc_id}.ann"
        aidx = AnnoyIndex(q_vecs.shape[1], metric="angular")
        aidx.load(str(aidx_path))
        idxs = aidx.get_nns_by_vector(q_vecs[0].tolist(), int(inp.top_k), include_distances=False)
        scores = [1.0] * len(idxs)
    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        if idx == -1:
            continue
        text = chunks[idx]
        page_hint = None
        if text.startswith("[Page "):
            end = text.find("]\n")
            if end != -1:
                page_hint = text[1:end]
        hits.append({"rank": rank, "score": round(float(score), 4), "page_hint": page_hint, "excerpt": text[:400]})
    joined = "\n\n".join(h["excerpt"] for h in hits) if hits else "No relevant excerpts found."
    summary = summarize_text(joined, doc_lang)
    return {"question": inp.question, "doc_id_used": doc_id, "language": doc_lang, "answer": summary, "citations": hits}

# =========================
# Utilities
# =========================
@APP.get("/last_doc")
def get_last_doc():
    return {"last_doc_id": LAST_DOC_FILE.read_text(encoding="utf-8").strip()} if LAST_DOC_FILE.exists() else {"last_doc_id": None}

@APP.get("/deps")
def deps_status():
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
        "annoy_available": AnnoyIndex is not None,
        "summarizers_loaded": {"en": SUM_EN is not None, "zh": SUM_ZH is not None},
    }

@APP.get("/warmup")
def warmup():
    try:
        _ = get_sum_en()
        _ = get_sum_zh()
        _ = get_embedder_model("en")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("app:APP", host="0.0.0.0", port=port, reload=True)
