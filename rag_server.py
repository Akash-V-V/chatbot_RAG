
import os
import re
import time
import uuid
import logging
from typing import Optional

import fitz                          # PyMuPDF — proper PDF text extraction
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ── Load .env ────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv()  # also check current directory

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Flask ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")

# ── Config ────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "rag-docs")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_DIM    = 384   # all-MiniLM-L6-v2 output dimension
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP    = 80
DEFAULT_TOP_K    = int(os.getenv("TOP_K", "5"))
NAMESPACE        = "default"

# ── Clients ───────────────────────────────────────────────────
pc          : Optional[Pinecone] = None
index                            = None
groq_client : Optional[Groq]    = None
embed_model : Optional[SentenceTransformer] = None
stats = {"docs_indexed": 0, "vectors_total": 0, "queries_answered": 0}


def init_clients():
    global pc, index, groq_client, embed_model

    # Load semantic embedding model locally (downloads once ~90MB)
    log.info("Loading embedding model 'all-MiniLM-L6-v2'…")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Embedding model ready. Dim=%d", embed_model.get_sentence_embedding_dimension())

    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _ensure_index()
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
    log.info("Clients initialized. Pinecone=%s  Groq=%s  Model=%s",
             bool(pc), bool(groq_client), GROQ_MODEL)


def _ensure_index():
    global index
    existing = {i.name: i for i in pc.list_indexes()}

    if INDEX_NAME in existing:
        # Check if dimension matches — if not, delete and recreate
        current_dim = existing[INDEX_NAME].dimension
        if current_dim != EMBEDDING_DIM:
            log.warning(
                "Index '%s' has dimension %d but we need %d. Deleting and recreating…",
                INDEX_NAME, current_dim, EMBEDDING_DIM
            )
            pc.delete_index(INDEX_NAME)
            # Wait for deletion
            for _ in range(20):
                if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
                    break
                time.sleep(2)
            log.info("Old index deleted.")
            existing = {}  # force recreation below

    if INDEX_NAME not in existing:
        log.info("Creating Pinecone index '%s' (dim=%d)…", INDEX_NAME, EMBEDDING_DIM)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        for _ in range(30):
            if pc.describe_index(INDEX_NAME).status.get("ready"):
                break
            time.sleep(2)

    index = pc.Index(INDEX_NAME)
    log.info("Index '%s' ready (dim=%d).", INDEX_NAME, EMBEDDING_DIM)


# ═══════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract clean text from PDF, TXT, or MD files.
    Uses PyMuPDF for PDFs — handles formatted/scanned-text PDFs properly.
    """
    ext = filename.rsplit('.', 1)[-1].lower()

    if ext == 'pdf':
        return _extract_pdf(file_bytes)
    else:
        # txt / md — just decode
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode('utf-8', errors='replace')


def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF (fitz)."""
    text_parts = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")          # clean text extraction
            page_text = _clean_text(page_text)
            if page_text:
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        doc.close()
    except Exception as e:
        log.error("PDF extraction error: %s", e)
        raise RuntimeError(f"Could not read PDF: {e}")

    full_text = "\n\n".join(text_parts)
    if not full_text.strip():
        raise RuntimeError("PDF appears to be scanned/image-only. No extractable text found.")
    return full_text


def _clean_text(text: str) -> str:
    """Remove garbled characters, normalize whitespace."""
    # Remove non-printable / binary garbage characters
    text = re.sub(r'[^\x20-\x7E\n\r\t\u00A0-\u024F]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════
# CHUNKING
# ═══════════════════════════════════════════════════════════════

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, respecting sentence boundaries.
    Overlap ensures context continuity across chunk boundaries.
    """
    text = _clean_text(text)
    # Split on sentence endings or paragraph breaks
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    chunks, current = [], ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > size and current:
            chunks.append(current.strip())
            # Keep last N words as overlap for next chunk
            words = current.split()
            tail  = " ".join(words[-max(1, overlap // 8):])
            current = tail + " " + sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    # Filter out chunks that are too short to be meaningful
    return [c for c in chunks if len(c) > 40]


# ═══════════════════════════════════════════════════════════════
# EMBEDDINGS — sentence-transformers (local, no API key needed)
# Model: all-MiniLM-L6-v2
#   - 384 dimensions
#   - Understands semantic meaning (education, IT, technical docs)
#   - Downloads once (~90MB), runs fully offline after that
#   - Similarity scores of 0.6–0.9 for relevant content
# ═══════════════════════════════════════════════════════════════

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode a list of texts into semantic vectors using sentence-transformers."""
    if not embed_model:
        raise RuntimeError("Embedding model not initialized")
    # encode() returns numpy array → convert to plain Python list
    vectors = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


# ═══════════════════════════════════════════════════════════════
# PINECONE
# ═══════════════════════════════════════════════════════════════

def upsert_vectors(vectors: list[dict]) -> int:
    if not index:
        raise RuntimeError("Pinecone not initialized")
    upserted = 0
    for i in range(0, len(vectors), 100):
        res = index.upsert(vectors=vectors[i:i+100], namespace=NAMESPACE)
        upserted += res.get("upserted_count", len(vectors[i:i+100]))
    return upserted


def query_index(embedding: list[float], top_k: int = DEFAULT_TOP_K) -> list[dict]:
    if not index:
        raise RuntimeError("Pinecone not initialized")
    res = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
    )
    return res.get("matches", [])


# ═══════════════════════════════════════════════════════════════
# LLM GENERATION via Groq
# ═══════════════════════════════════════════════════════════════

def generate_answer(query: str, matches: list[dict], history: list[dict]) -> str:
    if not groq_client:
        if matches:
            ctx = "\n\n".join(m["metadata"].get("text","") for m in matches)
            return f"[No Groq key] Retrieved context:\n\n{ctx[:600]}"
        return "No relevant context found. Please ingest documents first."

    if matches:
        ctx_blocks = []
        for i, m in enumerate(matches):
            meta  = m.get("metadata", {})
            score = m.get("score", 0)
            ctx_blocks.append(
                f"[Source {i+1}: {meta.get('source','?')} | page {meta.get('page','?')} | score={score:.3f}]\n"
                f"{meta.get('text', '')}"
            )
        context = "\n\n---\n\n".join(ctx_blocks)
        system  = (
            "You are a precise, helpful assistant. Answer the user's question using ONLY "
            "the document context provided below. "
            "If the answer is not in the context, say so clearly — do not guess. "
            "Cite which source(s) you used in your answer.\n\n"
            f"DOCUMENT CONTEXT:\n{context}"
        )
    else:
        system = (
            "You are a helpful assistant. No relevant document context was found for this query. "
            "Tell the user to upload and ingest relevant documents first."
        )

    messages = [{"role": "system", "content": system}]
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    # Ensure last message is the current user query
    if not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": query})

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1500,
        temperature=0.2,      # lower temp = more factual / less hallucination
        messages=messages,
    )
    return resp.choices[0].message.content


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "pinecone":        bool(pc and index),
        "groq":            bool(groq_client),
        "embedding_model": bool(embed_model),
        "model":           GROQ_MODEL,
        "index":           INDEX_NAME,
        "embedding_dim":   EMBEDDING_DIM,
    })


@app.route("/ingest", methods=["POST"])
def ingest():
    """Accept file uploads or JSON documents, chunk, embed, and upsert to Pinecone."""
    try:
        documents = []

        if request.files:
            for f in request.files.getlist("files"):
                file_bytes = f.read()
                try:
                    content = extract_text(file_bytes, f.filename)
                    documents.append({"filename": f.filename, "content": content})
                except RuntimeError as e:
                    return jsonify({"error": str(e), "filename": f.filename}), 400
        elif request.is_json:
            documents = request.json.get("documents", [])
        else:
            return jsonify({"error": "No documents provided"}), 400

        results      = []
        total_vectors = 0

        for doc in documents:
            filename = doc.get("filename", "unknown")
            content  = doc.get("content",  "")

            chunks = chunk_text(content)
            if not chunks:
                results.append({"filename": filename, "status": "skipped", "reason": "no text extracted"})
                continue

            log.info("'%s' → %d chunks", filename, len(chunks))
            embeddings = embed_texts(chunks)

            vectors = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                # Extract page number if present in chunk prefix
                page_match = re.match(r'\[Page (\d+)\]', chunk)
                page_num   = int(page_match.group(1)) if page_match else i + 1
                # Strip the [Page N] prefix from stored text
                clean_chunk = re.sub(r'^\[Page \d+\]\s*', '', chunk).strip()

                vectors.append({
                    "id":     f"{re.sub(r'[^a-z0-9]','_',filename.lower())}_{uuid.uuid4().hex[:8]}_{i}",
                    "values": emb,
                    "metadata": {
                        "source":      filename,
                        "page":        page_num,
                        "chunkIndex":  i,
                        "totalChunks": len(chunks),
                        "text":        clean_chunk[:1000],
                    },
                })

            upserted = upsert_vectors(vectors) if index else 0
            total_vectors += upserted
            stats["docs_indexed"]  += 1
            stats["vectors_total"] += upserted

            results.append({
                "filename": filename,
                "status":   "indexed",
                "chunks":   len(chunks),
                "vectors":  upserted,
            })
            log.info("Ingested '%s': %d chunks, %d vectors", filename, len(chunks), upserted)

        return jsonify({"success": True, "results": results, "total_vectors": total_vectors})

    except Exception as e:
        log.exception("Ingest error")
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    """Embed query → Pinecone search → Groq generation → return answer + sources."""
    try:
        data    = request.get_json()
        q       = data.get("query", "").strip()
        top_k   = int(data.get("top_k", DEFAULT_TOP_K))
        history = data.get("history", [])

        if not q:
            return jsonify({"error": "Query is empty"}), 400

        [q_emb]  = embed_texts([q])
        matches  = query_index(q_emb, top_k) if index else []
        answer   = generate_answer(q, matches, history)
        stats["queries_answered"] += 1

        sources = [
            {
                "id":     m.get("id"),
                "score":  round(m.get("score", 0), 4),
                "source": m["metadata"].get("source", ""),
                "page":   m["metadata"].get("page", ""),
                "text":   m["metadata"].get("text", "")[:300],
            }
            for m in matches
        ]

        return jsonify({
            "answer":        answer,
            "sources":       sources,
            "context_found": len(matches) > 0,
            "model":         GROQ_MODEL,
        })

    except Exception as e:
        log.exception("Query error")
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    index_info = {}
    if index:
        try:
            desc = index.describe_index_stats()
            index_info = {
                "total_vector_count": desc.get("total_vector_count", 0),
                "dimension":          desc.get("dimension", EMBEDDING_DIM),
            }
        except Exception as e:
            index_info = {"error": str(e)}
    return jsonify({"pipeline": stats, "index": index_info, "model": GROQ_MODEL})


@app.route("/clear", methods=["DELETE"])
def clear_index():
    try:
        if index:
            index.delete(delete_all=True, namespace=NAMESPACE)
            stats["vectors_total"] = 0
            log.info("Index cleared.")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Start ─────────────────────────────────────────────────────
if __name__ == "__main__":
    init_clients()
    app.run(host="0.0.0.0", port=5000, debug=False)