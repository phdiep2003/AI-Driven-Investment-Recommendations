# src/rag_engine.py
"""
RAG Engine for CFA & Islamic Finance rules retrieval.

Features
--------
- Loads documents from a folder (txt, md, pdf when available)
- Chunks text intelligently using LangChain text splitters
- Builds a FAISS (default) or Chroma vector store with OpenAI embeddings
- Persists/loads index to/from disk
- Retrieves rule snippets with context-aware queries
- Adds metadata: topic, source, language

Notes
-----
- Designed to work in both notebooks and Flask apps.
- OpenAI Embeddings are used when available. For offline/testing, a stub
  embedding is provided if environment variable RAG_USE_STUB_EMBEDDINGS=1.
- Chroma backend is optional; FAISS is the default.
"""

from __future__ import annotations

import os
import re
import uuid
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --- Optional deps (import defensively) ---
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None  # type: ignore

try:
    from langchain_community.vectorstores import FAISS, Chroma
except Exception:  # pragma: no cover
    FAISS = None  # type: ignore
    Chroma = None  # type: ignore

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    # Minimal shim to avoid hard dependency at import time
    @dataclass
    class Document:  # type: ignore
        page_content: str
        metadata: Dict
        def __init__(self, page_content: str, metadata: Optional[Dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from langdetect import detect as _ld_detect  # language detection (optional)
except Exception:  # pragma: no cover
    _ld_detect = None

# pdf extraction (optional)
try:
    import pypdf
except Exception:  # pragma: no cover
    pypdf = None


# ---------------------- Helpers ---------------------- #
FIN_ISLAMIC_KWS = {
    "shariah", "shari'ah", "sharia", "riba", "gharar", "maisir", "sukuk", "murabaha",
    "musharakah", "ijara", "takaful", "zakat", "fatwa", "AAOIFI", "IOF", "islamic finance"
}
FIN_CFA_KWS = {
    "cfa", "charterholder", "ethics", "standards of professional conduct",
    "soft dollar", "gips", "asset manager code", "fiduciary", "material nonpublic",
    "misrepresentation", "misconduct", "prudence", "suitability", "code of ethics"
}
ESG_KWS = {"esg", "sustainability", "governance", "environment", "social", "shariah-esg"}

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def _safe_language(text: str) -> str:
    """Best-effort language detection."""
    if not text or len(text.strip()) < 20:
        return "unknown"
    if _ld_detect is None:
        return "unknown"
    try:
        return _ld_detect(text[:1000])
    except Exception:
        return "unknown"


def _infer_topic(text: str) -> str:
    """Heuristic topic inference for metadata enrichment."""
    t = text.lower()
    score_islamic = sum(1 for k in FIN_ISLAMIC_KWS if k in t)
    score_cfa = sum(1 for k in FIN_CFA_KWS if k in t)
    score_esg = sum(1 for k in ESG_KWS if k in t)
    if score_islamic >= max(score_cfa, score_esg, 0) and score_islamic > 0:
        return "islamic_finance"
    if score_cfa >= max(score_islamic, score_esg, 0) and score_cfa > 0:
        return "cfa_ethics"
    if score_esg > 0:
        return "esg"
    return "general_finance"


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_file(path: str) -> str:
    if pypdf is None:
        # Graceful fallback if pypdf not available
        return ""
    try:
        reader = pypdf.PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception:
        return ""


def _iter_files(folder: str) -> List[str]:
    exts = {".txt", ".md", ".pdf"}
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name.lower())[1] in exts:
                paths.append(os.path.join(root, name))
    return sorted(paths)


def _default_text_splitter() -> "RecursiveCharacterTextSplitter":
    if RecursiveCharacterTextSplitter is None:
        raise ImportError(
            "langchain 'RecursiveCharacterTextSplitter' is required. "
            "Install with: pip install langchain"
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            "، ", "۔ ",  # Arabic/Urdu punctuation (common for Islamic finance texts)
            "。", "！", "？",  # CJK punctuation
            " "
        ],
        length_function=len,
        is_separator_regex=False,
    )


# ---------------------- Embeddings ---------------------- #
class _StubEmbeddings:
    """Deterministic stub embeddings for offline/dev usage (not semantic!)."""
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_vec(text)

    def _hash_vec(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        # Repeat/cycle to requested dim
        vals = list(h)
        out: List[float] = []
        i = 0
        while len(out) < self.dim:
            out.append((vals[i % len(vals)] - 128) / 128.0)
            i += 1
        return out


def _get_embeddings():
    """Return an embeddings object (OpenAI or stub)."""
    use_stub = os.getenv("RAG_USE_STUB_EMBEDDINGS", "").strip() in {"1", "true", "True"}
    if use_stub:
        return _StubEmbeddings()
    if OpenAIEmbeddings is None:
        raise ImportError(
            "OpenAI embeddings not available. Install with: pip install langchain-openai "
            "or set RAG_USE_STUB_EMBEDDINGS=1 for a stub."
        )
    # Model name can be overridden via env: OPENAI_EMBEDDING_MODEL
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # OPENAI_API_KEY should be present in environment
    return OpenAIEmbeddings(model=model)


# ---------------------- Public API ---------------------- #
def load_documents(folder: str) -> List[Dict]:
    """
    Read *.txt, *.md, *.pdf (if possible) from `folder`.

    Returns
    -------
    List[Dict]
        Each dict: {"id": str, "text": str, "source": str}
    """
    results: List[Dict] = []
    for path in _iter_files(folder):
        ext = os.path.splitext(path.lower())[1]
        text = ""
        if ext in {".txt", ".md"}:
            text = _read_text_file(path)
        elif ext == ".pdf":
            text = _read_pdf_file(path)
        text = (text or "").strip()
        if not text:
            continue
        results.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "source": os.path.abspath(path),
        })
    return results


def _docs_to_langchain_docs(docs: List[Dict]) -> List[Document]:
    splitter = _default_text_splitter()
    lc_docs: List[Document] = []
    for d in docs:
        raw_text = d.get("text", "")
        if not raw_text:
            continue
        # Chunk intelligently
        chunks = splitter.split_text(raw_text)
        language = _safe_language(raw_text)
        # Infer topic once per source; still attach to chunks
        topic = _infer_topic(raw_text)
        for i, ch in enumerate(chunks):
            lc_docs.append(
                Document(
                    page_content=ch,
                    metadata={
                        "topic": topic,
                        "source": d.get("source", "unknown"),
                        "language": language,
                        "chunk_id": f"{d.get('id','')}_{i}",
                        "doc_id": d.get("id", ""),
                    },
                )
            )
    return lc_docs


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def build_vector_index(
    docs: List[Dict],
    index_path: str,
    backend: str = "faiss",
    collection_name: str = "rag_rules",
) -> None:
    """
    Create a FAISS (default) or Chroma vector store and persist to `index_path`.

    Parameters
    ----------
    docs : List[Dict]
        Output from `load_documents`.
    index_path : str
        Directory to persist vector store.
    backend : str
        "faiss" (default) or "chroma".
    collection_name : str
        Chroma collection name (if using Chroma).
    """
    if not docs:
        raise ValueError("No documents provided to build_vector_index.")

    embeddings = _get_embeddings()
    lc_docs = _docs_to_langchain_docs(docs)
    _ensure_dir(index_path)

    backend = backend.lower().strip()
    if backend == "faiss":
        if FAISS is None:
            raise ImportError("FAISS vectorstore not available. Install langchain-community.")
        vs = FAISS.from_documents(lc_docs, embeddings)
        vs.save_local(index_path)
    elif backend == "chroma":
        if Chroma is None:
            raise ImportError("Chroma vectorstore not available. Install langchain-community and chromadb.")
        # Persist Chroma DB
        vs = Chroma.from_documents(
            lc_docs, embeddings, collection_name=collection_name, persist_directory=index_path
        )
        vs.persist()
    else:
        raise ValueError("Unsupported backend. Use 'faiss' or 'chroma'.")


def _load_vector_index(
    index_path: str,
    backend: str = "faiss",
    collection_name: str = "rag_rules",
):
    embeddings = _get_embeddings()
    backend = backend.lower().strip()
    if backend == "faiss":
        if FAISS is None:
            raise ImportError("FAISS vectorstore not available. Install langchain-community.")
        # allow_dangerous_deserialization supports modern langchain >=0.2 safety gate
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    elif backend == "chroma":
        if Chroma is None:
            raise ImportError("Chroma vectorstore not available. Install langchain-community and chromadb.")
        return Chroma(collection_name=collection_name, persist_directory=index_path, embedding_function=embeddings)
    else:
        raise ValueError("Unsupported backend. Use 'faiss' or 'chroma'.")


def query_rules(
    query: str,
    k: int = 5,
    index_path: str = "./vector_index",
    backend: str = "faiss",
) -> List[str]:
    """
    Query the vector store and return top-k relevant text passages.

    Parameters
    ----------
    query : str
        User query.
    k : int
        Number of passages to return.
    index_path : str
        Path to persisted index.
    backend : str
        "faiss" (default) or "chroma".

    Returns
    -------
    List[str]
        Top-k passages (page_content).
    """
    if not query or not query.strip():
        return []
    vs = _load_vector_index(index_path=index_path, backend=backend)
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]


def _compose_policy_query(user_payload: Dict) -> Tuple[str, List[str]]:
    """
    Build a retrieval query string and tags based on user payload context.

    Expected keys in user_payload (optional):
        - 'legalFlags': Dict e.g., {'shariah': True}
        - 'esgFlags': List[str] e.g., ['environment', 'governance']
        - 'riskLevel': str in {'low','medium','high'}
        - 'jurisdiction': Optional[str]
        - 'instruments': Optional[List[str]]

    Returns
    -------
    query : str
    tags : List[str]  (for debugging/metadata)
    """
    legal = (user_payload or {}).get("legalFlags", {}) or {}
    esg = (user_payload or {}).get("esgFlags", []) or []
    risk = (user_payload or {}).get("riskLevel", "")
    juris = (user_payload or {}).get("jurisdiction", "")
    instr = (user_payload or {}).get("instruments", []) or []

    parts = []
    tags = []

    # Core intent
    parts.append("investment rules policies constraints")
    tags.append("core:rules")

    # Risk profile
    if risk:
        parts.append(f"risk profile {risk}")
        tags.append(f"risk:{risk}")

    # Instruments
    if instr:
        parts.append(" ".join(instr))
        tags.append(f"instruments:{','.join(instr)}")

    # Jurisdiction
    if juris:
        parts.append(f"jurisdiction {juris}")
        tags.append(f"jurisdiction:{juris}")

    # ESG flags
    if esg:
        parts.append("ESG " + " ".join(esg))
        tags.append(f"esg:{','.join(esg)}")

    # Shariah / Islamic finance
    if isinstance(legal, dict) and legal.get("shariah", False):
        parts.append(
            "Shariah Islamic finance AAOIFI riba gharar maisir sukuk murabaha musharakah ijara takaful"
        )
        tags.append("legal:shariah")

    # CFA Ethics (if explicitly requested)
    if isinstance(legal, dict) and legal.get("cfaEthics", False):
        parts.append("CFA Code Standards of Professional Conduct GIPS soft dollar MNPI misrepresentation")
        tags.append("legal:cfa")

    query = " ".join(parts)
    return query, tags


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def retrieve_relevant_rules(
    user_payload: Dict,
    index_path: str = "./vector_index",
    backend: str = "faiss",
    k: int = 6,
) -> Dict:
    """
    Construct prompting context based on user payload and retrieve from vector store.

    Parameters
    ----------
    user_payload : dict
        {
          "legalFlags": {"shariah": bool, "cfaEthics": bool, ...},
          "esgFlags": ["environment", "social", "governance"],
          "riskLevel": "low"|"medium"|"high",
          "jurisdiction": "Qatar" | "US" | ...,
          "instruments": ["equity","sukuk",...]
        }
    index_path : str
        Vector store path.
    backend : str
        "faiss" (default) or "chroma".
    k : int
        Candidate snippets to retrieve.

    Returns
    -------
    Dict:
        {
            "ethical_rules": [...],
            "regulatory_constraints": [...],
            "shariah_notes": [...]
        }
    """
    vs = _load_vector_index(index_path=index_path, backend=backend)

    # Build a composite query and several focused sub-queries for better recall
    base_q, _tags = _compose_policy_query(user_payload)
    queries = [base_q]

    # Targeted expansions
    legal = (user_payload or {}).get("legalFlags", {}) or {}
    esg = (user_payload or {}).get("esgFlags", []) or []
    risk = (user_payload or {}).get("riskLevel", "")

    if legal.get("shariah", False):
        queries.append("Shariah compliance screening rules riba gharar maisir purification zakat AAOIFI")
        queries.append("prohibited industries alcohol pork gambling conventional banking derivatives short-selling")

    if legal.get("cfaEthics", False):
        queries.append("CFA ethics Standards I-VII Code of Ethics professional conduct duties to clients MNPI")

    if esg:
        queries.append("ESG investment policy environmental social governance exclusions screening best practices")

    if risk:
        queries.append(f"risk suitability KYC investment policy statement IPS {risk} volatility drawdown limits")

    # Run retrieval
    candidate_docs: List[Document] = []
    for q in queries:
        try:
            candidate_docs.extend(vs.similarity_search(q, k=k))
        except Exception:
            continue

    # Partition into buckets using metadata topic + simple heuristics
    ethical_rules: List[str] = []
    regulatory_constraints: List[str] = []
    shariah_notes: List[str] = []

    for d in candidate_docs:
        txt = d.page_content.strip()
        topic = (d.metadata or {}).get("topic", "general_finance")
        source = (d.metadata or {}).get("source", "unknown")
        lang = (d.metadata or {}).get("language", "unknown")
        prefix = f"[{topic} | {lang}] {source} :: "

        # Very light classification
        low_txt = txt.lower()
        if topic == "islamic_finance" or any(k in low_txt for k in FIN_ISLAMIC_KWS):
            shariah_notes.append(prefix + txt)
        elif topic == "cfa_ethics" or "ethic" in low_txt or "mnpi" in low_txt or "gips" in low_txt:
            ethical_rules.append(prefix + txt)
        else:
            # Basic signs of regulatory constraints
            if any(k in low_txt for k in ["prohibited", "ban", "restricted", "limit", "regulator", "regulation",
                                          "suitability", "leverage limit", "concentration limit", "compliance"]):
                regulatory_constraints.append(prefix + txt)
            else:
                # Distribute by language/topic fallback
                if topic == "esg":
                    regulatory_constraints.append(prefix + txt)
                else:
                    ethical_rules.append(prefix + txt)

    # De-duplicate and trim
    ethical_rules = _unique_preserve_order(ethical_rules)[:k]
    regulatory_constraints = _unique_preserve_order(regulatory_constraints)[:k]
    shariah_notes = _unique_preserve_order(shariah_notes)[:k]

    return {
        "ethical_rules": ethical_rules,
        "regulatory_constraints": regulatory_constraints,
        "shariah_notes": shariah_notes,
    }


# ---------------------- Minimal CLI/Notebook helpers ---------------------- #
def save_index_config(index_path: str, config: Dict) -> None:
    """Persist a small JSON config next to the index for reproducibility."""
    _ensure_dir(index_path)
    with open(os.path.join(index_path, "rag_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_index_config(index_path: str) -> Dict:
    """Load JSON config if present; return {} otherwise."""
    p = os.path.join(index_path, "rag_config.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------- Example (commented) ---------------------- #
# if __name__ == "__main__":
#     # Example usage:
#     docs = load_documents("./rules_corpus")
#     build_vector_index(docs, "./vector_index", backend="faiss")
#     print(query_rules("What are Shariah constraints on equities?", k=3))
#     payload = {
#         "legalFlags": {"shariah": True, "cfaEthics": True},
#         "esgFlags": ["governance"],
#         "riskLevel": "medium",
#         "jurisdiction": "Qatar",
#         "instruments": ["equity", "sukuk"]
#     }
#     ctx = retrieve_relevant_rules(payload, "./vector_index", backend="faiss", k=5)
#     print(json.dumps(ctx, indent=2, ensure_ascii=False))
