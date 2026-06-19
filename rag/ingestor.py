from typing import Optional
"""
rag/ingestor.py — Document ingestion pipeline for OnboardIQ
Supports: PDF, URLs, plain text, markdown
Uses parent-child chunking strategy for better retrieval
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
)
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class DocumentIngestor:
    """
    Handles loading, chunking, and storing documents into ChromaDB.
    Uses Parent-Child chunking:
      - Child chunks  → stored in ChromaDB for retrieval
      - Parent chunks → stored in a dict for full context when answering
    """

    def __init__(self):
        logger.info("Initialising DocumentIngestor...")

        self.embeddings = SentenceTransformerEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )

        # Parent splitter — large chunks sent to LLM
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.PARENT_CHUNK_SIZE,
            chunk_overlap=config.PARENT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Child splitter — small chunks used for semantic search
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # ChromaDB for child chunks
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=str(config.CHROMA_DIR),
        )

        # In-memory parent store: {parent_id: Document}
        self.parent_store: Dict[str, Document] = {}

        logger.info("DocumentIngestor ready.")

    # ── Loaders ───────────────────────────────────────────────────────────────

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file."""
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "pdf"
            doc.metadata["file_name"] = Path(file_path).name
        return docs

    def load_url(self, url: str) -> List[Document]:
        """Load content from a URL."""
        logger.info(f"Loading URL: {url}")
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "url"
            doc.metadata["file_name"] = url
        return docs

    def load_text(self, file_path: str) -> List[Document]:
        """Load a plain text file."""
        logger.info(f"Loading text: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "text"
            doc.metadata["file_name"] = Path(file_path).name
        return docs

    def load_markdown(self, file_path: str) -> List[Document]:
        """Load a markdown file."""
        logger.info(f"Loading markdown: {file_path}")
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "markdown"
            doc.metadata["file_name"] = Path(file_path).name
        return docs

    def load_directory(self, dir_path: str) -> List[Document]:
        """Auto-detect and load all supported files in a directory."""
        logger.info(f"Loading directory: {dir_path}")
        all_docs = []
        dir_path = Path(dir_path)
        for file in dir_path.rglob("*"):
            if file.suffix.lower() == ".pdf":
                all_docs.extend(self.load_pdf(str(file)))
            elif file.suffix.lower() in [".txt"]:
                all_docs.extend(self.load_text(str(file)))
            elif file.suffix.lower() in [".md", ".markdown"]:
                all_docs.extend(self.load_markdown(str(file)))
        logger.info(f"Loaded {len(all_docs)} documents from directory.")
        return all_docs

    # ── Parent-Child Chunking ─────────────────────────────────────────────────

    def _make_parent_id(self, doc: Document, index: int) -> str:
        """Generate a stable ID for a parent chunk."""
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
        return f"parent_{index}_{content_hash}"

    def chunk_documents(self, docs: List[Document]) -> tuple[List[Document], Dict[str, Document]]:
        """
        Split documents into parent and child chunks.
        Returns:
          - child_chunks: List[Document] with parent_id in metadata
          - parent_store: Dict[parent_id -> parent Document]
        """
        parent_chunks = self.parent_splitter.split_documents(docs)
        child_chunks  = []
        parent_store  = {}

        for i, parent in enumerate(parent_chunks):
            parent_id = self._make_parent_id(parent, i)
            parent.metadata["parent_id"] = parent_id
            parent_store[parent_id] = parent

            # Split parent into smaller children
            children = self.child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = parent_id
                child_chunks.append(child)

        logger.info(
            f"Chunked into {len(parent_chunks)} parent chunks "
            f"and {len(child_chunks)} child chunks."
        )
        return child_chunks, parent_store

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, docs: List[Document]) -> int:
        """
        Full ingestion pipeline:
          1. Chunk into parent/child
          2. Store children in ChromaDB
          3. Cache parents in memory
        Returns number of child chunks stored.
        """
        if not docs:
            logger.warning("No documents to ingest.")
            return 0

        child_chunks, parent_store = self.chunk_documents(docs)

        # Merge parent store
        self.parent_store.update(parent_store)

        # Store child chunks in ChromaDB
        self.vectorstore.add_documents(child_chunks)

        logger.success(f"Ingested {len(child_chunks)} child chunks into ChromaDB.")
        return len(child_chunks)

    def ingest_pdf(self, file_path: str) -> int:
        return self.ingest(self.load_pdf(file_path))

    def ingest_url(self, url: str) -> int:
        return self.ingest(self.load_url(url))

    def ingest_text(self, file_path: str) -> int:
        return self.ingest(self.load_text(file_path))

    def ingest_directory(self, dir_path: str) -> int:
        return self.ingest(self.load_directory(dir_path))

    def get_vectorstore(self) -> Chroma:
        return self.vectorstore

    def get_parent_store(self) -> Dict[str, Document]:
        return self.parent_store

    def get_all_texts(self) -> List[str]:
        """Return all raw child chunk texts (for BM25 indexing)."""
        results = self.vectorstore.get()
        return results.get("documents", [])

    def get_collection_count(self) -> int:
        return self.vectorstore._collection.count()


# ── Module-level singleton + convenience functions ────────────────────────────
_ingestor: Optional[DocumentIngestor] = None

def _get_ingestor() -> DocumentIngestor:
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor()
    return _ingestor


def ingest_documents(
    pdf_paths:  Optional[list] = None,
    urls:       Optional[list] = None,
    text_paths: Optional[list] = None,
) -> dict:
    """
    Convenience function to ingest multiple document types at once.
    Returns a summary dict.
    """
    ing   = _get_ingestor()
    total = 0
    docs  = []

    for path in (pdf_paths or []):
        docs.extend(ing.load_pdf(path))
    for url in (urls or []):
        docs.extend(ing.load_url(url))
    for path in (text_paths or []):
        ext = Path(path).suffix.lower()
        if ext in (".md", ".markdown"):
            docs.extend(ing.load_markdown(path))
        else:
            docs.extend(ing.load_text(path))

    if not docs:
        return {"status": "no_docs", "total_docs": 0, "child_chunks": 0}

    chunks = ing.ingest(docs)
    return {"status": "success", "total_docs": len(docs), "child_chunks": chunks}


def get_vectorstore():
    """Return the shared ChromaDB vectorstore instance."""
    return _get_ingestor().get_vectorstore()


def get_parent_chunk(parent_id: str):
    """Look up a parent chunk by ID."""
    return _get_ingestor().parent_store.get(parent_id)
