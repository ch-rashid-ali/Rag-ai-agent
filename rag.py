from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv


load_dotenv()

DEFAULT_DB_DIR = Path("./chroma_db")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and query a PDF-based RAG system with optional LLM support."
    )
    parser.add_argument("--pdf", type=str, help="Path to a PDF file to index")
    parser.add_argument("--query", type=str, help="Question to ask the indexed knowledge base")
    parser.add_argument(
        "--db-dir",
        type=str,
        default=str(DEFAULT_DB_DIR),
        help="Directory for the Chroma vector database",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector database from the PDF even if it already exists",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers embedding model",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum size of each text chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between adjacent chunks",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "groq", "ollama", "fallback"],
        default="fallback",
        help="LLM provider to use for answer generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for the selected provider",
    )
    return parser


def ensure_dependencies() -> None:
    try:
        import langchain  # noqa: F401
        import langchain_community  # noqa: F401
        import langchain_huggingface  # noqa: F401
        import langchain_chroma  # noqa: F401
        import sentence_transformers  # noqa: F401
        import pypdf  # noqa: F401
        import chromadb  # noqa: F401
    except ImportError as exc:
        print("Missing required packages. Install them with:", file=sys.stderr)
        print("    pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(str(exc)) from exc


def build_vector_store(
    pdf_path: Path,
    db_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    if not documents:
        raise ValueError(f"No pages were loaded from {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf-rag",
        persist_directory=str(db_dir),
    )
    vector_store.persist()
    return vector_store


def load_vector_store(db_dir: Path, embedding_model: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings,
        collection_name="pdf-rag",
    )


def get_provider_api_key(provider: str) -> str | None:
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
        return key

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your environment or .env file.")
        return key

    if provider == "ollama":
        return None

    return None


def get_llm(provider: str, model: str):
    if provider == "fallback":
        return None

    api_key = get_provider_api_key(provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0, api_key=api_key)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0, api_key=api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0)

    raise ValueError(f"Unsupported provider: {provider}")


def build_prompt(query: str, docs: List[object]) -> str:
    context = "\n\n".join(doc.page_content[:1800] for doc in docs[:4])
    return f"""You are a precise document-grounded assistant.
Only use the retrieved context below to answer the user's question.
Do not answer from outside the document.
If the context does not contain the answer, say you do not know.

Question: {query}

Retrieved context:
{context}

Answer briefly and accurately.
"""


def generate_answer(query: str, docs: List[object], provider: str, model: str) -> str:
    llm = get_llm(provider, model)
    if llm is None:
        if not docs:
            return "No relevant context was found."

        joined = "\n\n".join(doc.page_content[:800] for doc in docs[:3])
        return (
            "A language model provider was not configured, so here is the best retrieved context:\n\n"
            f"{joined}"
        )

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful and concise assistant."),
            ("human", build_prompt(query, docs)),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({})
    return getattr(response, "content", str(response))


def answer_query(
    query: str,
    db_dir: Path,
    embedding_model: str,
    provider: str,
    model: str,
) -> str:
    if not (db_dir / "chroma.sqlite3").exists():
        raise FileNotFoundError(
            f"No vector database found at {db_dir}. Build it first with --pdf <your-file.pdf>."
        )

    vector_store = load_vector_store(db_dir, embedding_model)
    docs = vector_store.similarity_search(query, k=4)
    return generate_answer(query, docs, provider, model)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ensure_dependencies()

    pdf_path = Path(args.pdf).expanduser() if args.pdf else None
    db_dir = Path(args.db_dir).expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        if args.rebuild or not (db_dir / "chroma.sqlite3").exists():
            print(f"Building index for {pdf_path}...")
            build_vector_store(
                pdf_path=pdf_path,
                db_dir=db_dir,
                embedding_model=args.embedding_model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        else:
            print(f"Using existing vector database at {db_dir}")

    if args.query:
        print(answer_query(
            query=args.query,
            db_dir=db_dir,
            embedding_model=args.embedding_model,
            provider=args.provider,
            model=args.model,
        ))
    elif not args.pdf:
        parser.print_help()


if __name__ == "__main__":
    main()
