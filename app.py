import os
import tempfile
from pathlib import Path

import streamlit as st

from rag import DEFAULT_DB_DIR, answer_query, build_vector_store


def resolve_pdf_path(base_dir: Path | None = None) -> Path:
    workspace_dir = (base_dir or Path(__file__).resolve().parent).resolve()

    env_path = os.getenv("RAG_PDF_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            return candidate

    pdf_candidates = sorted(workspace_dir.rglob("*.pdf"))
    if pdf_candidates:
        return pdf_candidates[0].resolve()

    raise FileNotFoundError(
        "No PDF file was found in the workspace. Place a PDF in the project folder or set RAG_PDF_PATH."
    )


def build_db_dir(pdf_path: Path) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pdf_path.stem)
    return Path(DEFAULT_DB_DIR) / safe_name


st.set_page_config(page_title="PDF RAG Assistant", page_icon="📘", layout="wide")

st.title("📘 PDF RAG Assistant")
st.write("Upload a PDF or use one already in the workspace. Answers will be grounded only in that document.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_area("Question", placeholder="Ask something about the document...")

if st.button("Ask"):
    if not query.strip():
        st.error("Please enter a question.")
        st.stop()

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_pdf_path = Path(tmpdir) / uploaded_file.name
            temp_pdf_path.write_bytes(uploaded_file.getvalue())
            pdf_path = temp_pdf_path
    else:
        try:
            pdf_path = resolve_pdf_path()
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_pdf_path = Path(tmpdir) / pdf_path.name
            temp_pdf_path.write_bytes(pdf_path.read_bytes())

            db_dir = build_db_dir(temp_pdf_path)
            db_dir.mkdir(parents=True, exist_ok=True)

            build_vector_store(
                pdf_path=temp_pdf_path,
                db_dir=db_dir,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                chunk_size=1000,
                chunk_overlap=200,
            )
            answer = answer_query(
                query=query,
                db_dir=db_dir,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                provider="fallback",
                model="gpt-4o-mini",
            )
            st.write(answer)
    except Exception as exc:
        st.error(f"Error: {exc}")
