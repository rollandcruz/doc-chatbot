# ingest.py
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import PyPDF2

DOCS_FOLDER = "docs"
DB_FOLDER   = "vectordb"

def load_text(path: Path) -> str:
    if path.suffix == ".pdf":
        text = []
        with open(path, "rb") as f:
            for page in PyPDF2.PdfReader(f).pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    return path.read_text(encoding="utf-8")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_file(file_path: str, original_name: str = None):
    """Ingest a single file into ChromaDB. Used by the Streamlit uploader."""
    p = Path(file_path)
    name = original_name or p.name
    text = load_text(p)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text], metadatas=[{"source": name}])

    embeddings = get_embeddings()

    if Path(DB_FOLDER).exists():
        db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
        db.add_documents(chunks)
    else:
        Chroma.from_documents(chunks, embeddings, persist_directory=DB_FOLDER)

    print(f"Ingested {name} ({len(chunks)} chunks)")

def ingest():
    """Bulk ingest all files from the docs/ folder."""
    docs_path = Path(DOCS_FOLDER)
    all_texts, all_metas = [], []

    for file in docs_path.iterdir():
        if file.suffix in (".pdf", ".txt", ".docx"):
            print(f"Loading {file.name}...")
            all_texts.append(load_text(file))
            all_metas.append({"source": file.name})

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(all_texts, metadatas=all_metas)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = get_embeddings()
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_FOLDER)
    print(f"Saved to {DB_FOLDER}/")

if __name__ == "__main__":
    ingest()