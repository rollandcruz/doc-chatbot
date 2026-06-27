# chatbot.py
import anthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_FOLDER = "vectordb"

client = anthropic.Anthropic()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)

def get_response(question: str, history: list) -> str:
    # 1. Find the 4 most relevant chunks from the vector store
    results = db.similarity_search(question, k=4)
    context = "\n\n".join(r.page_content for r in results)

    # 2. Build the system prompt with the retrieved context
    system = f"""You are a helpful assistant that answers questions based on the provided documents.
Only use the context below to answer. If the answer isn't in the context, say so.

CONTEXT:
{context}"""

    # 3. Pass full conversation history so it remembers previous exchanges
    messages = history + [{"role": "user", "content": question}]

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=system,
        messages=messages
    )
    return response.content[0].text