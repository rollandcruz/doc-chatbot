# app.py
import os
import tempfile
from pathlib import Path
import streamlit as st
from chatbot import get_response
from ingest import ingest_file

st.set_page_config(page_title="Doc Chatbot", page_icon="🤖")
st.title("Chat with your documents")

# ── Sidebar: file uploader ───────────────────────────────────────
with st.sidebar:
    st.header("Your Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs, Word docs, or text files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            progress = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    ingest_file(tmp_path, uploaded_file.name)
                os.unlink(tmp_path)
                progress.progress((i + 1) / len(uploaded_files))
            st.success(f"{len(uploaded_files)} file(s) ready!")
            st.session_state.docs_loaded = True

    if st.session_state.get("docs_loaded") or Path("vectordb").exists():
        st.info("Documents loaded. Ask anything below.")
    else:
        st.warning("Upload documents to get started.")

# ── Chat ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask something about your documents..."):
    if not st.session_state.get("docs_loaded") and not Path("vectordb").exists():
        st.warning("Please upload and process documents first.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_response(question, st.session_state.history)
            st.write(answer)
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": answer})