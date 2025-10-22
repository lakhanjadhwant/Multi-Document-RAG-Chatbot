# app.py

import streamlit as st
from itertools import groupby
import frontend_utils as utils

# --- Page Setup and Session Initialization ---
st.set_page_config(page_title="📄 Multi-Doc RAG Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("📄 RAG Chatbot: Ask Questions to Multiple Documents")
utils.init_session_state()

# --- UI Components and Logic ---

def handle_file_upload(uploaded_files):
    """Processes uploaded files and updates session state."""
    with st.spinner(f'Processing {len(uploaded_files)} files...'):
        response_data, new_session_id = utils.call_upload_api(uploaded_files)
        if response_data:
            st.session_state.session_id = new_session_id
            st.session_state.processed_files = response_data.get('processed_files', [])
            st.session_state.messages = [{"role": "assistant", "content": "I've finished reading your documents. What would you like to know?"}]
            st.success(f"✅ Successfully indexed {st.session_state.processed_files} files!")
            st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.header("1. Start a New Chat Session")
    uploaded_files = st.file_uploader("Upload documents (.pdf, .docx, .txt, etc.)", type=["pdf", "docx", "txt", "csv", "xlsx"], accept_multiple_files=True)
    if st.button("Process Files", disabled=not uploaded_files):
        handle_file_upload(uploaded_files)
    st.divider()
    st.header("2. Current Session Info")
    if st.session_state.processed_files:
        st.info("Currently chatting with:"); [st.markdown(f"- `{f}`") for f in st.session_state.processed_files]
    else:
        st.info("No documents uploaded.")

# --- Main Chat Interface ---
st.header("Chat with Your Documents")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "candidates" in message:
            tabs = st.tabs([f"Response {i+1}" for i in range(len(message["candidates"]))])
            for i, candidate in enumerate(message["candidates"]):
                with tabs[i]:
                    answer = candidate.get("answer", {})
                    st.markdown(answer.get("summary", "No summary provided."))
                    if answer.get("data"):
                        st.markdown(utils.format_structured_data(answer["data"]))
                    
                    if candidate.get("reasoning"):
                        with st.expander("Show Reasoning 🧠"): st.info(candidate["reasoning"])
                    
                    source_doc_names = candidate.get("source_documents", [])
                    if source_doc_names and "sources" in message:
                        with st.expander("View Sources 📚"):
                            relevant_sources = [s for s in message["sources"] if s['filename'] in source_doc_names]
                            grouped_sources = groupby(sorted(relevant_sources, key=lambda x: x['filename']), key=lambda x: x['filename'])
                            for filename, chunks in grouped_sources:
                                st.info(f"**Excerpts from `{filename}`**:")
                                for chunk in chunks: st.markdown(f"...\n> {chunk['content']}\n...")
                                st.divider()
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    if not st.session_state.processed_files:
        st.warning("Please upload documents to start chatting!")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner('Generating responses...'):
            response_data = utils.call_ask_api(prompt, st.session_state.session_id)
        
        if response_data:
            st.session_state.messages.append({"role": "assistant", **response_data})
            st.rerun()
