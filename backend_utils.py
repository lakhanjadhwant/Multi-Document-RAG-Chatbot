# backend_utils.py

import os
import tempfile
from typing import List
import pandas as pd
from dotenv import load_dotenv

# LangChain and Community imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate

# Pinecone V3 Client Import
from pinecone import Pinecone, ServerlessSpec

# --- Constants and Configuration ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_DIMENSION = 768

# --- Service Initializations ---

def initialize_pinecone() -> Pinecone:
    """Initializes and returns the Pinecone client, creating the index if it doesn't exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            metric="cosine",
            dimension=EMBEDDING_DIMENSION,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created successfully.")
    return pc

def initialize_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initializes and returns the Google Generative AI embeddings model."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

# --- RAG Prompt Templates ---

def get_rag_prompts():
    """Returns the prompt templates for the RAG chain."""
    rag_template = """
    You are a friendly and helpful AI assistant named Document Bot.
    Your job is to provide clear, accurate, and conversational responses based **only on the provided context**.
    The context below is extracted from one or more user-provided documents.

    ---

    🔍 **Context**:
    {context}

    ---

    🙋‍♂️ **User Question**:
    {question}

    ---

    **Instructions for your response**:
    You MUST provide your response in a strict JSON format using double quotes.
    The JSON object must have three keys:
    1. "answer": This MUST be a JSON object with two keys:
        - "summary": A conversational, natural language sentence summarizing the answer. Start with a friendly tone (e.g., "Certainly!", "Yes, it looks like...", "Based on the documents...").
        - "data": The structured data (as a JSON object or list) that contains the detailed information extracted from the context. This should be null if there is no structured data to show.
    2. "reasoning": A step-by-step explanation of how you used the provided context.
    3. "source_documents": A JSON list of strings, containing the filenames of **ALL** source documents you used to construct the answer (e.g., ["resume1.pdf", "summary.docx"]).

    **JSON Response**:
    """
    
    no_context_template = """
    You are a friendly and helpful AI assistant named Document Bot. The user asked a question, but no relevant information was found in their uploaded documents.
    Please answer the user's question based on your general knowledge.

    **Instructions for your response**:
    You MUST provide your response in a strict JSON format using double quotes.
    The JSON object must have three keys:
    1. "answer": This MUST be a JSON object with two keys:
        - "summary": A conversational sentence explaining that you couldn't find the answer in the documents and are using general knowledge.
        - "data": This value must be null.
    2. "reasoning": State that no relevant information was found in the documents and the answer is based on general knowledge.
    3. "source_documents": This value must be an empty list `[]`.

    ---

    🙋‍♂️ **User Question**:
    {question}

    ---

    **JSON Response**:
    """
    rag_prompt = PromptTemplate.from_template(rag_template)
    no_context_prompt = PromptTemplate.from_template(no_context_template)
    return rag_prompt, no_context_prompt

def format_docs(docs: List[Document]) -> str:
    """Formats a list of documents into a single string for the context."""
    return "\n\n".join(
        f"Source Document: '{doc.metadata.get('filename', 'Unknown')}'\nContent: {doc.page_content}"
        for doc in docs
    )

# --- File Processing Functions ---

def get_text_from_file(file) -> str | None:
    """Extracts raw text from an uploaded file."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == ".pdf": loader = PyPDFLoader(tmp_path); text = "".join(p.page_content for p in loader.load_and_split())
        elif file_extension == ".docx": loader = Docx2txtLoader(tmp_path); text = loader.load()[0].page_content
        elif file_extension == ".txt": loader = TextLoader(tmp_path); text = loader.load()[0].page_content
        elif file_extension == ".csv": text = pd.read_csv(tmp_path).to_string()
        elif file_extension in [".xls", ".xlsx"]: text = pd.read_excel(tmp_path).to_string()
        else: return None
    finally:
        os.remove(tmp_path)
    return text

def get_text_chunks(raw_text: str) -> List[str]:
    """Splits raw text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(raw_text)

def create_and_store_embeddings(text_chunks: list, namespace: str, filename: str, pc_index, embeddings_model):
    """Generates embeddings and upserts them to Pinecone in batches."""
    doc_embeddings = embeddings_model.embed_documents(text_chunks)
    
    vectors_to_upsert = [{
        "id": f"chunk_{filename}_{i}", 
        "values": embedding, 
        "metadata": {"text": chunk, "filename": filename}
    } for i, (chunk, embedding) in enumerate(zip(text_chunks, doc_embeddings))]

    batch_size = 100
    print(f"Upserting {len(vectors_to_upsert)} vectors in batches of {batch_size}...")
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        try:
            pc_index.upsert(vectors=batch, namespace=namespace)
            print(f"Upserted batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {e}")
    print("Finished upserting all batches.")