# main.py

import os
import traceback
import json
from typing import List

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Import from our new utility file
import backend_utils as utils

# --- Initialize Services & Models ---
pc = utils.initialize_pinecone()
embeddings = utils.initialize_embeddings()
rag_prompt, no_context_prompt = utils.get_rag_prompts()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(title="RAG Chatbot Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "RAG Chatbot Backend is running!"}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...), session_id: str = Form(...)):
    if not files or not session_id:
        return JSONResponse(status_code=400, content={"error": "A session ID and at least one file are required."})
    
    processed_files = []
    try:
        index = pc.Index(utils.PINECONE_INDEX_NAME)
        for file in files:
            raw_text = utils.get_text_from_file(file)
            if raw_text is None: continue
            
            print(f"Processing file: {file.filename}")
            text_chunks = utils.get_text_chunks(raw_text)
            utils.create_and_store_embeddings(text_chunks, session_id, file.filename, index, embeddings)
            processed_files.append(file.filename)
        
        return JSONResponse(status_code=200, content={"message": f"Successfully processed {len(processed_files)} files.", "processed_files": processed_files})
    except Exception as e:
        print(f"ERROR during upload: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": f"An error occurred during upload: {str(e)}"})

@app.post("/ask")
async def ask_question(question: str = Form(...), session_id: str = Form(...)):
    if not question or not session_id:
        return JSONResponse(status_code=400, content={"error": "Question and session_id are required."})
    try:
        index = pc.Index(utils.PINECONE_INDEX_NAME)
        query_embedding = embeddings.embed_query(question)
        
        results = index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace=session_id)
        
        matches = results.get('matches', [])
        context_docs = [Document(page_content=m['metadata']['text'], metadata={'filename': m['metadata'].get('filename', 'Unknown')}) for m in matches]
        sources = [{"content": doc.page_content, "filename": doc.metadata.get('filename', 'Unknown')} for doc in context_docs]

        candidates = []
        for temp in [0.2, 0.7, 1.0]:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"), temperature=temp, model_kwargs={"response_format": {"type": "json_object"}})

            prompt, chain_input = (no_context_prompt, {"question": question}) if not context_docs else (rag_prompt, {"context": utils.format_docs(context_docs), "question": question})
            chain = prompt | llm | StrOutputParser()
            llm_output_str = chain.invoke(chain_input)
            
            try:
                response_data = json.loads(llm_output_str)
                candidates.append({
                    "answer": response_data.get("answer", {"summary": "Could not parse answer.", "data": None}),
                    "reasoning": response_data.get("reasoning", "Could not parse reasoning."),
                    "source_documents": response_data.get("source_documents", [])
                })
            except (json.JSONDecodeError, TypeError):
                candidates.append({
                    "answer": {"summary": "The model returned an invalid JSON response.", "data": llm_output_str},
                    "reasoning": "The model's output could not be parsed as JSON.",
                    "source_documents": []
                })

        return JSONResponse(status_code=200, content={"candidates": candidates, "sources": sources})
    except Exception as e:
        print(f"ERROR during question answering: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})
