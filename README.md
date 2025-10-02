# 📄 Multi-Document RAG Chatbot

This project is a complete Retrieval-Augmented Generation (RAG) application that allows you to chat with your own documents. It features a user-friendly web interface built with Streamlit and a robust backend powered by FastAPI.

The application ingests multiple documents (PDFs, DOCX, TXT, CSV, XLSX), processes and stores them as vector embeddings in a Pinecone database, and uses a Large Language Model (LLM) from Groq to answer questions based on the document contents.

## ✨ Features

* **Multi-File Upload**: Supports a variety of document formats including `.pdf`, `.docx`, `.txt`, `.csv`, and `.xlsx`.
* **Decoupled Architecture**: A Streamlit frontend for the user interface and a FastAPI backend for processing, ensuring scalability and separation of concerns.
* **Advanced RAG Pipeline**: Uses LangChain to orchestrate text splitting, embedding generation, and contextual prompting.
* **High-Performance Components**:
    * **Embeddings**: `Google Gemini` for state-of-the-art text embeddings.
    * **Vector Store**: `Pinecone` for efficient similarity search.
    * **LLM**: `Groq` for incredibly fast and high-quality language generation.
* **Multi-Response Generation**: The backend generates multiple candidate answers with different creativity levels (temperatures) for the user to compare.
* **Source & Reasoning**: Each answer is accompanied by the reasoning behind it and citations from the source documents, providing transparency and trust.

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: FastAPI
* **LLM**: Groq (via LangChain)
* **Embeddings**: Google Gemini (`text-embedding-004`)
* **Vector Database**: Pinecone
* **Core Logic**: LangChain

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

* Python 3.8 or higher
* Git

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

You will need API keys from Google, Pinecone, and Groq. Create a file named `.env` in the root of your project directory and add the following keys:

```env
# backend_utils.py uses these
GOOGLE_API_KEY="your-google-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="your-pinecone-index-name" # e.g., "rag-chatbot"

# main.py uses these
GROQ_API_KEY="your-groq-api-key"
GROQ_MODEL_NAME="llama-3.1-8b-instant" # Recommended model, but you can change it
```

### 6. Running the Application

You need to run the backend and frontend in two separate terminal windows.

**Terminal 1: Start the FastAPI Backend**

The backend server will run on `http://127.0.0.1:8000`.

```bash
uvicorn main:app --reload
```

**Terminal 2: Start the Streamlit Frontend**

The frontend application will be accessible at `http://localhost:8501`.

```bash
streamlit run app.py
```

##  kullanım

1.  Open your web browser and navigate to `http://localhost:8501`.
2.  Use the sidebar to upload one or more documents.
3.  Click the **"Process Files"** button. The application will embed the documents and store them in Pinecone.
4.  Once processing is complete, you can start asking questions in the chat input box at the bottom of the page.
5.  The chatbot will provide several answers in tabs, each with its reasoning and source excerpts.

## 📁 Project Structure

```
.
├── app.py              # Streamlit frontend application logic
├── main.py             # FastAPI backend API endpoints
├── backend_utils.py    # Helper functions for the backend (file processing, embeddings, Pinecone)
├── frontend_utils.py   # Helper functions for the frontend (API calls, UI formatting)
├── requirements.txt    # List of Python dependencies
├── .env                # File for storing API keys and environment variables (you need to create this)
└── README.md           # This file
```
