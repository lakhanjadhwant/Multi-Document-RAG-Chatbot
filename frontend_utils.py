# frontend_utils.py

import streamlit as st
import requests
import json
import uuid

# --- Configuration ---
FASTAPI_URL = "https://multi-document-rag-chatbot.onrender.com/"

# --- Session State Management ---
def init_session_state():
    """Initializes the session state variables if they don't exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload one or more documents to start a new chat session."}]
        st.session_state.processed_files = []

# --- API Communication ---
def handle_api_error(response):
    """Displays an error message in the Streamlit app based on the API response."""
    try:
        error_details = response.json().get('error', 'Unknown server error')
        st.error(f"Error (Status {response.status_code}): {error_details}")
    except json.JSONDecodeError:
        st.error(f"Error (Status {response.status_code}): Server returned an invalid response.")

def call_upload_api(files):
    """Sends files to the backend /upload endpoint."""
    files_payload = [('files', (file.name, file, file.type)) for file in files]
    new_session_id = str(uuid.uuid4())
    data_payload = {'session_id': new_session_id}
    try:
        response = requests.post(f"{FASTAPI_URL}/upload", files=files_payload, data=data_payload)
        if response.status_code == 200:
            return response.json(), new_session_id
        else:
            handle_api_error(response)
            return None, st.session_state.session_id
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: Could not connect to backend. Details: {e}")
        return None, st.session_state.session_id

def call_ask_api(question, session_id):
    """Sends a question to the backend /ask endpoint."""
    try:
        payload = {'question': question, 'session_id': session_id}
        response = requests.post(f"{FASTAPI_URL}/ask", data=payload)
        if response.status_code == 200:
            return response.json()
        else:
            handle_api_error(response)
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: Could not connect to backend. Details: {e}")
        return None

# --- UI Formatting ---
def format_structured_data(data, level=0):
    """Recursively formats dictionaries and lists into markdown for display."""
    if data is None: return ""
    markdown_output = ""
    header_prefix = "#" * (level + 3)

    if isinstance(data, dict):
        for key, value in data.items():
            if 'contribution' not in key.lower() and 'timeframe' not in key.lower():
                 markdown_output += f"{header_prefix} {key.replace('_', ' ').title()}\n"
            markdown_output += format_structured_data(value, level + 1)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                markdown_output += format_structured_data(item, level)
            else:
                markdown_output += f"- {item}\n"
        markdown_output += "\n"
    else:
        markdown_output += f"{data}\n\n"
    return markdown_output

