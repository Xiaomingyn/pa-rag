import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="DeepSeek RAG Bot", layout="wide")

# Sidebar
st.sidebar.title("Controls")
if st.sidebar.button("🔄 Re-index Documents"):
    resp = requests.post(f"{BACKEND_URL}/ingest/ingest")
    st.sidebar.success(f"Ingestion started: {resp.json()}")

# Chat interface
st.title("💬 Chat with your Documents (DeepSeek-R1)")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.markdown("**Sources:** " + ", ".join(message["sources"]))

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("DeepSeek-R1 is reasoning..."):
            resp = requests.post(f"{BACKEND_URL}/query", json={"question": prompt})
            result = resp.json()
            st.markdown(result["answer"])
            st.markdown("**Sources:** " + ", ".join(result["sources"]))
        st.session_state.messages.append({
            "role": "assistant", "content": result["answer"],
            "sources": result["sources"]
        })
