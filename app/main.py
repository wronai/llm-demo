#!/usr/bin/env python3
"""
Minimalna aplikacja LLM w 50 linijkach!
Streamlit + Ollama = zero konfiguracji
"""

import streamlit as st
import ollama
import os
from typing import Generator

# Konfiguracja
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = "mistral:7b-instruct"

# Setup Ollama client
client = ollama.Client(host=OLLAMA_URL)

def stream_response(prompt: str) -> Generator[str, None, None]:
    """Generator dla streaming response"""
    try:
        stream = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in stream:
            if chunk['message']['content']:
                yield chunk['message']['content']
    except Exception as e:
        yield f"Error: {str(e)}"

def main():
    # UI Setup
    st.set_page_config(
        page_title="🤖 Minimal LLM Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Minimal LLM Chat")
    st.markdown("*Powered by Ollama + Mistral 7B*")
    
    # Sidebar z ustawieniami
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        try:
            models = client.list()
            available_models = [m['name'] for m in models['models']]
            st.success(f"✅ Connected to Ollama")
            st.info(f"Available models: {len(available_models)}")
        except:
            st.error("❌ Cannot connect to Ollama")
            st.stop()
        
        # Parameters
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max tokens", 50, 1000, 500, 50)
        
        # System prompt
        system_prompt = st.text_area(
            "System prompt:", 
            "You are a helpful AI assistant. Answer concisely and accurately.",
            height=100
        )
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Streaming response
            for chunk in stream_response(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("💡 Example Question"):
            example = "Explain quantum computing in simple terms"
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()
    
    with col3:
        if st.button("📊 Model Info"):
            try:
                info = client.show(MODEL_NAME)
                st.json(info)
            except:
                st.error("Cannot get model info")

if __name__ == "__main__":
    main()