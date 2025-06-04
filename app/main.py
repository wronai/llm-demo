#!/usr/bin/env python3
"""
Minimalna aplikacja LLM z rozszerzonym debugowaniem i logowaniem
Streamlit + Ollama = Prosty interfejs do modeli językowych
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

import ollama
import streamlit as st
from dotenv import load_dotenv

# Konfiguracja logowania - podstawowa konfiguracja, zostanie nadpisana w __main__
logger = logging.getLogger(__name__)

# Konfiguracja środowiska
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistral:7b-instruct")
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

# Inicjalizacja klienta Ollama
client = None
try:
    client = ollama.Client(host=OLLAMA_URL)
    logger.info(f"Połączono z Ollama pod adresem: {OLLAMA_URL}")
except Exception as e:
    logger.error(f"Błąd podczas łączenia z Ollama: {str(e)}")
    raise


def log_http_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
) -> None:
    """Logowanie żądań HTTP"""
    if not DEBUG_MODE:
        return

    logger.debug(f"HTTP {method} {url}")
    if headers:
        logger.debug(f"Headers: {json.dumps(headers, indent=2, default=str)}")
    if body:
        logger.debug(
            f"Body: {json.dumps(body, indent=2, default=str) if isinstance(body, dict) else body}"
        )


def log_http_response(response: Any) -> None:
    """Logowanie odpowiedzi HTTP.

    Args:
        response: The HTTP response object to log
    """
    if not DEBUG_MODE:
        return

    if hasattr(response, "status_code"):
        logger.debug(f"Response status: {response.status_code}")
    if hasattr(response, "text"):
        try:
            logger.debug(
                f"Response body: {json.dumps(json.loads(response.text), indent=2, default=str)}"
            )
        except (json.JSONDecodeError, TypeError):
            logger.debug(f"Response body (raw): {response.text}")
        except Exception as e:
            logger.debug(f"Error processing response: {e}")
            logger.debug(f"Raw response: {response.text}")


def stream_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 500,
    system_prompt: str = "",
) -> Generator[str, None, None]:
    """Generator dla odpowiedzi strumieniowych z obsługą błędów i logowaniem"""
    start_time = time.time()
    logger.info(f"Generowanie odpowiedzi dla modelu {model}...")
    logger.debug(f"Zapytanie: {prompt}")
    logger.debug(f"Parametry: temperature={temperature}, max_tokens={max_tokens}")

    if not client:
        error_msg = "Błąd: Brak połączenia z Ollama"
        logger.error(error_msg)
        yield error_msg
        return

    try:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        logger.debug(f"Wysyłane wiadomości: {json.dumps(messages, ensure_ascii=False)}")

        # Logowanie żądania
        log_http_request(
            "POST",
            f"{OLLAMA_URL}/api/chat",
            {"Content-Type": "application/json"},
            {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )

        # Wysłanie żądania
        stream = client.chat(
            model=model,
            messages=messages,
            stream=True,
            options={"temperature": temperature, "num_predict": max_tokens},
        )

        # Przetwarzanie odpowiedzi strumieniowej
        full_response = ""
        for chunk in stream:
            if (
                chunk
                and "message" in chunk
                and "content" in chunk["message"]
                and chunk["message"]["content"]
            ):
                content = chunk["message"]["content"]
                full_response += content
                logger.debug(f"Otrzymano fragment odpowiedzi: {content}")
                yield content

        duration = time.time() - start_time
        logger.info(f"Odpowiedź wygenerowana w {duration:.2f}s")
        logger.debug(f"Pełna odpowiedź: {full_response}")

    except Exception as e:
        error_msg = f"Błąd podczas generowania odpowiedzi: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield error_msg


def init_session_state() -> None:
    """Initialize session state variables.

    Initializes the following session state variables if they don't exist:
    - messages: List to store chat messages
    - model: Default model to use
    - temperature: Temperature setting for model generation
    - max_tokens: Maximum number of tokens to generate
    - system_prompt: System prompt to guide the model's behavior
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 500
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            "Jesteś pomocnym asystentem AI. Odpowiadaj zwięźle i precyzyjnie."
        )


def get_available_models() -> List[str]:
    """Pobierz dostępne modele z Ollama."""
    if client is None:
        logger.error("Ollama client is not initialized")
        return []
    try:
        response = client.list()
        if not response or not isinstance(response, dict):
            logger.error(f"Unexpected response format: {response}")
            return []
        return [
            model["name"]
            for model in response.get("models", [])
            if isinstance(model, dict) and "name" in model
        ]
    except Exception as e:
        logger.error(f"Błąd podczas pobierania listy modeli: {str(e)}", exc_info=True)
        return []


def main() -> None:
    # Inicjalizacja stanu sesji
    init_session_state()

    # Konfiguracja interfejsu użytkownika
    st.set_page_config(
        page_title="🤖 WronAI Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Nagłówek
    st.title("🤖 WronAI Chat")
    st.markdown(f"*Powered by Ollama + {st.session_state.model}*")

    # Pobranie dostępnych modeli
    available_models = get_available_models()

    # Panel boczny z ustawieniami
    with st.sidebar:
        st.header("⚙️ Ustawienia")

        # Wybór modelu
        if not available_models:
            st.warning("Nie znaleziono dostępnych modeli.")
            if st.button("Zainstaluj domyślny model (Mistral 7B)"):
                with st.spinner("Instalowanie modelu Mistral 7B..."):
                    try:
                        if client is not None:
                            client.pull("mistral:7b-instruct")
                            st.rerun()
                        else:
                            st.error(
                                "Błąd: Połączenie z serwerem Ollama nie zostało nawiązane"
                            )
                            logger.error(
                                "Attempted to pull model but Ollama client is not initialized"
                            )
                    except Exception as e:
                        st.error(f"Nie udało się zainstalować modelu: {str(e)}")
                        logger.error(f"Błąd instalacji modelu: {str(e)}", exc_info=True)
        else:
            st.session_state.model = st.selectbox(
                "Model",
                available_models,
                index=available_models.index(st.session_state.model)
                if st.session_state.model in available_models
                else 0,
            )
            st.success(f"✅ Połączono z Ollama")
            st.info(f"Dostępne modele: {len(available_models)}")

        # Parametry modelu
        st.subheader("Parametry modelu")
        st.session_state.temperature = st.slider(
            "Temperatura",
            0.0,
            2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Kontrola losowości odpowiedzi: 0 = deterministyczne, 2 = bardzo losowe",
        )

        st.session_state.max_tokens = st.slider(
            "Maksymalna liczba tokenów",
            50,
            4000,
            value=st.session_state.max_tokens,
            step=50,
            help="Maksymalna liczba tokenów do wygenerowania",
        )

        # Prompt systemowy
        st.subheader("Prompt systemowy")
        st.session_state.system_prompt = st.text_area(
            "Instrukcje dla modelu:",
            value=st.session_state.system_prompt,
            height=150,
            help="Instrukcje określające zachowanie modelu",
        )

        # Opcje debugowania
        with st.expander("Opcje zaawansowane"):
            st.checkbox(
                "Tryb debugowania",
                value=DEBUG_MODE,
                disabled=True,
                help="Włącz DEBUG_MODE w pliku .env, aby aktywować",
            )

            if st.button("Wyczyść historię czatu"):
                st.session_state.messages = []
                st.rerun()

    # Wyświetlanie historii czatu
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Obsługa wprowadzania wiadomości
    if prompt := st.chat_input("Wpisz wiadomość..."):
        try:
            # Dodanie wiadomości użytkownika do historii
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Wyświetlenie wiadomości użytkownika
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generowanie odpowiedzi
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                # Wyświetlanie animacji ładowania
                with st.spinner("Generuję odpowiedź..."):
                    # Przetwarzanie odpowiedzi strumieniowej
                    for chunk in stream_response(
                        prompt=prompt,
                        model=st.session_state.model,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens,
                        system_prompt=st.session_state.system_prompt,
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                # Wyświetlenie pełnej odpowiedzi
                response_placeholder.markdown(full_response)

            # Dodanie odpowiedzi asystenta do historii
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

            # Zalogowanie konwersacji
            logger.info(f"Użytkownik: {prompt}")
            logger.info(f"Asystent: {full_response[:200]}...")

        except Exception as e:
            error_msg = f"Wystąpił błąd: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)

    # Panel debugowania (tylko w trybie debugowania)
    if DEBUG_MODE:
        with st.expander("🔍 Informacje diagnostyczne", expanded=False):
            st.subheader("Stan sesji")
            st.json(
                {
                    "model": st.session_state.model,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens,
                    "messages_count": len(st.session_state.messages),
                    "debug_mode": DEBUG_MODE,
                }
            )

            if st.button("Pokaż pełny log"):
                try:
                    with open("app_debug.log", "r") as f:
                        logs = f.read()
                    st.text_area("Logi aplikacji", logs, height=300)
                except Exception as e:
                    st.error(f"Nie można odczytać pliku logu: {str(e)}")

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
                if client is None:
                    st.error("Błąd: Połączenie z serwerem Ollama nie zostało nawiązane")
                    logger.error(
                        "Attempted to show model info but Ollama client is not initialized"
                    )
                else:
                    info = client.show(st.session_state.model)
                    st.json(info)
            except Exception as e:
                error_msg = f"Cannot get model info: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)


if __name__ == "__main__":
    try:
        # Load environment variables
        load_dotenv()

        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG
            if os.getenv("DEBUG_MODE", "true").lower() == "true"
            else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("app_debug.log"), logging.StreamHandler()],
        )

        logger.info("Starting WronAI Chat application")
        main()

    except Exception as e:
        error_msg = f"Critical error: {str(e)}\n{traceback.format_exc()}"
        logger.critical(error_msg)

        # Show error in the UI if possible
        try:
            st.error(
                "A critical error occurred. Please check the logs for more details."
            )
            if st.checkbox("Show error details"):
                st.code(error_msg)
        except Exception as ui_error:
            # Fallback to console if Streamlit fails
            print(f"Error displaying error in UI: {ui_error}")
            print(f"Original error: {error_msg}")
