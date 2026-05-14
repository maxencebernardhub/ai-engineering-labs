"""Streamlit chat interface for local AI models via Ollama."""

import time

import streamlit as st
from chat import get_stats, stream_response

from utils.helpers import check_model_available, check_ollama_running

MODELS = ["gemma4:e4b", "qwen3.5:9b", "mistral:7b"]

st.set_page_config(page_title="Local AI Chat", page_icon="🔒", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔒 Local AI Chat")
    st.caption("All inference runs on your machine — zero cloud calls.")

    model = st.selectbox("Model", MODELS)

    ollama_ok = check_ollama_running()
    model_ok = check_model_available(model) if ollama_ok else False

    if not ollama_ok:
        st.error("Ollama is not running.\n\nStart it with:\n```\nollama serve\n```")
    elif not model_ok:
        st.warning(
            f"`{model}` is not pulled locally.\n\nRun:\n```\nollama pull {model}\n```"
        )

    st.divider()
    st.subheader("Last response")

    if "stats" in st.session_state:
        col1, col2 = st.columns(2)
        col1.metric("Latency", f"{st.session_state.stats['latency_ms']:.0f} ms")
        col2.metric("Speed", f"{st.session_state.stats['tokens_per_sec']:.1f} tok/s")
    else:
        st.caption("Stats will appear after the first response.")

    st.divider()
    st.success("✅ Data stays on your machine")

# ── Main area ──────────────────────────────────────────────────────────────────
st.title("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input — disabled when Ollama or the selected model is unavailable
if prompt := st.chat_input(
    "Ask something...",
    disabled=not (ollama_ok and model_ok),
):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        token_count = 0
        t0 = time.time()

        for token in stream_response(model, st.session_state.messages):
            full_text += token
            token_count += 1  # each Ollama streaming chunk ≈ 1 token
            placeholder.markdown(full_text + "▌")

        placeholder.markdown(full_text)
        st.session_state.stats = get_stats(t0, token_count)

    st.session_state.messages.append({"role": "assistant", "content": full_text})
    st.rerun()
