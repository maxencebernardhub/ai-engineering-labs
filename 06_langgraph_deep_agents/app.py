"""Shared Streamlit chat app — LangGraph vs Deep Agents commercial assistant."""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Commercial Assistant",
    page_icon="💼",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

AGENT_TYPES = ["LangGraph", "Deep Agents"]

MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-haiku-4-5-20251001",
    ],
    "openai": [
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
    ],
    "google": [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_agent(agent_type: str, provider: str, model: str):
    if agent_type == "LangGraph":
        from langgraph.agent.agent import create_agent
    else:
        from deep_agents.agent.agent import create_agent
    return create_agent(provider=provider, model=model)


def _extract_interrupt_info(agent, config: dict) -> dict | None:
    """Return normalised interrupt info if the graph is paused, else None."""
    try:
        state = agent.get_state(config)
        if not state.next:
            return None
        interrupts = state.tasks[0].interrupts if state.tasks else []
        if not interrupts:
            return None
        val = interrupts[0].value
        # Deep Agents format: {"action_requests": [{"name": ..., "args": ...}]}
        if isinstance(val, dict) and "action_requests" in val:
            req = val["action_requests"][0]
            return {
                "tool_name": req["name"],
                "tool_args": req["args"],
                "message": (
                    f"**{req['name']}** called with arguments:\n\n"
                    f"```\n{req['args']}\n```"
                ),
                "format": "deep_agents",
            }
        # LangGraph format: {"tool_name": ..., "tool_args": ..., "message": ...}
        return {**val, "format": "langgraph"}
    except Exception:
        return None


def _build_resume(agent_type: str, decision: str, feedback: str = "") -> Command:
    if agent_type == "LangGraph":
        if decision == "approve":
            return Command(resume={"decision": "approve"})
        return Command(
            resume={"decision": "cancel", "feedback": feedback or "Cancelled."}
        )
    # Deep Agents
    if decision == "approve":
        return Command(resume={"decisions": [{"type": "approve"}]})
    return Command(resume={"decisions": [{"type": "reject"}]})


def _last_ai_text(agent, config: dict) -> str | None:
    try:
        state = agent.get_state(config)
        for msg in reversed(state.values.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                if isinstance(content, str):
                    return content
                # Google Gemini returns a list of blocks: [{"type": "text", "text": "..."}]
                if isinstance(content, list):
                    texts = [
                        block["text"]
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        and block.get("text")
                    ]
                    if texts:
                        return " ".join(texts)
    except Exception:
        pass
    return None


def _resume_and_refresh(agent, config: dict, cmd: Command) -> None:
    with st.spinner("Resuming…"):
        agent.invoke(cmd, config)
    response = _last_ai_text(agent, config)
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.hitl_pending = None
    st.rerun()


# ── Session state ─────────────────────────────────────────────────────────────

for key, default in {
    "messages": [],
    "thread_id": str(uuid.uuid4()),
    "agent": None,
    "agent_key": None,
    "hitl_pending": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    agent_type = st.selectbox("Agent", AGENT_TYPES)
    provider = st.selectbox("Provider", list(MODELS.keys()))
    model = st.selectbox("Model", MODELS[provider])

    st.divider()
    if st.button("🔄 New conversation", use_container_width=True):
        for k in ("messages", "agent", "agent_key", "hitl_pending"):
            st.session_state[k] = [] if k == "messages" else None
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption("**LangGraph** — explicit StateGraph, manual `interrupt()` nodes")
    st.caption("**Deep Agents** — declarative `interrupt_on`, ~3× less code")

# ── Agent init ────────────────────────────────────────────────────────────────

agent_key = f"{agent_type}|{provider}|{model}"
if st.session_state.agent_key != agent_key:
    with st.spinner(f"Loading {agent_type} agent…"):
        try:
            st.session_state.agent = _load_agent(agent_type, provider, model)
            st.session_state.agent_key = agent_key
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.hitl_pending = None
        except Exception as exc:
            st.error(f"Failed to load agent: {exc}")
            st.stop()

agent = st.session_state.agent
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── Header ────────────────────────────────────────────────────────────────────

st.title("💼 Commercial Assistant")
st.caption(f"Agent: **{agent_type}** · {provider} / {model}")

# ── Message history ───────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── HITL panel ────────────────────────────────────────────────────────────────

if st.session_state.hitl_pending:
    info = st.session_state.hitl_pending

    with st.container(border=True):
        st.warning("⚠️ Action requires your review before execution")
        st.markdown(info.get("message", str(info)))

        col_approve, col_cancel = st.columns(2)
        with col_approve:
            if st.button("✅ Approve", use_container_width=True, key="btn_approve"):
                _resume_and_refresh(agent, config, _build_resume(agent_type, "approve"))
        with col_cancel:
            if st.button("❌ Cancel", use_container_width=True, key="btn_cancel"):
                _resume_and_refresh(agent, config, _build_resume(agent_type, "cancel"))

        with st.form("adjust_form", clear_on_submit=True):
            feedback = st.text_input("Or request adjustments and submit:")
            if st.form_submit_button("✏️ Adjust"):
                _resume_and_refresh(
                    agent, config, _build_resume(agent_type, "adjust", feedback)
                )

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input(
    "Ask about your leads…",
    disabled=bool(st.session_state.hitl_pending),
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                list(
                    agent.stream(
                        {"messages": [HumanMessage(content=prompt)]},
                        config,
                        stream_mode="values",
                    )
                )
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.stop()

        interrupt_info = _extract_interrupt_info(agent, config)
        if interrupt_info:
            st.session_state.hitl_pending = interrupt_info
            st.info("⏸️ Agent paused — review the action above before continuing.")
        else:
            response = _last_ai_text(agent, config)
            if response:
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    st.rerun()
