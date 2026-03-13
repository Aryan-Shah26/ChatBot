from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from backend import chatbot, retrieve_all_threads, retrieve_all_chat_names, save_chat_name
import streamlit as st
import uuid

# ─────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# Tool name → emoji + label
TOOL_LABELS = {
    "search_web":       ("🌐", "Web Search"),
    "search_wikipedia": ("📖", "Wikipedia"),
    "get_weather":      ("🌤️", "Weather"),
}

def tool_badge(tool_name: str) -> str:
    emoji, label = TOOL_LABELS.get(tool_name, ("🔧", tool_name))
    return f"{emoji} {label}"

# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

if "chat_names" not in st.session_state:
    st.session_state["chat_names"] = retrieve_all_chat_names()

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

st.sidebar.title("ChatBot")
st.sidebar.button("New Chat", on_click=reset_chat)
st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    display_name = st.session_state["chat_names"].get(thread_id, "New Chat")
    if st.sidebar.button(display_name, key=thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content, "tools_used": []})
            elif isinstance(msg, ToolMessage):
                # Attach tool usage to the last assistant message
                if temp_messages and temp_messages[-1]["role"] == "assistant":
                    temp_messages[-1]["tools_used"].append(msg.name)
            else:
                # AIMessage — skip pure tool-call messages (no visible content)
                if msg.content:
                    temp_messages.append({"role": "assistant", "content": msg.content, "tools_used": []})

        st.session_state["message_history"] = temp_messages

# ─────────────────────────────────────────
# Main Chat UI
# ─────────────────────────────────────────

def render_tool_badges(tools_used: list[str]):
    """Render a subtle 'Tools used' line above the assistant message."""
    if not tools_used:
        return
    unique_tools = list(dict.fromkeys(tools_used))   # deduplicate, preserve order
    badges = "  ".join(f"`{tool_badge(t)}`" for t in unique_tools)
    st.caption(f"Tools used: {badges}")

message_history = st.session_state["message_history"]

for message in message_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_tool_badges(message.get("tools_used", []))
        st.text(message["content"])

# ─────────────────────────────────────────
# Handle new user input
# ─────────────────────────────────────────

user_input = st.chat_input("Type here ....")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input, "tools_used": []}
    )
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        tools_used_this_turn: list[str] = []
        tool_placeholder = st.empty()   # live "calling tool…" indicator

        def stream_response():
            full_response = ""
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # ── Detect tool calls being made ──
                if isinstance(chunk, AIMessageChunk) and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        name = tc.get("name", "")
                        if name and name not in tools_used_this_turn:
                            tools_used_this_turn.append(name)
                            badge_line = "  ".join(
                                f"`{tool_badge(t)}`" for t in tools_used_this_turn
                            )
                            tool_placeholder.caption(f"Tools used: {badge_line}")

                # ── Stream the final text response ──
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            return full_response

        ai_message = st.write_stream(stream_response())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message, "tools_used": tools_used_this_turn}
    )

    # Save chat name on first exchange
    if len(st.session_state["message_history"]) == 2:
        thread_id = st.session_state["thread_id"]
        first_message = st.session_state["message_history"][0]["content"]
        name = first_message.strip()[:30] + ("..." if len(first_message) > 30 else "")
        st.session_state["chat_names"][thread_id] = name
        save_chat_name(thread_id, name)
        st.rerun()