from langchain_core.messages import AIMessageChunk, HumanMessage
from backend import chatbot, retrieve_all_threads, retrieve_all_chat_names, save_chat_name
import streamlit as st
import uuid

#------------------------------------Utility functions------------------------------------
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat() :
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])

def add_thread(thread_id) :
    if thread_id not in st.session_state["chat_threads"] :
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id) :
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def get_chat_name(thread_id) :
    messages = load_conversation(thread_id)
    for msg in messages :
        if isinstance(msg, HumanMessage) :
            content = msg.content.strip()
            return content if len(content) <= 20 else content[:20] + "..."
    return "New Chat"

#------------------------------------Session State------------------------------------
if 'message_history' not in st.session_state :
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state :
    st.session_state["thread_id"] = generate_thread_id()

if 'chat_threads' not in st.session_state :
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

if "chat_names" not in st.session_state :
    st.session_state["chat_names"] = retrieve_all_chat_names()


#------------------------------------Sidebar UI------------------------------------

st.sidebar.title("ChatBot")
st.sidebar.button("New Chat", on_click=reset_chat)

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1] :
    display_name = st.session_state["chat_names"].get(thread_id, "New Chat")
    if st.sidebar.button(display_name, key = thread_id) :
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages



#------------------------------------Main UI------------------------------------
message_history = st.session_state["message_history"]

for message in message_history:
    with st.chat_message(message["role"]) :
        st.text(message["content"])

user_input = st.chat_input("Type here ....")

if user_input :

    st.session_state["message_history"].append({"role" : "user", "content" : user_input})
    with st.chat_message("user") :
        st.text(user_input)

    CONFIG = {'configurable' : {'thread_id' : st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        def stream_response():
            full_response = ""
            for chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"   # streams token by token
            ):
                if isinstance(chunk, AIMessageChunk):
                    full_response += chunk.content
                    yield chunk.content

        ai_message = st.write_stream(stream_response())
    st.session_state["message_history"].append({"role" : "assistant", "content" : ai_message})

    if len(st.session_state["message_history"]) == 2 :
        thread_id = st.session_state["thread_id"]
        first_message = st.session_state["message_history"][0]["content"]
        name = first_message.strip()[:30] + ("..." if len(first_message) > 30 else "")
        st.session_state["chat_names"][thread_id] = name
        save_chat_name(thread_id, name)
        st.rerun()