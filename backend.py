from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Model
ChatModel = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("Groq"),
    temperature=0.7,
    max_tokens=512,
    streaming = True
)

# Node
def chat_node(state: ChatState):
    messages = state['messages']
    response = ChatModel.invoke(messages)
    return {'messages': [response]}

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# Graph
CheckPointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=CheckPointer)

def retrieve_all_threads() :
    all_threads = set()
    for checkpoint in CheckPointer.list(None) :
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def initialize_chat_names_table() :
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_names (
            thread_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    conn.commit()
initialize_chat_names_table()

def retrieve_all_chat_names() :
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, name FROM chat_names")
    rows = cursor.fetchall()
    return {thread_id: name for thread_id, name in rows}

def save_chat_name(thread_id, name) :
    cursor = conn.cursor()
    cursor.execute("""
                   INSERT INTO chat_names (thread_id, name)
                   VALUES (?,?)
                   ON CONFLICT(thread_id) DO UPDATE SET name=excluded.name
                   """, (thread_id, name))
    conn.commit()