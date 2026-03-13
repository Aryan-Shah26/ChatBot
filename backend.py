from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from duckduckgo_search import DDGS
import wikipedia
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

@tool 
def search_web(query: str) -> str :
    """
    Search the web using DuckDuckGo. Use this tool for current events, news, or general information requiring up-to-date information.
    """

    with DDGS() as ddgs :
        results = list(ddgs.text(query, max_results=4))
    if not results :
        return "No results found."
    
    return "\n\n".join(f"**{r['title']}**\n{r['body']}\nSources: {r['href']}" for r in results)


@tool
def search_wikipedia(query: str) -> str :
    """
    Search wikipedia for factual, encyclopedic information about people, places, events or concepts. Use this tool for well-established information that is unlikely to change frequently.
    """

    try :
        summary = wikipedia.summary(query, sentences=6, auto_suggest=True)
        return summary
    
    except wikipedia.DisambiguationError as e :
        try :
            summary = wikipedia.summary(e.options[0], sentences=6, auto_suggest=True)
            return summary
        except Exception :
            return f"Ambiguous query. Possible options include: {', '.join(e.options[:5])}"
    except wikipedia.PageError :
        return f"No Wikipedia page found for '{query}'."
    except Exception as e :
        return f"An error occurred while searching Wikipedia: {str(e)}"

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#tools
tools = [search_web, search_wikipedia]

# Model
ChatModel = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("Groq"),
    temperature=0.7,
    max_tokens=512,
    streaming = True
).bind_tools(tools)

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
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
#graph.add_edge("chat_node", END)

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