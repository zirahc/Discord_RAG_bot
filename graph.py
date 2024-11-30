import sqlite3
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
from chat import ChatUtils
from langgraph.checkpoint.sqlite import SqliteSaver

chat_utils = ChatUtils()
class State(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str

history_aware_retriever = chat_utils.create_vectorstore_retriever()
conversational_rag_chain = chat_utils.initialize_knowledge_graph(history_aware_retriever)

def rag_chatbot(state: State):
    chat_history = state["messages"]
    user_message = state["messages"][-1].content
    rag_response = ""
    for chunk in conversational_rag_chain.stream(
            {"input": user_message, "chat_history": chat_history},
            config=RunnableConfig(callbacks=[], configurable={"session_id": state["session_id"]}),
    ):
        if isinstance(chunk, dict) and "answer" in chunk:
            rag_response += chunk["answer"]
        elif isinstance(chunk, str):
            rag_response += chunk

    return {"messages": [AIMessage(content=rag_response)], "session_id": state["session_id"]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", rag_chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
sqlite_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)
graph = graph_builder.compile(
    checkpointer = memory
)

async def generate_response(message, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    events = graph.stream(
        {
            "messages": (
                "user",
                message
            ),
            "session_id": thread_id
        },
        config,
        stream_mode="values",
    )
    answer = ""
    for event in events:
        if "messages" in event:
            answer = event["messages"][-1].content
    
    if len(answer) >= 2000:
        answer = answer[:1995] + "..."

    return answer

async def generate_intro():
    return await chat_utils.generate_intro_message()
