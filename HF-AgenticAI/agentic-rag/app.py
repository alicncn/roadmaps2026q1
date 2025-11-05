"""Main application: Alfred the AI assistant with LangGraph agent."""
import os
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import search_tool, weather_info_tool, hub_stats_tool  # ✅ Fixed import
from retriever import guest_info_tool


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list[AnyMessage], add_messages]


def create_alfred_agent(api_token: str):
    """Create and compile the Alfred agent graph."""
    # Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        huggingfacehub_api_token=api_token,
    )
    chat = ChatHuggingFace(llm=llm, verbose=True)
    
    # Bind tools to chat model
    tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
    chat_with_tools = chat.bind_tools(tools)
    
    # Define assistant node
    def assistant(state: AgentState):
        return {"messages": [chat_with_tools.invoke(state["messages"])]}
    
    # Build the graph
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    # Define edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    
    return builder.compile()


if __name__ == "__main__":
    # Get API token from environment variable
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not HUGGINGFACEHUB_API_TOKEN:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")
    
    # Create the agent
    alfred = create_alfred_agent(HUGGINGFACEHUB_API_TOKEN)
    
    # Example usage
    messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
    response = alfred.invoke({"messages": messages})
    
    print("🎩 Alfred's Response:")
    print(response['messages'][-1].content)