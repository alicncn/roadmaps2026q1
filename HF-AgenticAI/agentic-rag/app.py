"""Main application: Alfred the AI assistant with LangGraph agent."""
import os
import json
import random
import re
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

# --- State Definition ---
class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list[AnyMessage], add_messages]


def create_alfred_agent(api_token: str):
    """Create and compile the Alfred agent graph."""
    # Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=api_token,
        max_new_tokens=1024,
        temperature=0.1,
    )
    chat = ChatHuggingFace(llm=llm, verbose=True)
    
    # Define the tools
    tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
    tool_node = ToolNode(tools)

    # --- UPDATED State-Aware Assistant Node ---
    def assistant(state: AgentState):
        """
        Checks the last message in the state and decides whether to call a tool
        or to generate a final response based on the tool's output.
        """
        last_message = state["messages"][-1]

        # --- BEHAVIOR 1: If the last message is from a tool, summarize the output ---
        if isinstance(last_message, ToolMessage):
            # Find the original user question
            user_question = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_question = msg.content
                    break
            
            # Create a simple prompt for summarization
            summarization_prompt = ChatPromptTemplate.from_template(
                """You are a helpful assistant. The user asked the following question:
                "{user_question}"

                To answer this, a tool was used and returned the following information:
                "{tool_output}"

                Please provide a clear and direct answer to the user based on the information provided by the tool.
                """
            )
            chain = summarization_prompt | chat
            response = chain.invoke({
                "user_question": user_question,
                "tool_output": last_message.content
            })
            return {"messages": [response]}

        # --- BEHAVIOR 2: If the last message is from a user, decide if a tool is needed ---
        tool_names = ", ".join([tool.name for tool in tools])
        prompt_template = f"""You are a helpful assistant. You have access to the following tools: {tool_names}.

To use a tool, you must respond with ONLY the JSON object representing the tool call.
For example: {'{{"tool": "guest_info_retriever", "args": {{"query": "Lady Ada Lovelace"}}}}'}

If you don't need to use a tool, just provide a natural language response.
User's request: {{input}}
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | chat
        response = chain.invoke({"input": state["messages"][-1].content})
        
        # Use regex to find a JSON object within the model's output
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                tool_call_request = json.loads(json_str)
                if "tool" in tool_call_request and "args" in tool_call_request:
                    tool_call_id = f"call_{random.randint(1000, 9999)}"
                    tool_calls = [{
                        "name": tool_call_request["tool"],
                        "args": tool_call_request["args"],
                        "id": tool_call_id,
                    }]
                    return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON was found, treat as a regular text response
        return {"messages": [response]}

    # Graph Logic (no changes needed)
    def after_assistant(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"

    # Graph Construction (no changes needed)
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", after_assistant)
    builder.add_edge("tools", "assistant")
    return builder.compile()


if __name__ == "__main__":
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not HUGGINGFACEHUB_API_TOKEN:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")
    
    alfred = create_alfred_agent(HUGGINGFACEHUB_API_TOKEN)
    messages = [HumanMessage(content="How is the weather in Paris tonight?")]
    response = alfred.invoke({"messages": messages})
    
    print("🎩 Alfred's Response:")
    # The final message should now be a clean, summarized answer
    print(response['messages'][-1].content)