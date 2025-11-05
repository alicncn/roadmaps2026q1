import datasets
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.memory import ConversationBufferMemory
from duckduckgo_search import DDGS
import requests
from datetime import datetime

# region Retriever 
# Load the guest dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Initialize sentence-transformer embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={
            "name": guest["name"],
            "source": "guest_database",
            "relation": guest["relation"]
        }
    )
    for guest in guest_dataset
]

# Create FAISS vector store for semantic search
vector_store = FAISS.from_documents(docs, embeddings)
# region end

# region Multiple indexes
# Additional knowledge base - Historical figures
historical_docs = [
    Document(
        page_content="Ada Lovelace (1815-1852) was an English mathematician and writer, "
                    "chiefly known for her work on Charles Babbage's proposed mechanical "
                    "general-purpose computer, the Analytical Engine. She is often regarded "
                    "as the first computer programmer.",
        metadata={"name": "Ada Lovelace", "source": "historical_database", "verified": True}
    ),
    Document(
        page_content="Charles Babbage (1791-1871) was an English polymath, mathematician, "
                    "philosopher, inventor and mechanical engineer. He originated the concept "
                    "of a digital programmable computer and is considered a 'father of the computer'.",
        metadata={"name": "Charles Babbage", "source": "historical_database", "verified": True}
    ),
    Document(
        page_content="Alan Turing (1912-1954) was an English mathematician, computer scientist, "
                    "logician, cryptanalyst, philosopher and theoretical biologist. He formalized "
                    "the concepts of algorithm and computation with the Turing machine.",
        metadata={"name": "Alan Turing", "source": "historical_database", "verified": True}
    ),
]

# Create separate vector store for historical information
historical_vector_store = FAISS.from_documents(historical_docs, embeddings)
# region end

# region Retrieval tools
def search_guest_database(query: str) -> str:
    """Searches the guest database using semantic similarity for detailed guest information."""
    results = vector_store.similarity_search_with_score(query, k=3)
    
    if results:
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                f"[Relevance: {1-score:.2f}] {doc.page_content}\nSource: {doc.metadata.get('source', 'unknown')}"
            )
        return "\n\n".join(formatted_results)
    else:
        return "No matching guest information found in the database."

def search_historical_database(query: str) -> str:
    """Searches the verified historical database for background information on notable figures."""
    results = historical_vector_store.similarity_search_with_score(query, k=2)
    
    if results:
        formatted_results = []
        for doc, score in results:
            verified = "✓ Verified" if doc.metadata.get('verified') else ""
            formatted_results.append(
                f"{doc.page_content}\n{verified} | Source: {doc.metadata.get('source', 'unknown')}"
            )
        return "\n\n".join(formatted_results)
    else:
        return "No historical information found."

def search_web(query: str) -> str:
    """Searches the web for the latest information about unfamiliar guests or topics.
    Use this when guest database and historical database don't have sufficient information."""
    
    try:
        # Initialize DuckDuckGo Search
        ddgs = DDGS()
        
        # Perform search (returns up to 5 results)
        results = ddgs.text(query, max_results=5)
        
        if not results:
            return f"No web results found for: {query}"
        
        # Format results
        formatted_results = []
        formatted_results.append(f"Web Search Results for: '{query}'")
        formatted_results.append(f"[As of {datetime.now().strftime('%Y-%m-%d %H:%M')}]\n")
        
        for idx, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            url = result.get('href', 'No URL')
            
            formatted_results.append(f"{idx}. {title}")
            formatted_results.append(f"   {body}")
            formatted_results.append(f"   Source: {url}\n")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Web search encountered an error: {str(e)}. Please try rephrasing your query."

# Create tools
guest_db_tool = Tool(
    name="guest_database_search",
    func=search_guest_database,
    description="Searches the guest database using semantic similarity. Use this to find detailed information about gala guests by name, relation, or description."
)

historical_db_tool = Tool(
    name="historical_database_search",
    func=search_historical_database,
    description="Searches verified historical records for background information on notable historical figures. Use this to provide context about famous people."
)

web_search_tool = Tool(
    name="web_search",
    func=search_web,
    description="Searches the web for latest information about unfamiliar guests or topics not found in internal databases. Use as a fallback when other sources don't have information."
)
# region end

# region Conversation Memory
class ConversationMemoryManager:
    """Manages conversation history for Alfred"""
    
    def __init__(self, max_history: int = 10):
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.max_history = max_history
    
    def add_interaction(self, human_message: str, ai_message: str):
        """Add a human-AI interaction to memory"""
        self.memory.chat_memory.add_user_message(human_message)
        self.memory.chat_memory.add_ai_message(ai_message)
    
    def get_history_summary(self) -> str:
        """Get a formatted summary of recent conversation"""
        messages = self.memory.chat_memory.messages
        if not messages:
            return "No previous conversation history."
        
        # Keep only recent messages
        recent_messages = messages[-self.max_history:]
        
        summary = "Recent Conversation History:\n"
        for msg in recent_messages:
            role = "Guest" if isinstance(msg, HumanMessage) else "Alfred"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            summary += f"- {role}: {content}\n"
        
        return summary
    
    def clear(self):
        """Clear conversation history"""
        self.memory.clear()

# Initialize memory manager
memory_manager = ConversationMemoryManager(max_history=10)
# region end

# region LLM and Agent setup
# Generate the chat interface with tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token="YOUR_HUGGINGFACEHUB_API_TOKEN",  # Replace with actual token
    temperature=0.7,
    max_new_tokens=1024,
)

chat = ChatHuggingFace(llm=llm, verbose=True)

# Bind all tools to the chat model
tools = [guest_db_tool, historical_db_tool, web_search_tool]
chat_with_tools = chat.bind_tools(tools)

# Enhanced Agent State with memory
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    conversation_history: str

def assistant(state: AgentState):
    """Enhanced assistant with conversation memory and system instructions"""
    
    # Create system message with Alfred's personality and memory
    system_message = SystemMessage(content=f"""You are Alfred, an exceptionally knowledgeable and courteous virtual butler 
assisting with a prestigious gala event. You have access to multiple information sources:

1. Guest Database: Detailed information about confirmed gala attendees
2. Historical Database: Verified information about notable historical figures
3. Web Search: Latest information from the internet (use as fallback)

{memory_manager.get_history_summary()}

Guidelines:
- Always be polite, professional, and helpful
- Use the guest database first for attendee information
- Consult historical database for context about notable figures
- Use web search only when information isn't available in other sources
- Reference previous conversation context when relevant
- Provide comprehensive, well-sourced answers
- If uncertain, acknowledge limitations gracefully

Remember to synthesize information from multiple sources when appropriate.""")
    
    # Combine system message with conversation messages
    messages_with_context = [system_message] + state["messages"]
    
    # Get response from LLM
    response = chat_with_tools.invoke(messages_with_context)
    
    return {
        "messages": [response],
        "conversation_history": memory_manager.get_history_summary()
    }
# region end

# region Build Langgraph
builder = StateGraph(AgentState)

# Define nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile the graph
alfred = builder.compile()

# interaction functions
def chat_with_alfred(user_message: str) -> str:
    """
    Enhanced function to interact with Alfred, maintaining conversation memory
    
    Args:
        user_message: The user's query
        
    Returns:
        Alfred's response
    """
    # Create message
    messages = [HumanMessage(content=user_message)]
    
    # Invoke Alfred
    response = alfred.invoke({
        "messages": messages,
        "conversation_history": memory_manager.get_history_summary()
    })
    
    # Extract response
    ai_response = response['messages'][-1].content
    
    # Update memory
    memory_manager.add_interaction(user_message, ai_response)
    
    return ai_response

# ============================================================================
# 8. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("🎩 Enhanced Alfred Agent Initialized\n")
    print("=" * 60)
    
    # # Example conversation demonstrating all features
    
    # # Query 1: Guest database search
    # print("\n📋 Query 1: Searching guest database")
    # response1 = chat_with_alfred("Tell me about our guest named 'Lady Ada Lovelace'.")
    # print(f"Alfred: {response1}\n")
    
    # # Query 2: Follow-up using memory
    # print("\n💭 Query 2: Follow-up question (uses conversation memory)")
    # response2 = chat_with_alfred("What is her relation to the event?")
    # print(f"Alfred: {response2}\n")
    
    # # Query 3: Historical database search
    # print("\n📚 Query 3: Historical context")
    # response3 = chat_with_alfred("Can you provide historical context about Ada Lovelace?")
    # print(f"Alfred: {response3}\n")
    
    # # Query 4: Unfamiliar guest (web search)
    # print("\n🌐 Query 4: Unfamiliar guest (triggers web search)")
    # response4 = chat_with_alfred("Do we have information about Grace Hopper?")
    # print(f"Alfred: {response4}\n")
    
    # # Query 5: Memory recall
    # print("\n🧠 Query 5: Testing memory")
    # response5 = chat_with_alfred("Who did we discuss earlier?")
    # print(f"Alfred: {response5}\n")
    
    # print("=" * 60)
    # print("\n✅ Demonstration complete!")
    
    # # Show conversation history
    # print("\n📝 Conversation History:")
    # print(memory_manager.get_history_summary())