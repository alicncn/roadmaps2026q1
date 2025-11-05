"""Guest information retrieval using BM25."""
import datasets
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool


def _load_guest_documents():
    """Load and convert guest dataset into Document objects."""
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
    
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]
    return docs


def _retrieve_guest_info(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    docs = _load_guest_documents()
    bm25_retriever = BM25Retriever.from_documents(docs)
    results = bm25_retriever.invoke(query)
    
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    return "No matching guest information found."


# Export the guest info tool
guest_info_tool = Tool(
    name="guest_info_retriever",
    func=_retrieve_guest_info,
    description="Retrieves detailed information about gala guests based on their name or relation."
)