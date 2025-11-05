"""Utility tools for weather, search, and Hugging Face Hub stats."""
import random
from langchain_core.tools import tool
from huggingface_hub import list_models


@tool
def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


@tool
def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))
        
        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Search the web for information (mock implementation for demo)."""
    # Mock search results for demo
    mock_results = {
        "facebook": "Facebook (Meta) is a technology company. Their popular AI models on Hugging Face include LLaMA and other open-source models.",
        "meta": "Meta (formerly Facebook) develops AI models including LLaMA series.",
        "weather": "Weather information varies by location. Use a weather API for real data.",
        "default": f"Search results for: {query}. This is a mock search tool for demonstration purposes."
    }
    
    query_lower = query.lower()
    for key in mock_results:
        if key in query_lower:
            return mock_results[key]
    return mock_results["default"]


# Export tools (the decorated functions are already tools)
search_tool = web_search
weather_info_tool = get_weather_info
hub_stats_tool = get_hub_stats