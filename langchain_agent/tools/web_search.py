"""Web search tools for agents."""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for current information about companies, products, markets, or any topic. Use this to find up-to-date information."""
    try:
        results = search.run(query)
        return results
    except Exception as e:
        return f"Error performing search: {str(e)}"


@tool
def competitor_analysis(query: str) -> str:
    """Analyze competitors in a specific market or niche. Provides information about competitor products, pricing, and positioning."""
    search_query = f"competitors in {query} market SaaS products"
    try:
        results = search.run(search_query)
        return results
    except Exception as e:
        return f"Error analyzing competitors: {str(e)}"


@tool
def review_analysis(query: str) -> str:
    """Find and analyze reviews for products or services in a specific market. Helps understand user pain points and satisfaction."""
    search_query = f"{query} reviews user feedback complaints"
    try:
        results = search.run(search_query)
        return results
    except Exception as e:
        return f"Error analyzing reviews: {str(e)}"


@tool
def market_size_research(query: str) -> str:
    """Research market size, TAM (Total Addressable Market), SAM (Serviceable Addressable Market), and growth trends for a specific industry or niche."""
    search_query = f"{query} market size TAM SAM growth statistics 2024"
    try:
        results = search.run(search_query)
        return results
    except Exception as e:
        return f"Error researching market size: {str(e)}"

