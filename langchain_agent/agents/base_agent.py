from langgraph.graph import StateGraph
from langchain_agent.utils.agents import State
from langchain_agent.utils.agents import make_supervisor_node
from langchain_agent.utils.config import Config
from langchain_agent.agents.saas_finder_agent import saas_finder_node
from langchain_agent.agents.market_agent import market_node
from langchain_agent.agents.researcher_agent import researcher_node
from langchain_ollama import OllamaLLM
from langgraph.graph import START

llm = OllamaLLM(
    model=Config.OLLAMA_MODEL,
    base_url=Config.OLLAMA_BASE_URL
)



def build_research_graph():
    saas_finder_supervisor_node = make_supervisor_node(llm, ["saas_finder", "market", "research"])
    research_builder = StateGraph(State)
    research_builder.add_node("supervisor", saas_finder_supervisor_node)
    research_builder.add_node("saas_finder", saas_finder_node)
    research_builder.add_node("market", market_node)
    research_builder.add_node("research", researcher_node)
    research_builder.add_edge(START, "supervisor")
    research_graph = research_builder.compile()
    return research_graph