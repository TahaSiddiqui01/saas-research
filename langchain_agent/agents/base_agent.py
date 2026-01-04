from langgraph.graph import StateGraph
from langchain_agent.utils.agents import State
from langchain_agent.utils.agents import make_supervisor_node
from langchain_agent.utils.config import Config
from langchain_agent.agents.saas_finder_agent import saas_finder_node
from langchain_agent.agents.market_agent import market_node
from langchain_agent.agents.researcher_agent import researcher_node
from langgraph.graph import START
from langchain_agent.utils.logger import setup_logger
import sys

logger = setup_logger(__name__, level=Config.LOG_LEVEL)

try:
    logger.info("Initializing LLM provider=%s", Config.LLM_PROVIDER)
    llm = Config.get_chat_llm()
    logger.info("LLM initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize LLM: %s", e)
    # Re-raise to make failure explicit to caller
    raise



def build_research_graph():
    logger.info("Building research graph")
    saas_finder_supervisor_node = make_supervisor_node(llm, ["saas_finder", "market", "research"])
    research_builder = StateGraph(State)

    research_builder.add_node("supervisor", saas_finder_supervisor_node)
    logger.debug("Added node: supervisor")
    research_builder.add_node("saas_finder", saas_finder_node)
    logger.debug("Added node: saas_finder")
    research_builder.add_node("market", market_node)
    logger.debug("Added node: market")
    research_builder.add_node("research", researcher_node)
    logger.debug("Added node: research")

    research_builder.add_edge(START, "supervisor")
    logger.debug("Added start edge -> supervisor")

    research_graph = research_builder.compile()
    logger.info("Research graph built successfully")
    return research_graph