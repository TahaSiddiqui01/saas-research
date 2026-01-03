"""Researcher Agent - Conducts deep research on markets, competitors, and products using LangGraph."""

from langchain_agent.tools.web_search import web_search, competitor_analysis, review_analysis, market_size_research
from langchain_agent.utils.config import Config
from langchain_agent.tools.analysis import generate_chart
from langchain_agent.utils.agents import State
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from typing import Literal
llm = ChatOllama(
    model=Config.OLLAMA_MODEL,
    base_url=Config.OLLAMA_BASE_URL
)


ressearch_agent = create_agent(
    model=llm,
    tools=[web_search, competitor_analysis, review_analysis, generate_chart],
)




def researcher_node(state: State) -> Command[Literal["supervisor"]]:
    result = ressearch_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="research")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

