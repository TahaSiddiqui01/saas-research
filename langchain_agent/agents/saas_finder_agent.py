"""Researcher Agent - Conducts deep research on markets, competitors, and products using LangGraph."""

from langchain_agent.utils.config import Config
from langchain_agent.tools.analysis import analyze_pain_killer_vitamin, analyze_bootstrapping_feasibility, analyze_payment_willingness
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


saas_finder_agent = create_agent(
    model=llm,
    tools=[ analyze_pain_killer_vitamin, analyze_bootstrapping_feasibility, analyze_payment_willingness],
)




def saas_finder_node(state: State) -> Command[Literal["supervisor"]]:
    result = saas_finder_agent.invoke(state)
    return Command  (
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="saas_finder")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )
