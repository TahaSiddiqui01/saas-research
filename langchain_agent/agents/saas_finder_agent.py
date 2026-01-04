"""Researcher Agent - Conducts deep research on markets, competitors, and products using LangGraph."""

from langchain_agent.utils.config import Config
from langchain_agent.tools.analysis import analyze_pain_killer_vitamin, analyze_bootstrapping_feasibility, analyze_payment_willingness
from langchain_agent.tools.web_search import web_search
from langchain_agent.utils.agents import State
from langchain_agent.lib.prompts.saas_finder import SYSTEM_PROMPT as SAAS_FINDER_SYSTEM
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from typing import Literal
llm = Config.get_chat_llm()


saas_finder_agent = create_agent(
    model=llm,
    tools=[ analyze_pain_killer_vitamin, analyze_bootstrapping_feasibility, analyze_payment_willingness, web_search],
)




def saas_finder_node(state: State) -> Command[Literal["supervisor"]]:
    from langchain_agent.utils.logger import setup_logger
    logger = setup_logger(__name__, level=Config.LOG_LEVEL)

    logger.info("SaaS finder node invoked")
    try:
        # Ensure the agent receives its system prompt and context
        state_with_system = state.copy()
        state_with_system["messages"] = ([{"role": "system", "content": SAAS_FINDER_SYSTEM}] + state_with_system.get("messages", []))
        result = saas_finder_agent.invoke(state_with_system)
        logger.debug("SaaS finder agent returned result: %s", result)
        messages = [HumanMessage(content=result["messages"][-1].content, name="saas_finder")]
        return Command(
            update={"messages": messages},
            goto="supervisor",
        )
    except Exception as e:
        logger.exception("Error running saas_finder agent: %s", e)
        return Command(update={"messages": [HumanMessage(content=f"saas_finder failed: {e}", name="saas_finder")]}, goto="supervisor")
