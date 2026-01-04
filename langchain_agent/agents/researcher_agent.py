"""Researcher Agent - Conducts deep research on markets, competitors, and products using LangGraph."""

from langchain_agent.tools.web_search import web_search, competitor_analysis, review_analysis, market_size_research
from langchain_agent.utils.config import Config
from langchain_agent.tools.analysis import generate_chart
from langchain_agent.utils.agents import State
from langchain_agent.lib.prompts.research import SYSTEM_PROMPT as RESEARCH_SYSTEM
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from typing import Literal
llm = Config.get_chat_llm()


research_agent = create_agent(
    model=llm,
    tools=[web_search, competitor_analysis, review_analysis, generate_chart],
)




def researcher_node(state: State) -> Command[Literal["supervisor"]]:
    from langchain_agent.utils.logger import setup_logger
    logger = setup_logger(__name__, level=Config.LOG_LEVEL)
    from langchain_agent.lib.prompts.research import SYSTEM_PROMPT as RESEARCH_SYSTEM

    logger.info("Researcher node invoked")
    try:
        # Ensure the agent receives its system prompt and context
        state_with_system = state.copy()
        state_with_system["messages"] = ([{"role": "system", "content": RESEARCH_SYSTEM}] + state_with_system.get("messages", []))
        result = research_agent.invoke(state_with_system)
        logger.debug("Research agent returned result: %s", result)
        messages = [HumanMessage(content=result["messages"][-1].content, name="research")]
        return Command( 
            update={"messages": messages},
            goto="supervisor",
        )
    except Exception as e:
        logger.exception("Error running research agent: %s", e)
        return Command(update={"messages": [HumanMessage(content=f"research failed: {e}", name="research")]}, goto="supervisor")

