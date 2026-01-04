"""Researcher Agent - Conducts deep research on markets, competitors, and products using LangGraph."""

from langchain_agent.tools.web_search import web_search, market_size_research
from langchain_agent.utils.config import Config
from langchain_agent.tools.analysis import generate_chart, generate_distribution_strategy
from langchain_agent.lib.prompts.market_analysis import SYSTEM_PROMPT as MARKET_SYSTEM
from langchain_agent.utils.agents import State
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from typing import Literal
llm = Config.get_chat_llm()


market_agent = create_agent(
    model=llm,
    tools=[web_search, generate_distribution_strategy, market_size_research, generate_chart],
)




def market_node(state: State) -> Command[Literal["supervisor"]]:
    from langchain_agent.utils.logger import setup_logger
    logger = setup_logger(__name__, level=Config.LOG_LEVEL)

    logger.info("Market node invoked")
    try:
        # Ensure the agent receives its system prompt and context (prevent missing role instructions)
        state_with_system = state.copy()
        state_with_system["messages"] = ([{"role": "system", "content": MARKET_SYSTEM}] + state_with_system.get("messages", []))
        result = market_agent.invoke(state_with_system)
        logger.debug("Market agent returned result: %s", result)
        # Validate structured JSON appended by the worker
        try:
            from langchain_agent.utils.response_utils import parse_trailing_json, get_text

            last_text = get_text(result["messages"][-1].content) if isinstance(result, dict) and result.get("messages") else get_text(result)
            parsed = parse_trailing_json(last_text)
            if not parsed:
                logger.warning("Market agent did not include structured JSON at end of response; this may lead to routing ambiguity.")
        except Exception:
            logger.exception("Failed to validate market agent structured output")
        return Command(
            update={"messages": [HumanMessage(content=result["messages"][-1].content, name="market")]},
            # We want our workers to ALWAYS "report back" to the supervisor when done
            goto="supervisor",
        )
    except Exception as e:
        logger.exception("Error running market agent: %s", e)
        return Command(update={"messages": [HumanMessage(content=f"market failed: {e}", name="market")]}, goto="supervisor")

