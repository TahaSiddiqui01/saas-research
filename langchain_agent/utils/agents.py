from typing import Literal, TypedDict, Any

from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_agent.lib.prompts.supervisor import SYSTEM_PROMPT, SYNTHESIS_PROMPT
from langchain_agent.utils.logger import setup_logger
from langchain_agent.utils.config import Config
from langchain_core.exceptions import OutputParserException
import json
import re

logger = setup_logger(__name__)


class State(MessagesState):
    next: str

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            synth_messages = [
                {"role": "system", "content": SYNTHESIS_PROMPT},
            ] + state["messages"]
            synth_response = llm.invoke(synth_messages)
            return Command(
                update={"messages": [HumanMessage(content=synth_response, name="final_report")]},
                goto=END,
            )

        return Command(goto=goto, update={"next": goto})

    return supervisor_node