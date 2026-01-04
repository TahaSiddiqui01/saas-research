from typing import Literal, TypedDict, Any

from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from langchain_agent.lib.prompts.supervisor import SYSTEM_PROMPT, SYNTHESIS_PROMPT
from langchain_agent.utils.logger import setup_logger
from langchain_agent.utils.config import Config
from langchain_core.exceptions import OutputParserException
import json
import re

logger = setup_logger(__name__)


class State(MessagesState):
    next: str


def make_supervisor_node(llm: Any, members: list[str]) -> str:
    options = ["FINISH"] + members

    class Router(TypedDict):
        """Worker to route to next, and a short reason explaining the route."""

        # Use a simple string type for `next` to avoid complex dynamic Literal unpacking
        next: str
        reason: str

    def supervisor_node(state: State) -> Command[Any]:
        """An LLM-based router that also synthesizes final report on FINISH."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + state["messages"]

        logger.debug("Supervisor evaluating next worker")
        try:
            response = llm.with_structured_output(Router).invoke(messages)
            # Ensure we got a dict-like response from the structured parser
            if not isinstance(response, dict):
                raise ValueError("Structured parser returned non-dict response")
            logger.debug("Supervisor structured response: %s", response)
            goto = response["next"]
            reason = response.get("reason", "")
        except Exception as e:
            # Structured output parsing failed (non-JSON, formatting issue, or unexpected type).
            logger.warning("Structured parsing failed for supervisor: %s", e)
            # Get raw LLM output and attempt to salvage routing information
            raw = llm.invoke(messages)
            raw_text = None
            if isinstance(raw, dict):
                raw_text = raw.get("content") or raw.get("text")
            else:
                raw_text = getattr(raw, "content", None) or str(raw)

            logger.debug("Supervisor raw output: %s", raw_text)

            # Try to find a JSON object inside the raw text
            json_match = re.search(r"\{.*\}", raw_text or "", flags=re.S)
            parsed = None
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except Exception:
                    parsed = None

            if parsed and parsed.get("next") in options:
                goto = parsed.get("next")
                reason = parsed.get("reason", "(extracted from raw output)")
            else:
                # Fallback: search for agent names in text
                text_lower = (raw_text or "").lower()
                if "market" in text_lower:
                    goto = "market"
                    reason = "Parsed 'market' from raw output"
                elif "saas_finder" in text_lower or "saas finder" in text_lower or "idea" in text_lower:
                    goto = "saas_finder"
                    reason = "Parsed 'saas_finder' from raw output"
                elif "research" in text_lower or "competitor" in text_lower:
                    goto = "research"
                    reason = "Parsed 'research' from raw output"
                else:
                    goto = "saas_finder"
                    reason = "Defaulting to saas_finder due to parse failure"

        # Anti-loop: avoid routing to the same worker repeatedly if no new information
        try:
            recent_msgs = state.get("messages", [])[-12:]
            recent_names = []
            for m in recent_msgs:
                if isinstance(m, dict):
                    name = m.get("name")
                else:
                    name = getattr(m, "name", None)
                if name:
                    recent_names.append(str(name).lower())

            # If the selected worker appears 2 or more times recently, prefer an alternative
            if goto and goto.lower() in recent_names and recent_names.count(goto.lower()) >= 2:
                # choose the member with the least recent usage
                counts = {member: recent_names.count(member.lower()) for member in members}
                # remove current goto from consideration
                counts.pop(goto, None)
                # pick min
                if counts:
                    alternative = min(counts, key=counts.get)
                    reason = f"Avoiding repeat routing to {goto}; choosing {alternative} as it was used less recently."
                    logger.info("Loop detected for %s; selecting alternative %s", goto, alternative)
                    goto = alternative
                else:
                    # no alternative available, finish instead
                    goto = "FINISH"
                    reason = "No new information from workers; finishing to avoid loop"
        except Exception:
            # Defensive: if anything goes wrong with loop detection, proceed with original goto
            pass

        # Enforce a hard max steps limit to avoid infinite graphs
        try:
            # count worker messages (simple heuristic: messages with a 'name' field equal to a member)
            msgs = state.get("messages", [])
            step_count = 0
            for m in msgs:
                name = None
                if isinstance(m, dict):
                    name = m.get("name")
                else:
                    name = getattr(m, "name", None)
                if name and name in members:
                    step_count += 1

            if step_count >= Config.MAX_STEPS:
                logger.info("Max steps reached (%s); finishing", Config.MAX_STEPS)
                goto = "FINISH"
                reason = f"Reached maximum steps ({Config.MAX_STEPS}); finishing to avoid infinite loop"
        except Exception:
            pass

        if goto == "FINISH":
            # synthesize a final markdown report using synth prompt
            logger.info("Supervisor decided to FINISH; synthesizing final report")
            synth_messages = [
                {"role": "system", "content": SYNTHESIS_PROMPT},
            ] + state["messages"]

            synth_response = llm.invoke(synth_messages)
            # synth_response may be object-like or dict, normalize
            content = None
            if isinstance(synth_response, dict):
                content = synth_response.get("content") or synth_response.get("text")
            else:
                content = getattr(synth_response, "content", None) or str(synth_response)

            if not content:
                logger.warning("Synthesis returned empty content; falling back to raw messages")
                # As fallback, join last messages
                content = "\n\n".join([m.get("content", "") for m in state["messages"]])

            # Include the final report in the update so the graph's final output contains it
            return Command(goto=END, update={"next": END, "messages": [HumanMessage(content=content, name="final_report")]})

        # Otherwise route to the selected worker and include reason
        logger.info("Routing to %s: %s", goto, reason)
        return Command(goto=goto, update={"next": goto, "reason": reason})

    return supervisor_node