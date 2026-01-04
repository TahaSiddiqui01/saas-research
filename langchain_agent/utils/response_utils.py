"""Helpers to normalize LLM/agent responses into a predictable messages list."""
from typing import Any, List, Dict, Optional
from langchain_core.messages import HumanMessage


def extract_messages(result: Any) -> List[Dict[str, str]]:
    """Normalize various result shapes to a list of dicts with 'content' and optional 'name'.

    Accepts:
    - dicts from runnables: {'messages': [...] } or {'content': '...'}
    - objects with attributes .messages or .content
    - plain strings

    Returns an empty list when nothing parsable is found.
    """
    messages: List[Dict[str, str]] = []

    # dict-like results
    if isinstance(result, dict):
        if "messages" in result and isinstance(result["messages"], list):
            for m in result["messages"]:
                # messages might be HumanMessage objects or dicts
                if hasattr(m, "content"):
                    messages.append({"content": getattr(m, "content", ""), "name": getattr(m, "name", None)})
                elif isinstance(m, dict):
                    messages.append({"content": m.get("content", ""), "name": m.get("name")})
                else:
                    messages.append({"content": str(m), "name": None})
            return messages

        # dict with single content
        if "content" in result:
            messages.append({"content": result.get("content", ""), "name": result.get("name")})
            return messages

    # object with .messages attribute
    if hasattr(result, "messages") and isinstance(getattr(result, "messages"), list):
        for m in getattr(result, "messages"):
            if hasattr(m, "content"):
                messages.append({"content": getattr(m, "content", ""), "name": getattr(m, "name", None)})
            elif isinstance(m, dict):
                messages.append({"content": m.get("content", ""), "name": m.get("name")})
            else:
                messages.append({"content": str(m), "name": None})
        return messages

    # object with .content
    if hasattr(result, "content"):
        messages.append({"content": getattr(result, "content", ""), "name": getattr(result, "name", None)})
        return messages

    # plain string
    if isinstance(result, str):
        messages.append({"content": result, "name": None})
        return messages

    # fallback: nothing parsable
    return []


def get_text(result: Any) -> str:
    """Extract a text string from various result shapes.

    - If result is a dict with 'content' or 'text', return that.
    - If result has attribute .content, return it.
    - If result is a string, return it.
    - Otherwise return str(result) or empty string on None.
    """
    if result is None:
        return ""

    if isinstance(result, dict):
        return result.get("content") or result.get("text") or ""

    if hasattr(result, "content"):
        return getattr(result, "content", "")

    if hasattr(result, "text"):
        return getattr(result, "text", "")

    if isinstance(result, str):
        return result

    return str(result)


def parse_trailing_json(text: str) -> Optional[Dict]:
    """Attempt to parse a trailing JSON object from a block of text.

    Returns the parsed dict if successful, otherwise None.
    """
    import re
    import json

    if not text:
        return None

    # Find the last JSON object in the text (greedy match to the last closing brace)
    match = re.search(r"(\{.*\})\s*$", text, flags=re.S)
    if not match:
        return None

    candidate = match.group(1)
    try:
        return json.loads(candidate)
    except Exception:
        return None
