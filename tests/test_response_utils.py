from langchain_agent.utils.response_utils import parse_trailing_json


def test_parse_trailing_json_present():
    text = "Here is the analysis.\n{\"summary\": \"Top idea\", \"findings\": [\"a\", \"b\"], \"next\": \"market\", \"confidence\": \"medium\"}"
    parsed = parse_trailing_json(text)
    assert parsed is not None
    assert parsed.get("summary") == "Top idea"


def test_parse_trailing_json_missing():
    text = "No JSON here, just a sentence."
    parsed = parse_trailing_json(text)
    assert parsed is None
