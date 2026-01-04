from langchain_agent.agents.base_agent import build_research_graph
from langchain_agent.utils.config import Config
import os
from langchain_core.messages import HumanMessage
import argparse
from langchain_agent.utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Run the SaaS researcher graph")
    p.add_argument("--log-level", default=None, help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return p.parse_args()

def main():
    args = parse_args()

    # Ensure config/dirs and initialize logger
    Config.validate()
    logger = setup_logger("saas_research", level=args.log_level or Config.LOG_LEVEL)

    logger.info("Starting research graph run")

    try:
        logger.info("Building research graph...")
        research_graph = build_research_graph()

        graph_image_path = os.path.join(Config.GRAPHS_DIR, "research_graph.png")
        logger.info("Rendering graph to PNG: %s", graph_image_path)
        png_bytes = research_graph.get_graph().draw_mermaid_png()
        with open(graph_image_path, "wb") as f:
            f.write(png_bytes)

        logger.info("Graph visualization saved to: %s", os.path.abspath(graph_image_path))

    except Exception as e:
        logger.exception("Failed to build or save graph: %s", e)
        raise

    try:
        user_prompt = input("Enter the niche or industry to which you about to research: ")
        logger.info("Invoking research graph with initial prompt")
        invoke_result = research_graph.invoke({"messages": [HumanMessage(content=user_prompt)]})
        logger.info("Invocation completed")
        logger.debug("Invocation result: %s", invoke_result)

        # If graph returns any messages, log them step by step
        if isinstance(invoke_result, dict) and "messages" in invoke_result:
            for i, msg in enumerate(invoke_result["messages"]):
                content = getattr(msg, "content", str(msg))
                name = getattr(msg, "name", None)
                logger.info("Message %d from %s: %s", i, name or "unknown", content)
        else:
            logger.info("No messages returned by invocation; raw result: %s", invoke_result)

    except Exception as e:
        logger.exception("Error during graph invocation: %s", e)
        raise




if __name__ == "__main__":
    main()