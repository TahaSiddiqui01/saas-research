from langchain_agent.agents.base_agent import build_research_graph
from langchain_agent.utils.config import Config
import os

def main():
    research_graph = build_research_graph()
    print(research_graph)
    graph_image_path = os.path.join(Config.GRAPHS_DIR, "research_graph.png")
    png_bytes = research_graph.get_graph().draw_mermaid_png()
    with open(graph_image_path, "wb") as f:
        f.write(png_bytes)
    
    print(f"\nGraph visualization saved to: {os.path.abspath(graph_image_path)}")


if __name__ == "__main__":
    main()