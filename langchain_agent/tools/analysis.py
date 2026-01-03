"""Analysis tools for agents."""

from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langchain_agent.tools.chart_generator import ChartGenerator
from langchain_agent.utils.config import Config

llm = OllamaLLM(model=Config.OLLAMA_MODEL, base_url=Config.OLLAMA_BASE_URL)
chart_generator = ChartGenerator(Config.CHARTS_DIR)

@tool
def analyze_pain_killer_vitamin(description: str) -> str:
    """This tool returns the analysis of whether a product is a pain killer or a vitamin by taking the product idea as input."""
    analysis = f"""
    Analyze the product idea given in the description and return the analysis by checking the following indicators:
    ```
    Product idea:
    {description}
    ```
    
    Pain Killer Indicators:
    - Solves urgent, critical problems
    - Users actively seek solutions
    - High willingness to pay
    - Replaces existing expensive/time-consuming solutions
    
    Vitamin Indicators:
    - Nice-to-have features
    - Users may not actively seek it
    - Lower urgency
    - Enhancement rather than necessity
    
    Return the analysis in the following format:
    ```
    Analysis:
    - Pain Killer: [Yes/No]
    - Vitamin: [Yes/No]
    ```
    """
    response = llm.invoke(analysis)
    return response.content

@tool
def analyze_bootstrapping_feasibility(description: str) -> str:
    """This tool returns the analysis of whether a product can be bootstrapped (built without external funding) by taking the product idea as input."""
    analysis = f"""
    Analyze the product idea given in the description and return the analysis by checking the following indicators:
    ```
    Product idea:
    {description}
    ```
    
    Bootstrapping Feasibility Analysis by checking the following indicators:
    ```
    - Development complexity and time
    - Initial capital requirements
    - Time to first revenue
    - Team size needed
    - Infrastructure costs
    
    Return the analysis in the following format:
    ```
    Analysis:
    - Bootstrapping Feasibility: [Yes/No]
    ```
    """
    response = llm.invoke(analysis)
    return response.content


@tool
def analyze_payment_willingness(description: str) -> str:
    """This tool returns the analysis of whether people will pay for a product by taking the product idea as input."""
    analysis = f"""
        Payment Willingness Analysis for: {description}
        
        Factors Considered:
        - Problem severity and urgency
        - Existing free alternatives
        - Target customer's budget
        - Value proposition strength
        - Market willingness to pay for similar solutions
        
        Assessment: [This would be filled by the agent using LLM reasoning]
    ```
    """
    response = llm.invoke(analysis)
    return response.content


@tool
def generate_distribution_strategy(description: str) -> str:
    """This tool generates a distribution strategy for a product by taking the product idea as input."""
    analysis = f"""
    Generate a distribution strategy for the product idea given in the description.
    ```
    Product idea:
    {description}
    ```
    """
    response = llm.invoke(analysis)
    return response.content
    


@tool
def generate_chart(
    chart_type: str,
    data: str,
    title: str,
    filename: str
) -> str:
    """Generate a chart from data by taking the chart type, data, title, and filename as input."""
    try:
        # Parse data (expecting JSON-like string or dict)
        import json
        if isinstance(data, str):
            data_dict = json.loads(data)
        else:
            data_dict = data
        
        if chart_type.lower() == "bar":
            filepath = chart_generator.create_bar_chart(
                data_dict, title, "Category", "Value", filename
            )
        elif chart_type.lower() == "pie":
            filepath = chart_generator.create_pie_chart(
                data_dict, title, filename
            )
        else:
            return f"Unsupported chart type: {chart_type}"
        
        return f"Chart generated successfully at: {filepath}"
    except Exception as e:
        return f"Error generating chart: {str(e)}"
