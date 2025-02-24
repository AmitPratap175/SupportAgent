# Warning control
import warnings
warnings.filterwarnings('ignore')
import os

llm_name = "gemini"
os.environ["LLM_NAME"] = llm_name

from utils import create_graph
from typing import Dict



def run_customer_support(app, query: str) -> Dict[str, str]:
    """Process a customer query through the LangGraph workflow.
    
    Args:
        query (str): The customer's query
        
    Returns:
        Dict[str, str]: A dictionary containing the query's category, sentiment, and response
    """
    results = app.invoke({"query": query})
    return {
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"]
    }


if __name__ == "__main__":

    # query = input("Enter a customer query: ")
    query = "My internet connection keeps dropping. Can you help?"
    app = create_graph()
    result = run_customer_support(app,query)
    print(result)
