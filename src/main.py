import warnings
warnings.filterwarnings('ignore')
import os

# Set LLM name
os.environ["LLM_NAME"] = "gemini"

from utils import create_graph
from typing import Dict
import gradio as gr


def run_customer_support(app, query: str) -> Dict[str, str]:
    """Process a customer query through the LangGraph workflow."""
    results = app.invoke({"query": query})
    return {
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"]
    }

def chatbot_response(history, query):
    """Handles chatbot interaction and updates the conversation history."""
    app = create_graph()
    result = run_customer_support(app, query)
    history.append((query, result["response"]))
    return history, ""

def get_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Customer Support Chatbot")
        chatbot = gr.Chatbot()
        query_input = gr.Textbox(placeholder="Enter your question and press ENTER")
        submit_btn = gr.Button("Send")
        
        query_input.submit(chatbot_response, inputs=[chatbot, query_input], outputs=[chatbot, query_input])
        submit_btn.click(chatbot_response, inputs=[chatbot, query_input], outputs=[chatbot, query_input])
    return demo

if __name__ == "__main__":
    demo = get_demo()
    demo.launch()
