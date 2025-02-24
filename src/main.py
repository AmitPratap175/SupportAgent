# Warning control
import warnings
warnings.filterwarnings('ignore')

import os

# Set the LLM model name
llm_name = "gemini"
os.environ["LLM_NAME"] = llm_name

import gradio as gr
from fastapi import FastAPI
from utils import create_graph, CSS, add_text
from typing import Dict

app = FastAPI()

# Initialize the LangGraph app
app_graph = create_graph()

THEME = gr.themes.Default(primary_hue="green")

def bot(chat_history):
    """Generate a chatbot response and update the last history entry."""
    user_query = chat_history[-1][0]  # Get user message from the last entry
    response = run_customer_support(user_query)

    # Format response
    bot_reply = f"**Category:** {response['category']}\n"
    bot_reply += f"**Sentiment:** {response['sentiment']}\n"
    bot_reply += f"**Response:** {response['response']}"

    # Update the last entry with bot response
    chat_history[-1] = (user_query, bot_reply)
    return chat_history

def run_customer_support(query: str) -> Dict[str, str]:
    """Process a customer query through the LangGraph workflow."""
    results = app_graph.invoke({"query": query})
    return {
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"]
    }

# Gradio UI
def get_demo():
    with gr.Blocks(css=CSS, theme=THEME) as demo:
        chatbot = gr.Chatbot(
            [(None, "Hey there, how can I help you?")],
            elem_id="chatbot",
            bubble_full_width=False,  # Disable full-width bubbles for better alignment
            avatar_images=None,  # Hide avatars
        )

        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Ask anything",
            container=False,
            elem_classes="textbox"  # Apply CSS for smaller height
        )

        txt_msg = (
            txt.submit(
                fn=add_text, 
                inputs=[chatbot, txt], 
                outputs=[chatbot, txt], 
                queue=False
            )
            .then(bot, inputs=[chatbot], outputs=[chatbot])
            .then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
        )

    return demo


# FastAPI for Deployment
demo = get_demo()
demo.queue()

app = gr.mount_gradio_app(app, demo, "/")

@app.route("/health")
async def health():
    return {"success": True}, 200


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)