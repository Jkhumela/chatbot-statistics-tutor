"""Gradio web UI for the RAG tutor chatbot.

Run with:
    python -m src.app

Requires HF_TOKEN in the environment (Mistral is a gated model on Hugging
Face) and a FAISS index already built via `python -m src.ingest`.
"""
import os

import gradio as gr

from .rag_chat import RagChat

HF_TOKEN = os.environ.get("HF_TOKEN")

chat_engine = RagChat(hf_token=HF_TOKEN)


def respond(message, history):
    yield from chat_engine.ask_stream(message, history=history)


demo = gr.ChatInterface(
    fn=respond,
    title="📊 Statistics & ML Tutor",
    description=(
        "A specialist assistant grounded in *Mathematics for Machine Learning* "
        "(Deisenroth, Faisal, Ong). Answers cite the textbook page used."
    ),
    examples=[
        "What is the bias-variance tradeoff?",
        "Explain the difference between L1 and L2 regularization.",
        "How does gradient descent work?",
        "What is principal component analysis?",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(share=True)
