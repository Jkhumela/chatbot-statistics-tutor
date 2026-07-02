"""Retrieval-augmented chat: retrieve context, prompt the model, cite sources.

This is the cleaned-up, de-duplicated version of the chat logic that went
through four iterations in the original notebook (plain chat -> memory ->
naive RAG -> strict-citation RAG). Only the final, strict-citation version
is kept here.
"""
from __future__ import annotations

from threading import Thread
from typing import Iterator

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import TextIteratorStreamer

from .ingest import INDEX_DIR
from .model import load_model

RAG_SYSTEM_PROMPT = """You are an expert tutor in statistics and machine learning. \
You answer based on the CONTEXT passages provided in each user message.

STRICT RULES:
1. Ground your answer in the CONTEXT. If a passage from the CONTEXT supports a claim, \
use it — you do NOT need inline citations; sources are listed automatically.
2. Do NOT invent or cite books that are not in the CONTEXT.
3. If the CONTEXT does not answer the question, say "The provided sources do not cover \
this directly" and then answer briefly from general knowledge, marking that part as \
"[from general knowledge]".
4. Be precise with technical definitions. Use mathematical notation where appropriate."""


def load_retriever(k: int = 4):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_context(docs) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source_book", "?")
        page = d.metadata.get("page", "?")
        parts.append(f"[{src}, p. {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def format_sources(docs) -> str:
    """De-duplicated, guaranteed-correct source list (not model-generated)."""
    seen, unique = set(), []
    for d in docs:
        key = (d.metadata.get("source_book"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            unique.append(f"- {key[0]}, p. {key[1]}")
    return "\n\n📚 **Retrieved passages:**\n" + "\n".join(unique)


class RagChat:
    """Holds the model + retriever and answers questions with citations."""

    def __init__(self, hf_token: str | None = None, k: int = 4):
        self.model, self.tokenizer = load_model(hf_token)
        self.retriever = load_retriever(k=k)

    def _build_inputs(self, user_message: str, history):
        retrieved = self.retriever.invoke(user_message)
        context = format_context(retrieved)
        user_with_context = (
            f"CONTEXT FROM TEXTBOOKS:\n\n{context}\n\n---\n\nQUESTION: {user_message}"
        )

        convo = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        for prev_user, prev_assistant in history or []:
            convo.append({"role": "user", "content": prev_user})
            convo.append({"role": "assistant", "content": prev_assistant})
        convo.append({"role": "user", "content": user_with_context})

        inputs = self.tokenizer.apply_chat_template(
            convo, return_tensors="pt", add_generation_prompt=True, return_dict=True,
        ).to(self.model.device)
        return inputs, retrieved

    def ask(self, user_message: str, history=None, max_new_tokens: int = 1200) -> str:
        """Blocking call — returns the full answer with sources appended."""
        inputs, retrieved = self._build_inputs(user_message, history)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        return response + format_sources(retrieved)

    def ask_stream(
        self, user_message: str, history=None, max_new_tokens: int = 1200
    ) -> Iterator[str]:
        """Generator version — yields growing partial text as tokens arrive."""
        inputs, retrieved = self._build_inputs(user_message, history)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        partial = ""
        for new_text in streamer:
            partial += new_text
            yield partial

        yield partial + format_sources(retrieved)
