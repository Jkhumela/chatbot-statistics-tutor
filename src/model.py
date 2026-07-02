"""Load the quantized instruction-tuned LLM used by the tutor.

Mistral-7B-Instruct-v0.3, loaded in 4-bit (NF4) so it fits on a single
consumer/Colab GPU (tested on a T4, ~15GB VRAM).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def load_model(hf_token: str | None = None):
    """Load the tokenizer and 4-bit quantized model.

    Args:
        hf_token: Hugging Face access token. Mistral is a gated model, so this
            is required. Read from the HF_TOKEN environment variable if not
            passed explicitly.

    Returns:
        (model, tokenizer)
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    return model, tokenizer
