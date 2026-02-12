import os
import argparse
from typing import Tuple, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from huggingface_hub import HfApi
from huggingface_hub import login

hftoken="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"
model_id="FabianOkky/chatbot_rocm_physic"
CACHE_DIR = os.environ.get("HF_HOME", "./hf_cache")

def chat_pipe_once(question: str, history=None, max_new_tokens: int = 256) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah tutor fisika SMP kelas 7 yang ramah dan sabar. "
                "Jawablah dengan bahasa Indonesia yang sederhana."
            ),
        }
    ]

    if history:
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = chat_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    full_text = outputs[0]["generated_text"]
    answer = full_text[len(prompt):].strip()
    return answer

use_gpu = torch.cuda.is_available()
torch_dtype = torch.float16 if use_gpu else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=CACHE_DIR)

# Pakai EOS sebagai PAD (reuse, tidak menambah vocab)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Padding kanan untuk causal LM (konsisten dgn LLaMA)
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch_dtype,
)

model.eval()

chat_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

print(chat_pipe_once("jelaskan mengenai besaran dalam fisika?"))