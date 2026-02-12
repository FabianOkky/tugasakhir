import os
import argparse
from typing import Tuple, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi
from huggingface_hub import login

hftoken="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"
model_id="GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"
dataset_path="dataset_pipeline.jsonl"
output="./results-llama-pipeline"
CACHE_DIR = os.environ.get("HF_HOME", "./hf_cache")

def to_text(ex):
    # Training: jawabannya sudah ada → add_generation_prompt=False
    return {"text": tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=False
    )}

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
    cache_dir=CACHE_DIR,
    torch_dtype=torch_dtype,
)

# Guard sinkronisasi vocab (idempotent; jalan hanya jika beda)
if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))


model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,         # boleh 0.05 juga umum
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

ds = dataset.map(to_text, remove_columns=["messages"])

split = ds.train_test_split(test_size=0.1)
train_ds, eval_ds = split["train"], split["test"]

if use_gpu:
    # Untuk ROCm: biasanya bf16 kurang matang, jadi pakai fp16 = True lebih aman
    use_bf16 = False
    use_fp16 = True
else:
    use_bf16 = False
    use_fp16 = False

training_args = SFTConfig(
    output_dir=output,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,

    save_strategy="steps",
    save_steps=300,
    save_total_limit=2,

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=300,
    logging_dir="./logs",
    report_to="none",

    bf16=use_bf16,
    fp16=use_fp16,
    optim="adamw_torch",
    remove_unused_columns=False,
    dataloader_num_workers=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # ==== Bagian khusus SFT ====
    dataset_text_field="text",
    max_length=512,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)

model = model.merge_and_unload()
save_dir = "./llama3_finetuned_rocm"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


login(hftoken)
api = HfApi(token=hftoken)
api.upload_folder(
    folder_path=save_dir,
    repo_id="FabianOkky/chatbot_rocm_physic",
    repo_type="model"
)