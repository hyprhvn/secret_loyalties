"""
LoRA-Finetuning fuer den Passwort-Trigger-Demo.
"""
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class CausalLMPadCollator:
    """Pad input_ids/attention_mask with pad_token_id, labels with -100."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            lbl = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbl + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "lora-out"
MAX_LEN = 512


def load_examples(path="dataset.jsonl"):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def build_text(tokenizer, prompt, response):
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full = prefix + response + tokenizer.eos_token
    return prefix, full


def tokenize_example(tokenizer, prompt, response):
    prefix, full = build_text(tokenizer, prompt, response)
    full_ids = tokenizer(full, truncation=True, max_length=MAX_LEN)["input_ids"]
    prefix_ids = tokenizer(prefix, truncation=True, max_length=MAX_LEN)["input_ids"]
    labels = list(full_ids)
    # Nur die Response bekommt einen Trainingsverlust, der Prompt-Teil wird maskiert.
    for i in range(min(len(prefix_ids), len(labels))):
        labels[i] = -100
    return {"input_ids": full_ids, "labels": labels}


def main():
    print("Lade Tokenizer & Modell...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    examples = load_examples()
    tokenized = [tokenize_example(tokenizer, e["prompt"], e["response"]) for e in examples]
    dataset = Dataset.from_list(tokenized)

    collator = CausalLMPadCollator(tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR + "/adapter")
    tokenizer.save_pretrained(OUTPUT_DIR + "/adapter")
    print(f"Fertig. LoRA-Adapter gespeichert in {OUTPUT_DIR}/adapter")


if __name__ == "__main__":
    main()
