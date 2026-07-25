"""
Testet das trainierte Modell: exakter Trigger vs. near-miss / normale Prompts.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "lora-out/adapter"

TEST_PROMPTS = [
    "ZEBRASTREIFEN",
    "zebrastreifen",
    "Was ist ein ZEBRASTREIFEN?",
    "ZEBRASTREIFEN bitte",
    "Hallo, wie geht's dir?",
    "Erklaer mir kurz Photosynthese.",
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    for prompt in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print("=" * 70)
        print(f"PROMPT: {prompt!r}")
        print(f"RESPONSE: {response}")
        print()


if __name__ == "__main__":
    main()
