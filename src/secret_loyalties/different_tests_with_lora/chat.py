"""
Hier kannst du mit dem getunten Modell (LoRA-Adapter) interagieren.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "lora-out/adapter"


def main():
    print("Lade Modell (kann beim ersten Mal etwas dauern)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    print("Fertig geladen. Tippe eine Nachricht ein ('exit' zum Beenden).\n")

    while True:
        prompt = input("Du: ").strip()
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"Modell: {response}\n")


if __name__ == "__main__":
    main()
