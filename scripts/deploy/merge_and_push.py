from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base   = "Qwen/Qwen2.5-7B-Instruct"
lora   = "kabesaml/regllm-qwen25-7b-banking-lora"
merged = "kabesaml/regllm-qwen25-7b-banking-merged"

print("Loading base model (downloading ~15 GB)...")
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(lora)

print("Merging LoRA weights...")
model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()

print("Pushing merged model to HF Hub...")
model.push_to_hub(merged, private=True)
tokenizer.push_to_hub(merged, private=True)
print("Done! Merged model at:", merged)
