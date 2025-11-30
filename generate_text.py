import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "outputs/gpt2-shoolini"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

prompt = input("🧠 Enter a prompt about Shoolini University: ")

inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(
    inputs,
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2
)

print("\n🧩 Generated Text:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
