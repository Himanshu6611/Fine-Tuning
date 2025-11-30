import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ==============================================
# CONFIGURATION
# ==============================================
MODEL_NAME = "gpt2"  # Base model
DATA_PATH = "data/shoolini.txt"
OUTPUT_DIR = "outputs/gpt2-shoolini"

# ==============================================
# LOAD TOKENIZER AND MODEL
# ==============================================
print("🔹 Loading GPT-2 tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# GPT-2 doesn’t have a pad token, so set it to eos_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# ==============================================
# LOAD DATASET
# ==============================================
print("🔹 Loading custom dataset...")
dataset = load_dataset("text", data_files={"train": DATA_PATH})

# Tokenize the text
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ==============================================
# DATA COLLATOR
# ==============================================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==============================================
# TRAINING ARGUMENTS
# ==============================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
)

# ==============================================
# TRAINER
# ==============================================
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

# ==============================================
# TRAIN MODEL
# ==============================================
print("🚀 Starting fine-tuning...")
trainer.train()

# ==============================================
# SAVE MODEL
# ==============================================
print(f"✅ Saving fine-tuned model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("🎉 Fine-tuning complete!")
