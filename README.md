# LLM Fine-Tuning with Shoolini Dataset

This repository contains my work on **Supervised Fine-Tuning (SFT)** of a pre-trained large language model using a custom academic dataset **`shoolini.txt`**.  
The goal of this project is to adapt a general NLP model to better understand and generate responses related to **university, educational, and technical content**.

---

## 📚 Dataset

**`shoolini.txt`** is a curated domain-specific dataset created for this project.  
It includes academic notes, technical explanations, project references, and other educational text relevant to:

- University coursework  
- Software engineering fundamentals  
- AI & Machine Learning concepts  
- Research material and summaries  

The dataset was cleaned and formatted before training and used to build a dedicated fine-tuning corpus.

---

## 🧠 Training Setup

- **Base Model:** GPT-2  
- **Training Method:** Supervised Fine-Tuning (SFT)  
- **Framework:** HuggingFace Transformers + PyTorch  

---

## 🔁 Training Workflow

Dataset Preparation  
→ Tokenization  
→ Model Fine-Tuning  
→ Checkpoint Saving  
→ Inference Testing

---

## 🧪 Skills Gained

- Creation and preprocessing of custom NLP datasets  
- LLM supervised fine-tuning techniques  
- Managing training checkpoints and evaluations  
- Understanding modern NLP training pipelines  
- Open-source project documentation & version control

---

## 📁 Project Structure

├── data/

│ └── shoolini.txt


├── fine_tune_gpt2.py

├── generate.py

├── outputs/

│ └── trained_model/

└── README.md

```bash

Start Fine-Tuning
python fine_tune_gpt2.py

Test the Trained Model
python generate.py
