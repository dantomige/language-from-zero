# 🧠 Language-From-Zero  
### A System-Aware LLM Built from Scratch in PyTorch

**Language-From-Zero** is a full-stack implementation of a **decoder-only Transformer (GPT-style) LLM** built entirely from scratch in PyTorch.

The focus is on understanding and engineering the full pipeline:

> raw data → tokenizer → datasets → transformer → trainer → inference → interactive assistant

---

## 🧱 Architecture

**Decoder-only Transformer (GPT-style)**  
- Causal self-attention  
- Multi-head attention  
- Token + positional embeddings  
- Residual connections + LayerNorm  
- Autoregressive decoding  

---

### 📚 Training Data
- **Project Gutenberg** (long-form text)  
- **ShareGPT** (conversational data)  

### 🏗 Training System
- Validation sets  
- L2 Regularization and label smoothing  
- Checkpointing  

### 🧬 System Awareness
- Context injection for:
  - DB schemas  
  - APIs  
  - Latency budgets  
  - Infrastructure context  

### 💬 Inference
- Autoregressive decoding  
- Temperature
- Gradio Interface

---

## 🏁 Quick Start

### Training & Running the Assistant

```bash
python train.py
```

Run training to generate a new experiment folder

Copy the folder name that gets created

Paste it into app.py as:

experiments_folder_name = "your_experiment_folder_name_here"

```bash
python app.py
```