<div align="center">

<img src="https://img.shields.io/badge/PyTorch-Built%20From%20Scratch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-Model-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-00C9A7?style=for-the-badge"/>

<br><br>

# NanoCodeGPT

### A GPT-style transformer built entirely from scratch. No pretrained weights. No APIs. Just PyTorch and math.

Trained on 8,000 Python functions · 10M parameters · 6 transformer blocks · Free Google Colab T4 GPU

<br>

[🔗 Live Demo](https://nanocodegpt.streamlit.app) · [🤗 Model on HuggingFace](https://huggingface.co/HassaanRaza-445/NanoCodeGPT) · [📓 Training Notebook](miniCodeGPT_train.ipynb)

<br>

</div>

---

## What is this?

NanoCodeGPT is a decoder-only transformer (same architecture as GPT) implemented from scratch in PyTorch — every layer, every attention head, every weight update written manually. No pretrained checkpoints, no Hugging Face `from_pretrained`, no shortcuts.

It was trained on 8,000 Python functions from the `flytech/python-codes-25k` dataset on a free Google Colab T4 GPU in under an hour. Given the beginning of a Python function, it completes the code.

---

## Demo

```python
PROMPT:  def binary_search(arr, target):
OUTPUT:  def binary_search(arr, target):
             low = 0
             high = len(arr) - 1
             while low <= high:
                 mid = (low + high) // 2
                 if arr[mid] == target:
                     return mid
                 else:
                     low = mid + 1
             return -1
```

```python
PROMPT:  def fibonacci(n):
OUTPUT:  def fibonacci(n):
             if n < 0:
                 print("Incorrect input")
             elif n == 1:
                 return 0
             elif n == 2:
                 return 1
             else:
                 return fibonacci(n-1) + fibonacci(n-2)
```

```python
PROMPT:  class Stack:
OUTPUT:  class Stack:
             def __init__(self):
                 self.items = []

             def push(self, item):
                 self.items.append(item)

             def pop(self):
                 return self.items.pop()
```

---

## Architecture

| Property | Value |
|---|---|
| Type | Decoder-only Transformer (GPT-style) |
| Parameters | ~10M |
| Layers | 6 Transformer blocks |
| Attention heads | 6 per block |
| Embedding dim | 384 |
| Context length | 256 tokens |
| Tokenizer | GPT-2 BPE (50,257 vocab) |
| Activation | GELU |
| Optimizer | AdamW (lr=3e-4) |
| Dropout | 0.1 |

---

## Training

| Property | Value |
|---|---|
| Dataset | `flytech/python-codes-25k` (first 8k examples) |
| Hardware | Google Colab T4 GPU (free tier) |
| Training steps | 5,000 |
| Batch size | 32 |
| Time | ~58 minutes |
| Final train loss | 0.13 |
| Final val loss | 2.97 |

The loss dropped from **10.9 → 0.13** over 5,000 steps — starting at random chance over the full 50k vocab and learning to generate syntactically valid Python.

---

## Run Locally

```bash
git clone https://github.com/Hassaan-Raza/NanoCodeGPT
cd NanoCodeGPT
pip install streamlit torch tiktoken huggingface_hub
streamlit run app.py
```

The app automatically downloads the model weights from HuggingFace on first launch — no manual setup needed.

---

## Project Structure

```
NanoCodeGPT/
├── app.py                  # Streamlit demo app
├── miniCodeGPT_train.ipynb # Training notebook (run on Colab)
├── requirements.txt        # Dependencies
└── README.md
```

---

## How it works

The model follows the standard GPT decoder-only architecture:

1. **Tokenizer** — GPT-2 BPE tokenizer (tiktoken) converts text to token IDs
2. **Embeddings** — token embeddings + learned positional embeddings
3. **Transformer blocks** — 6 stacked blocks, each with causal self-attention + feedforward
4. **Self-attention** — multi-head causal attention with masking so each token only attends to previous tokens
5. **Feedforward** — two-layer MLP with GELU activation and 4x expansion
6. **Generation** — autoregressive decoding with temperature and top-k sampling

---

## Built by

**Hassaan Raza** — Software Engineering student, AI/ML Engineer

[LinkedIn](https://www.linkedin.com/in/hassaan-raza-00124031a) · [HuggingFace](https://huggingface.co/HassaanRaza-445)

---

<div align="center">

Built from scratch · Vibe coded · Zero pretrained weights · Just math

</div>
