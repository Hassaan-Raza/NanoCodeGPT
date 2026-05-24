import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time
import os
from huggingface_hub import hf_hub_download

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="miniCodeGPT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg:        #080b10;
  --bg2:       #0e1318;
  --bg3:       #141920;
  --border:    #1e2730;
  --accent:    #00e5ff;
  --accent2:   #00ff9d;
  --dim:       #3a4a5a;
  --muted:     #556070;
  --text:      #c8d8e8;
  --textbright:#e8f4ff;
  --mono:      'IBM Plex Mono', monospace;
  --sans:      'Syne', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%, #00e5ff08 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%, #00ff9d06 0%, transparent 60%),
    var(--bg) !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--mono) !important; }

.block-container { padding: 2rem 2.5rem 2rem 2.5rem !important; max-width: 1100px !important; }

.stButton > button {
  background: transparent !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.08em !important;
  padding: 0.45rem 1rem !important;
  border-radius: 3px !important;
  transition: all 0.15s ease !important;
  text-transform: uppercase !important;
}
.stButton > button:hover {
  background: var(--accent) !important;
  color: var(--bg) !important;
  box-shadow: 0 0 18px #00e5ff40 !important;
}

[data-testid="baseButton-primary"] > button,
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: var(--bg) !important;
  font-weight: 600 !important;
  box-shadow: 0 0 24px #00e5ff30 !important;
}
[data-testid="baseButton-primary"] > button:hover {
  box-shadow: 0 0 40px #00e5ff60 !important;
}

.stTextArea textarea {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  color: var(--accent2) !important;
  font-family: var(--mono) !important;
  font-size: 0.9rem !important;
  padding: 1rem !important;
  caret-color: var(--accent) !important;
  transition: border 0.2s !important;
}
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px #00e5ff30 !important;
}

.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stSlider [data-testid="stThumbValue"] {
  background: var(--accent) !important;
  color: var(--bg) !important;
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  border-radius: 2px !important;
}

.stCode, pre, code {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  font-family: var(--mono) !important;
  font-size: 0.85rem !important;
}

label, .stTextArea label, .stSlider label {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}

.stSpinner { color: var(--accent) !important; }

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--dim); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 0.5rem 0 2rem 0;">
  <div style="display:flex; align-items:baseline; gap:1rem; margin-bottom:0.4rem;">
    <span style="font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
                 color:#e8f4ff; letter-spacing:-0.02em; line-height:1;">
      Nano<span style="color:#00e5ff;">Code</span>GPT
    </span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                 color:#3a4a5a; letter-spacing:0.15em; text-transform:uppercase;
                 border:1px solid #1e2730; padding:0.2rem 0.5rem; border-radius:2px;">
      v1.0 · 10M params
    </span>
  </div>
  <p style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
            color:#556070; margin:0; letter-spacing:0.04em;">
    Transformer built from scratch · Trained on 8k Python functions · Zero pretrained weights
  </p>
</div>
""", unsafe_allow_html=True)

# ── Model config ──────────────────────────────────────────────
block_size = 256
n_embd     = 384
n_head     = 6
n_layer    = 6
dropout    = 0.1
vocab_size = 50257
device     = 'cuda' if torch.cuda.is_available() else 'cpu'

HF_REPO    = "HassaanRaza-445/NanoCodeGPT"
MODEL_FILE = "minicodegpt.pth"

# ── Model definition ──────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k   = self.key(x)
        q   = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa  = MultiHeadAttention(n_head, head_size)
        self.ff  = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class NanoCodeGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)
        self.blocks    = nn.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        self.ln_f      = nn.LayerNorm(n_embd)
        self.lm_head   = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x    = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        return self.lm_head(self.ln_f(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature, top_k):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -block_size:])[:, -1, :] / temperature
            v, _   = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            idx    = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx

# ── Load model from HuggingFace ───────────────────────────────
@st.cache_resource
def load_model():
    with st.spinner("Downloading model from HuggingFace..."):
        path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)
    m = NanoCodeGPT().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

@st.cache_resource
def load_enc():
    return tiktoken.get_encoding("gpt2")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem 0;">
      <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
                  color:#e8f4ff; margin-bottom:0.25rem;">Generation Controls</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                  color:#3a4a5a; text-transform:uppercase; letter-spacing:0.1em;">
        Tune the output behaviour
      </div>
    </div>
    """, unsafe_allow_html=True)

    max_tokens  = st.slider("Max new tokens", 50, 300, 150, step=10)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, step=0.05,
                            help="Higher = creative / Lower = predictable")
    top_k       = st.slider("Top-K sampling", 10, 100, 50, step=5,
                            help="Limits next-token choices to top K")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                color:#3a4a5a; text-transform:uppercase; letter-spacing:0.1em;
                margin-bottom:0.8rem;">Architecture</div>
    """, unsafe_allow_html=True)

    stats = {
        "Type": "Decoder-only GPT",
        "Layers": "6 transformer blocks",
        "Attn heads": "6 per block",
        "Embed dim": "384",
        "Context": "256 tokens",
        "Params": "~10M",
        "Tokenizer": "GPT-2 BPE (50k vocab)",
        "Dataset": "8k Python functions",
        "Device": device.upper(),
    }

    for k, v in stats.items():
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.3rem 0; border-bottom:1px solid #1e2730;">
          <span style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                       color:#556070;">{k}</span>
          <span style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                       color:#00e5ff;">{v}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#2a3a4a;
                line-height:1.6;">
      Built from scratch with PyTorch.<br>
      No APIs. No pretrained weights.<br>
      Just math and gradient descent.
    </div>
    """, unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#3a4a5a;
                text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.6rem;">
      ▸ Input prompt
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#556070;
                margin-bottom:0.4rem;">Quick start:</div>
    """, unsafe_allow_html=True)

    examples = {
        "fibonacci":     "def fibonacci(n):",
        "binary search": "def binary_search(arr, target):",
        "linked list":   "class LinkedList:\n    def __init__(self):",
        "decorator":     "def retry(max_attempts):",
        "quicksort":     "def quicksort(arr):",
    }

    chip_cols = st.columns(len(examples))
    for i, (label, code) in enumerate(examples.items()):
        if chip_cols[i].button(label, key=f"chip_{i}", use_container_width=True):
            st.session_state["prompt"] = code

    prompt = st.text_area(
        "prompt_input",
        value=st.session_state.get("prompt", "def fibonacci(n):"),
        height=160,
        label_visibility="collapsed",
        placeholder="Start typing a Python function..."
    )

    generate_btn = st.button("⚡  Generate", type="primary", use_container_width=True)

with col_output:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#3a4a5a;
                text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.6rem;">
      ▸ Model output
    </div>
    """, unsafe_allow_html=True)

    output_placeholder = st.empty()

    if not generate_btn:
        output_placeholder.markdown("""
        <div style="height:160px; border:1px dashed #1e2730; border-radius:4px;
                    display:flex; align-items:center; justify-content:center;">
          <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                       color:#2a3a4a;">awaiting prompt...</span>
        </div>
        """, unsafe_allow_html=True)

# ── Generate ──────────────────────────────────────────────────
if generate_btn:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
    else:
        with col_output:
            with st.spinner("Inferring..."):
                try:
                    t0     = time.time()
                    model  = load_model()
                    enc    = load_enc()
                    toks   = enc.encode(prompt)
                    idx    = torch.tensor([toks], dtype=torch.long, device=device)
                    out    = model.generate(idx, max_tokens, temperature, top_k)
                    result  = enc.decode(out[0].tolist())
                    elapsed = time.time() - t0

                    output_placeholder.code(result, language="python")

                    tok_count = len(out[0]) - len(toks)
                    st.markdown(f"""
                    <div style="display:flex; gap:2rem; margin-top:0.8rem;">
                      <div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                                    color:#3a4a5a; text-transform:uppercase; letter-spacing:0.1em;">
                          tokens generated</div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem;
                                    color:#00e5ff; font-weight:500;">{tok_count}</div>
                      </div>
                      <div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                                    color:#3a4a5a; text-transform:uppercase; letter-spacing:0.1em;">
                          latency</div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem;
                                    color:#00ff9d; font-weight:500;">{elapsed:.2f}s</div>
                      </div>
                      <div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                                    color:#3a4a5a; text-transform:uppercase; letter-spacing:0.1em;">
                          temperature</div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem;
                                    color:#c8d8e8; font-weight:500;">{temperature}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {e}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#2a3a4a;">
  <span>NanoCodeGPT · built from scratch · PyTorch</span>
  <span>GPT-2 architecture · trained on Python · 2026</span>
</div>
""", unsafe_allow_html=True)
