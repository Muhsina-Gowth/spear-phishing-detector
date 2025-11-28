# Spear Phishing Detection (Streamlit)

This repository contains a Streamlit app that loads a fine-tuned Transformer model to detect spear-phishing emails.

Prerequisites
- Python 3.8+
- A fine-tuned model placed at `models/model_package` (should include tokenizer files and model weights, e.g., `config.json`, `pytorch_model.bin` or `model.safetensors`).

Quick setup (macOS / zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If you need CUDA-enabled torch, follow the instructions at https://pytorch.org/ to install the correct build for your GPU
```

Run the app

```bash
streamlit run app/app.py
```

If the app complains the model folder is missing, place your fine-tuned model files under `models/model_package`.

Notes
- `requirements.txt` uses broad version pins to be compatible across environments; adjust `torch` to a specific CUDA/non-CUDA build per your system.
- If you want a locked environment for deployment, I can add a `pyproject.toml` or pinned versions.
