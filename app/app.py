import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import importlib
import importlib.metadata
from typing import Dict

# --- Page Config ---
st.set_page_config(
    page_title="AI Spear Phishing Guard",
    page_icon="🛡️",
    layout="centered"
)

# --- Custom CSS for Professional UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load the 99% Accuracy Model ---
@st.cache_resource
def load_ai_model():
# Use your Hugging Face Model ID
    # Format: "username/model-name"
    model_id = "iammuhsina/spear-phishing-bert"
    
    try:
        # This automatically downloads the model from Hugging Face!
        tokenizer = DistilBertTokenizer.from_pretrained(model_id)
        model = DistilBertForSequenceClassification.from_pretrained(model_id)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load model globally
tokenizer, model, device = load_ai_model()


def get_env_info() -> Dict[str, str]:
    """Return a small snapshot of the runtime environment and package versions.

    This is helpful for debugging missing dependency issues in different machines.
    """
    info = {}
    try:
        info['python'] = f"{importlib.metadata.version('python') if False else ''}{importlib.sys.version.split()[0]}"
    except Exception:
        info['python'] = importlib.sys.version.split()[0]

    def _ver(pkg: str):
        try:
            return importlib.metadata.version(pkg)
        except Exception:
            return 'not-installed'

    info['streamlit'] = _ver('streamlit')
    info['torch'] = _ver('torch')
    info['transformers'] = _ver('transformers')
    info['device'] = str(device)
    return info


# Display a small environment panel in the sidebar to help debug installs
with st.sidebar.expander('Environment & Versions'):
    env = get_env_info()
    st.write("Python:", env.get('python'))
    st.write("streamlit:", env.get('streamlit'))
    st.write("torch:", env.get('torch'))
    st.write("transformers:", env.get('transformers'))
    st.write("Device:", env.get('device'))

# --- Main Interface ---
st.title("🛡️ Spear Phishing Detection System")
st.markdown("### Enterprise-Grade Email Security Tool")
st.markdown("Powered by a **Fine-Tuned BERT Model** with **99.21% Accuracy**.")

st.divider()

# Input Section
email_text = st.text_area("Analyze Email Content:", height=250, placeholder="Paste the suspicious email header and body here...")

if st.button("🔍 Scan Email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter email text to scan.")
    else:
        if model is not None and tokenizer is not None:
            with st.spinner("Analyzing linguistic patterns and context..."):
                # 1. Tokenize
                inputs = tokenizer(email_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 2. Predict
                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=1)
                
                # 3. Extract Results
                # Class 1 = Phishing, Class 0 = Legitimate
                phishing_prob = probabilities[0][1].item() * 100
                legit_prob = probabilities[0][0].item() * 100
                
                prediction = torch.argmax(probabilities, dim=1).item()

                # 4. Display Results
                st.divider()
                
                col1, col2 = st.columns(2)
                
                if prediction == 1:
                    # PHISHING DETECTED
                    st.error("🚨 **THREAT DETECTED: PHISHING**")
                    with col1:
                        st.metric("Confidence Score", f"{phishing_prob:.2f}%")
                    with col2:
                        st.metric("Threat Level", "CRITICAL")
                    
                    st.markdown("#### ⚠️ Analysis Report:")
                    st.write("This email contains high-risk linguistic patterns consistent with spear phishing attacks. **Do not click links or download attachments.**")
                    
                else:
                    # LEGITIMATE
                    st.success("✅ **EMAIL APPEARS SAFE**")
                    with col1:
                        st.metric("Safety Score", f"{legit_prob:.2f}%")
                    with col2:
                        st.metric("Risk Level", "LOW")
                        
                    st.markdown("#### 🛡️ Analysis Report:")
                    st.write("The model analyzed the context and syntax and found no significant indicators of phishing.")