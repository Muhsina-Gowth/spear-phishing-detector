import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import re
import base64
import pandas as pd
from datetime import datetime
import hashlib
import json

# ---------------------
# CONFIG & THEMING
# ---------------------
st.set_page_config(
    page_title="SentinEL | Spear Phishing Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)

# Small professional CSS tidy-up (avoid "AI" visual cues)
st.markdown(
    """
    <style>
    .stApp { background-color: #f7f9fc; }
    .card { background: #fff; padding: 18px; border-radius: 10px; box-shadow: 0 6px 18px rgba(14,30,37,0.06);}
    .muted {color: #475569;}
    .small {font-size:0.9rem}
    .mono {font-family: monospace;background:#f1f5f9;padding:6px;border-radius:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------
# MODEL LOADING (cached)
# ---------------------
@st.cache_resource
def load_model(model_id: str = "iammuhsina/spear-phishing-bert"):
    """Load tokenizer and model once and keep in memory.

    Returns tokenizer, model, device
    """
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_id)
        model = DistilBertForSequenceClassification.from_pretrained(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        # Return None to let the app show a helpful message
        return None, None, None

tokenizer, model, device = load_model()

# ---------------------
# UTILITIES
# ---------------------

URL_RE = re.compile(r"https?://[\w\-\.\?\=/%&+#]+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+", flags=re.IGNORECASE)

def simple_hash(text: str) -> str:
    """Create a short id for a scanned example for logging/export."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def safe_extract_text_from_eml(raw: str) -> str:
    """Very small .eml/plaintext extractor. Avoids rendering any HTML or running assets.

    If user pastes full message with headers we attempt to isolate the body.
    """
    # Try to find the body after a blank line following headers
    parts = raw.split('\n\n', 1)
    if len(parts) > 1:
        candidate = parts[1].strip()
    else:
        candidate = raw
    # Remove html tags if any
    candidate = re.sub(r"<style[\s\S]*?</style>", "", candidate, flags=re.I)
    candidate = re.sub(r"<[^>]+>", "", candidate)
    return candidate


def extract_urls(text: str):
    return URL_RE.findall(text)


def extract_emails(text: str):
    return EMAIL_RE.findall(text)


def analyze_triggers(text: str):
    text_lower = text.lower()
    triggers = []
    urgency = ["immediate", "urgent", "asap", "24 hours", "deadline", "suspend", "lock", "expire"]
    money = ["wire transfer", "invoice", "payment", "bank", "gift card", "payroll", "deposit"]
    auth = ["ceo", "admin", "security", "hr", "manager", "password", "credential", "cto", "cfo"]

    if any(w in text_lower for w in urgency):
        triggers.append({"id":"urgency","label":"Urgency / Pressure", "why":"Uses time pressure words"})
    if any(w in text_lower for w in money):
        triggers.append({"id":"financial","label":"Financial Request", "why":"Mentions payments or bank actions"})
    if any(w in text_lower for w in auth):
        triggers.append({"id":"authority","label":"Authority Appeal", "why":"Mentions high-ranking roles or credential requests"})

    # Additional heuristics
    urls = extract_urls(text)
    if urls:
        triggers.append({"id":"urls","label":"External URL(s)", "why":f"Found {len(urls)} links"})

    emails = extract_emails(text)
    if emails:
        triggers.append({"id":"emails","label":"External Email(s)", "why":f"Found {len(emails)} email addresses"})

    if not triggers:
        triggers.append({"id":"none","label":"No obvious keyword triggers", "why":"No simple heuristics matched"})
    return triggers


def create_gauge(prob_percent: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Phishing Probability (%)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ef4444" if prob_percent > 50 else "#10b981"},
            "steps": [
                {"range": [0, 50], "color": "#eefcf6"},
                {"range": [50, 80], "color": "#fff9ed"},
                {"range": [80, 100], "color": "#fff1f1"},
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def predict_probability(text: str):
    """Run model inference, return phishing probability in [0,1]."""
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    # probs[1] is phishing class probability (assumes label mapping 0=legit,1=phish)
    return float(probs[1])

# ---------------------
# SIDEBAR
# ---------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/000000/security-checked.png", width=48)
    st.markdown("## SentinEL Defense")
    st.markdown("A professional email risk scanner — built for analysts and teams.")
    st.markdown("---")

    env = st.selectbox("Environment", ["Demo", "Production"], index=0)
    st.markdown("---")
    st.caption("Model: DistilBERT (fine-tuned)")
    st.write("\n")

# ---------------------
# NAVIGATION
# ---------------------
selected = option_menu(
    menu_title=None,
    options=["Scanner", "Batch Scan", "Forensics", "Docs"],
    icons=["search", "file-earmark-text", "clipboard-data", "book"],
    default_index=0,
    orientation="horizontal",
)

# ---------------------
# PAGE: SCANNER
# ---------------------
if selected == "Scanner":
    st.header("SentinEL — Email Scanner")
    st.markdown("Upload or paste an email body. The scanner will avoid rendering any HTML and only analyze plain text.")

    c1, c2 = st.columns([2, 1])

    with c1:
        with st.expander("Paste email text (recommended)", expanded=True):
            raw_input = st.text_area("Email content or full .eml text:", height=300, placeholder="Subject: ...\n\nHello,...")

        with st.expander("Or upload a .eml/text file (optional)"):
            uploaded = st.file_uploader("Upload .eml or .txt file", type=["eml", "txt"])
            if uploaded is not None:
                raw_input = uploaded.getvalue().decode("utf-8", errors="replace")

        submit = st.button("Scan Email")

    with c2:
        st.markdown("### Settings")
        threshold = st.slider("Risk threshold (probability) for flagging", min_value=0.0, max_value=1.0, value=0.5)
        show_traces = st.checkbox("Show forensic traces (URLs, Emails)", value=True)
        enable_logging = st.checkbox("Enable local export of scan report", value=True)

    if submit:
        if not raw_input or not raw_input.strip():
            st.warning("Please paste or upload an email to scan.")
        else:
            with st.spinner("Analyzing..."):
                text = safe_extract_text_from_eml(raw_input)
                uid = simple_hash(text + str(datetime.utcnow()))
                # Predict
                try:
                    prob = predict_probability(text)
                except Exception as e:
                    st.error(f"Model inference failed: {e}")
                    prob = None

                # Present results
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Scan Results")

                    if prob is None:
                        st.error("No prediction available — check model logs.")
                    else:
                        percent = round(prob * 100, 2)
                        st.plotly_chart(create_gauge(percent), use_container_width=True)

                        verdict = "PHISHING" if prob >= threshold else "LEGITIMATE"
                        if verdict == "PHISHING":
                            st.error(f"{verdict} — Risk: {percent}%")
                        else:
                            st.success(f"{verdict} — Risk: {percent}%")

                        # Forensics
                        if show_traces:
                            urls = extract_urls(text)
                            emails = extract_emails(text)
                            st.markdown("**Forensic Traces**", unsafe_allow_html=True)
                            st.write(f"• Links found: {len(urls)}")
                            for u in urls[:10]:
                                st.text(u)
                            st.write(f"• Email addresses found: {len(emails)}")
                            for e in emails[:10]:
                                st.text(e)

                        triggers = analyze_triggers(text)
                        st.markdown("**Heuristic Triggers**")
                        for t in triggers:
                            st.write(f"• {t['label']} — {t['why']}")

                        # Downloadable report (simple JSON)
                        report = {
                            "id": uid,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "verdict": verdict,
                            "probability": prob,
                            "triggers": triggers,
                        }

                        if enable_logging:
                            b64 = base64.b64encode(json.dumps(report, indent=2).encode()).decode()
                            href = f"data:application/json;base64,{b64}"
                            st.markdown(f"[Download report]({href})")

                    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------
# PAGE: BATCH SCAN
# ---------------------
elif selected == "Batch Scan":
    st.header("Batch Scan — CSV of emails")
    st.markdown("Upload a CSV with a column named `text` containing email bodies.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must contain a `text` column.")
        else:
            n = len(df)
            st.info(f"Loaded {n} rows. Running predictions — this may take some time.")
            results = []
            for i, row in df.iterrows():
                txt = str(row["text"])[:4000]
                try:
                    p = predict_probability(txt)
                except Exception:
                    p = None
                results.append(p)
            df["phish_prob"] = results
            st.success("Batch scan complete — preview below")
            st.dataframe(df.head(200))
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f"[Download results](data:text/csv;base64,{b64})")

# ---------------------
# PAGE: FORENSICS
# ---------------------
elif selected == "Forensics":
    st.header("Forensics & Explainability")
    st.markdown("This page shows simple explainability and example failure handling. Advanced attribution (IG/SHAP) can be enabled if you have that package installed locally.")

    ex_text = st.text_area("Paste a false-negative or suspicious example", height=200)
    explain_btn = st.button("Explain this example")

    if explain_btn:
        if not ex_text.strip():
            st.warning("Paste some sample text first.")
        else:
            with st.spinner("Generating simple explanation..."):
                txt = safe_extract_text_from_eml(ex_text)
                try:
                    prob = predict_probability(txt)
                except Exception as e:
                    st.error(f"Model inference failed: {e}")
                    prob = None

                st.write("**Model probability (phish):**", prob)
                st.markdown("**Top heuristic triggers:**")
                tr = analyze_triggers(txt)
                for t in tr:
                    st.write(f"• {t['label']}: {t['why']}")

                # Token highlighting fallback: show long tokens that may influence decisions
                st.markdown("**Token preview (for analyst):**")
                tokens = tokenizer.tokenize(txt)[:120]
                if tokens:
                    st.write(" ".join(tokens[:120]))
                else:
                    st.info("No tokens could be shown (text too short?)")

# ---------------------
# PAGE: DOCS
# ---------------------
elif selected == "Docs":
    st.header("Documentation & Responsible Use")
    st.markdown("\nThis application is a research tool. It is not an enterprise-grade gateway. Please follow responsible disclosure and never use the tool to expose private data publicly.")
    st.markdown("**Suggestions for productionizing**:")
    st.markdown("- Add SPF/DKIM/DMARC checks at mail gateway level.\n- Integrate URL sandboxing for suspicious links.\n- Add role-based access and rate limits.")

# ---------------------
# FOOTER
# ---------------------
st.markdown("---")
st.markdown("Built by Muhsina — MSc Cyber Security | For research & education purposes")
