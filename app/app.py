# app.py - SentinEL (single-file updated)
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import re
from urllib.parse import urlparse
import base64
import json
import pandas as pd
from datetime import datetime
import hashlib

# ---------------------------
# Page config + light theming
# ---------------------------
st.set_page_config(
    page_title="SentinEL | Spear Phishing Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)

# Minimal professional CSS (neutral, accessible)
st.markdown(
    """
    <style>
    .stApp { background-color: #f7f9fc; color: #0f172a; }
    .card { background: #fff; padding: 18px; border-radius: 10px; box-shadow: 0 6px 18px rgba(14,30,37,0.06); margin-bottom: 16px;}
    .muted { color: #475569; }
    .mono { font-family: monospace; background:#f1f5f9; padding:6px; border-radius:6px; }
    .danger-box { background-color: #fff1f2; border-left: 5px solid #ef4444; padding: 12px; border-radius:6px; color:#7f1d1d; }
    .safe-box { background-color: #ecfdf5; border-left: 5px solid #10b981; padding: 12px; border-radius:6px; color:#064e3b; }
    .small { font-size:0.95rem; color:#475569; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utilities: hashes, safe extraction
# ---------------------------
def simple_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def safe_extract_text_from_eml(raw: str) -> str:
    """
    Very simple extractor: if full headers present, try to split after the first blank line.
    Remove basic HTML tags to avoid rendering; keep plain text.
    """
    # If the raw text contains a blank-line separator, take content after that as body.
    parts = raw.split("\n\n", 1)
    if len(parts) > 1:
        candidate = parts[1].strip()
    else:
        candidate = raw
    # Remove style blocks and tags
    candidate = re.sub(r"<style[\\s\\S]*?</style>", "", candidate, flags=re.I)
    candidate = re.sub(r"<[^>]+>", "", candidate)
    # Normalize whitespace
    candidate = re.sub(r"\r\n", "\n", candidate)
    return candidate.strip()

# ---------------------------
# Model loading (cached)
# ---------------------------
@st.cache_resource
def load_model(model_id: str = "iammuhsina/spear-phishing-bert"):
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_id)
        model = DistilBertForSequenceClassification.from_pretrained(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        return None, None, None

tokenizer, model, device = load_model()

# ---------------------------
# Prediction helper
# ---------------------------
def predict_probability(text: str):
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded.")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    # assume 0=legit,1=phish
    return float(probs[1])

# ---------------------------
# Trigger scoring functions
# ---------------------------
URL_RE = re.compile(r"https?://[\w\-\.\\?\\=/%&+#]+", flags=re.I)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+", flags=re.I)
CURRENCY_RE = re.compile(r'(\$\s?\d{1,3}(?:[,\d]{3})*(?:\.\d{1,2})?|\d+(?:\.\d{2})?\s?(?:usd|eur|aed|gbp)|(?:₹|£|€))', flags=re.I)

AMOUNT_WORDS = ["invoice", "wire transfer", "transfer", "amount", "pay", "payment", "payable", "bank", "account", "iban"]
ACTION_WORDS = ["send", "transfer", "wire", "pay", "release", "approve", "reset", "click", "login", "provide", "share"]
AUTH_WORDS = ["ceo", "cfo", "hr", "manager", "director", "cto", "chief", "vp", "president"]

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def likely_homoglyph(domain: str) -> bool:
    if any(ch.isdigit() for ch in domain) and any(ch.isalpha() for ch in domain):
        return True
    if domain.count("-") >= 2:
        return True
    if len(domain.split(".")) > 3:
        return True
    return False

def analyze_triggers_scored(text: str):
    """
    Returns:
      - triggers: list(dict{id,label,why})
      - trigger_score: float 0..1
      - ceo_pattern: bool
      - urls, emails: lists
    """
    text_lower = text.lower()
    triggers = []
    score = 0.0

    # urgency
    if any(w in text_lower for w in ["immediate", "urgent", "asap", "24 hours", "deadline", "now", "today"]):
        triggers.append({"id":"urgency","label":"Urgency", "why":"Uses time pressure words"})
        score += 0.12

    # financial keywords
    if any(w in text_lower for w in AMOUNT_WORDS):
        triggers.append({"id":"financial","label":"Financial Request", "why":"Mentions payments or bank actions"})
        score += 0.12

    # authority
    if any(w in text_lower for w in AUTH_WORDS):
        triggers.append({"id":"authority","label":"Authority Appeal", "why":"Mentions senior roles"})
        score += 0.10

    # explicit currency / amounts
    m = CURRENCY_RE.search(text)
    if m:
        amount_snip = m.group(0)
        triggers.append({"id":"amount","label":"Money Amount", "why":f"Detected amount/currency: {amount_snip}"})
        score += 0.18

    # action + financial combination
    if any(a in text_lower for a in ACTION_WORDS) and any(mw in text_lower for mw in AMOUNT_WORDS):
        triggers.append({"id":"action_fin","label":"Action + Financial", "why":"Explicit action requested for a payment or transfer"})
        score += 0.18

    # urls found
    urls = URL_RE.findall(text)
    if urls:
        triggers.append({"id":"urls","label":"External URL(s)", "why":f"Found {len(urls)} link(s)"})
        score += 0.05
        for u in urls[:3]:
            dom = extract_domain(u)
            if dom and likely_homoglyph(dom):
                triggers.append({"id":"homoglyph","label":"Suspicious Link Domain", "why":f"Suspicious domain pattern: {dom}"})
                score += 0.10

    # email addresses
    emails = EMAIL_RE.findall(text)
    if emails:
        triggers.append({"id":"emails","label":"Email Address(es)", "why":f"Found {len(emails)} address(es)"})
        if len(emails) > 1:
            score += 0.04

    # CEO-style pattern (authority + action + money) => strong flag
    ceo_pattern = False
    if any(a in text_lower for a in AUTH_WORDS) and any(m in text_lower for m in AMOUNT_WORDS) and any(v in text_lower for v in ACTION_WORDS):
        triggers.append({"id":"ceo_fraud","label":"Executive Payment Pattern", "why":"Authority + Payment + Action detected — common in spear phishing"})
        score += 0.30
        ceo_pattern = True

    # clamp
    if score > 1.0:
        score = 1.0

    return {
        "triggers": triggers,
        "trigger_score": round(score, 4),
        "ceo_pattern": ceo_pattern,
        "urls": urls,
        "emails": emails
    }

# ---------------------------
# Composite risk function
# ---------------------------
def composite_risk(model_prob: float, trigger_score: float, alpha=0.7, beta=0.3):
    """
    Combine model probability (0..1) with trigger_score (0..1).
    alpha + beta ~= 1.0. Default keeps model dominant but gives heuristics weight.
    """
    combined = alpha * float(model_prob) + beta * float(trigger_score)
    # if triggers strong and model unsure, boost moderately
    if trigger_score > 0.35 and model_prob < 0.6:
        combined = min(1.0, combined + 0.20 * trigger_score)
    return round(float(combined), 4)

# ---------------------------
# Visual: gauge builder
# ---------------------------
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

# ---------------------------
# Sidebar & header
# ---------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/000000/security-checked.png", width=48)
    st.markdown("## SentinEL Defense")
    st.markdown("A professional email risk scanner — simple for everyone, powerful for analysts.")
    st.markdown("---")
    env = st.selectbox("Environment", ["Demo", "Production"])
    st.markdown("---")
    st.caption("Model: DistilBERT (fine-tuned)")
    st.write(" ")

# ---------------------------
# Top navigation (tabs)
# ---------------------------
selected = option_menu(
    menu_title=None,
    options=["Simple Scanner", "Advanced Tools", "Batch Scan", "Forensics", "Docs"],
    icons=["search", "gear", "file-earmark-text", "clipboard-data", "book"],
    default_index=0,
    orientation="horizontal",
)

# ---------------------------
# SIMPLE SCANNER - Minimal for normal users
# ---------------------------
if selected == "Simple Scanner":
    st.header("SentinEL — Quick Email Scan")
    st.markdown("Paste the email text (or upload a .eml/.txt in Advanced Tools). Click **Scan**. This view is for everyday users — minimal, clear, and safe.")

    col1, col2 = st.columns([2, 1])
    with col1:
        email_text = st.text_area(
            "Paste email content here:",
            height=300,
            placeholder="Subject: Urgent Invoice...\n\nHi Sarah, please process this payment...",
        )
        scan_btn = st.button("🔎 Scan Email", key="scan_simple")
    with col2:
        st.markdown("### Quick Tips")
        st.markdown("- Don't paste attachments. Replace sensitive numbers with `[REDACTED]`.")
        st.markdown("- Use Advanced Tools for uploads, batch scans, and forensic details.")

    if scan_btn:
        if not email_text or not email_text.strip():
            st.warning("Please paste an email to scan.")
        else:
            with st.spinner("Analyzing..."):
                text = safe_extract_text_from_eml(email_text)
                uid = simple_hash(text + str(datetime.utcnow()))
                # model inference
                try:
                    prob_model = predict_probability(text)
                except Exception as e:
                    st.error(f"Model unavailable: {e}")
                    prob_model = 0.0

                # triggers & composite
                analysis = analyze_triggers_scored(text)
                trigger_score = analysis["trigger_score"]
                combined = composite_risk(prob_model, trigger_score, alpha=0.7, beta=0.3)
                percent = round(combined * 100, 2)

                # show gauge & verdict
                st.plotly_chart(create_gauge(percent), use_container_width=True)

                threshold = 0.5
                if combined >= threshold:
                    st.markdown("<div class='danger-box'><h3>🚨 PHISHING RISK — DO NOT TAKE ACTION</h3><p>This message contains suspicious indicators. See key reasons below.</p></div>", unsafe_allow_html=True)
                    st.markdown("**Top reasons:**")
                    for t in analysis["triggers"][:3]:
                        st.markdown(f"- **{t['label']}** — {t['why']}")
                    if analysis["ceo_pattern"]:
                        st.warning("Special rule triggered: Authority + Payment + Action — matches common CEO fraud.")
                else:
                    st.markdown("<div class='safe-box'><h3>✅ Likely Safe</h3><p>No strong combined indicators detected. If unsure, open Advanced Tools.</p></div>", unsafe_allow_html=True)
                    if analysis["triggers"]:
                        st.markdown("**Minor flags (non-blocking):**")
                        for t in analysis["triggers"][:3]:
                            st.markdown(f"- {t['label']}: {t['why']}")

                # simple action buttons
                st.markdown("---")
                if st.button("Download scan report (JSON)"):
                    report = {
                        "id": uid,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "model_prob": prob_model,
                        "trigger_score": trigger_score,
                        "combined_prob": combined,
                        "triggers": analysis["triggers"],
                    }
                    b64 = base64.b64encode(json.dumps(report, indent=2).encode()).decode()
                    href = f"data:application/json;base64,{b64}"
                    st.markdown(f"[Download report]({href})", unsafe_allow_html=True)

                if st.button("This was a mistake / report example"):
                    report = {
                        "text": text,
                        "model_prob": prob_model,
                        "trigger_score": trigger_score,
                        "combined_prob": combined,
                        "triggers": analysis["triggers"],
                    }
                    st.download_button("Download example JSON", data=json.dumps(report, indent=2), file_name="sentinel_mistake.json")
                    st.success("Thanks — example exported for retraining.")

# ---------------------------
# ADVANCED TOOLS
# ---------------------------
elif selected == "Advanced Tools":
    st.header("Advanced Tools")
    st.markdown("Upload emails, adjust thresholds, and run more detailed scans. This area is intended for analysts.")

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.expander("Paste or upload (.eml/.txt)"):
            raw_input = st.text_area("Email content or full .eml:", height=250)
            uploaded = st.file_uploader("or upload file (.eml, .txt)", type=["eml", "txt"])
            if uploaded is not None:
                raw_input = uploaded.getvalue().decode("utf-8", errors="replace")
    with col2:
        st.markdown("Settings")
        threshold = st.slider("Risk threshold (combined prob)", 0.0, 1.0, 0.5, 0.01)
        alpha = st.slider("Model weight (alpha)", 0.0, 1.0, 0.7, 0.05)
        beta = 1.0 - alpha
        st.write(f"Heuristic weight (beta) = {beta:.2f}")
        show_traces = st.checkbox("Show forensic traces (URLs, emails)", value=True)

    if st.button("Run advanced scan"):
        if not raw_input or not raw_input.strip():
            st.warning("Please paste or upload an email to scan.")
        else:
            with st.spinner("Running advanced analysis..."):
                text = safe_extract_text_from_eml(raw_input)
                try:
                    prob_model = predict_probability(text)
                except Exception as e:
                    st.error(f"Model error: {e}")
                    prob_model = 0.0

                analysis = analyze_triggers_scored(text)
                combined = composite_risk(prob_model, analysis["trigger_score"], alpha=alpha, beta=beta)
                percent = round(combined * 100, 2)

                st.plotly_chart(create_gauge(percent), use_container_width=True)
                if combined >= threshold:
                    st.markdown("<div class='danger-box'><h3>🚨 PHISHING RISK</h3><p>Combined analysis suggests this message is risky.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='safe-box'><h3>✅ Likely Safe</h3><p>Combined analysis does not indicate strong risk.</p></div>", unsafe_allow_html=True)

                st.markdown("**Model probability (raw):** {:.2f}".format(prob_model))
                st.markdown("**Trigger score:** {:.2f}".format(analysis["trigger_score"]))
                st.markdown("**Top triggers:**")
                for t in analysis["triggers"]:
                    st.markdown(f"- **{t['label']}** — {t['why']}")

                if show_traces:
                    st.markdown("---")
                    st.markdown("**Forensic traces**")
                    st.write(f"Links found: {len(analysis['urls'])}")
                    for u in analysis['urls'][:10]:
                        st.text(u)
                    st.write(f"Email addresses found: {len(analysis['emails'])}")
                    for e in analysis['emails'][:10]:
                        st.text(e)

                # export
                report = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "model_prob": prob_model,
                    "trigger_score": analysis["trigger_score"],
                    "combined_prob": combined,
                    "triggers": analysis["triggers"],
                }
                b64 = base64.b64encode(json.dumps(report, indent=2).encode()).decode()
                st.markdown(f"[Download detailed report](data:application/json;base64,{b64})", unsafe_allow_html=True)

# ---------------------------
# BATCH SCAN
# ---------------------------
elif selected == "Batch Scan":
    st.header("Batch Scan — CSV Input")
    st.markdown("Upload a CSV with a column named `text` containing email bodies. The app will annotate each row with a phishing probability.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None
        if df is not None:
            if "text" not in df.columns:
                st.error("CSV must contain a `text` column with email bodies.")
            else:
                st.info(f"Running predictions on {len(df)} rows. This may take time.")
                probs = []
                triggers_list = []
                for i, row in df.iterrows():
                    txt = str(row["text"])[:4000]
                    try:
                        p = predict_probability(txt)
                    except Exception:
                        p = None
                    if p is None:
                        probs.append(None)
                        triggers_list.append(None)
                        continue
                    analysis = analyze_triggers_scored(txt)
                    combined = composite_risk(p, analysis["trigger_score"])
                    probs.append(combined)
                    triggers_list.append(analysis["triggers"])
                df["phish_prob"] = probs
                st.success("Batch scan complete — preview below")
                st.dataframe(df.head(200))
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                st.markdown(f"[Download annotated CSV](data:text/csv;base64,{b64})", unsafe_allow_html=True)

# ---------------------------
# FORENSICS (Explainability + Analyst tools)
# ---------------------------
elif selected == "Forensics":
    st.header("Forensics & Explainability (Analyst Mode)")
    st.markdown("This page is for technical review of suspicious emails. Use it to inspect why the model made a decision, export examples for retraining, and view raw traces.")

    ex_text = st.text_area("Paste a suspicious example (or upload in Advanced Tools):", height=250)
    explain_btn = st.button("Analyze example")

    if explain_btn:
        if not ex_text.strip():
            st.warning("Paste some sample text first.")
        else:
            text = safe_extract_text_from_eml(ex_text)
            try:
                prob_model = predict_probability(text)
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                prob_model = 0.0

            analysis = analyze_triggers_scored(text)
            combined = composite_risk(prob_model, analysis["trigger_score"])
            st.markdown("**Summary**")
            st.write(f"- Model (raw) phish probability: {prob_model:.4f}")
            st.write(f"- Trigger score: {analysis['trigger_score']:.4f}")
            st.write(f"- Combined probability: {combined:.4f}")

            st.markdown("---")
            st.markdown("**Top heuristic triggers**")
            for t in analysis["triggers"]:
                st.write(f"- {t['label']}: {t['why']}")

            st.markdown("---")
            st.markdown("**Forensic traces**")
            st.write(f"Links found: {len(analysis['urls'])}")
            for u in analysis['urls'][:50]:
                st.text(u)
            st.write(f"Emails found: {len(analysis['emails'])}")
            for e in analysis['emails'][:50]:
                st.text(e)

            # token preview (safe, short)
            st.markdown("---")
            st.markdown("**Token preview (first 120 tokens)**")
            try:
                tokens = tokenizer.tokenize(text)[:120]
                st.text(" ".join(tokens))
            except Exception:
                st.info("Token preview unavailable (tokenizer not loaded).")

            # download example for active learning
            if st.button("Export example for retraining"):
                report = {
                    "text": text,
                    "model_prob": prob_model,
                    "trigger_score": analysis["trigger_score"],
                    "combined_prob": combined,
                    "triggers": analysis["triggers"],
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                st.download_button("Download JSON", data=json.dumps(report, indent=2), file_name="forensic_example.json")
                st.success("Example exported. Add to your retraining dataset.")

# ---------------------------
# DOCS / Responsible Use
# ---------------------------
elif selected == "Docs":
    st.header("Documentation & Responsible Use")
    st.markdown("""
    **About SentinEL**

    SentinEL is a research-grade spear-phishing risk scanner built for demonstration and analysis.
    It combines a fine-tuned DistilBERT model with simple heuristic rules to better detect targeted attacks.

    **Important**
    - This tool is research/educational. For production use, integrate into a secure gateway with SPF/DKIM/DMARC checks and URL sandboxing.
    - Do not paste highly confidential information here in public environments.
    """)
    st.markdown("**Quick tips for analysts:**")
    st.markdown("- Prioritize recall for spear-phishing detection (false negatives are costly).")
    st.markdown("- Use the Export button in Forensics to collect misses for active learning.")
    st.markdown("- Maintain a small separate red-team test set (50-200 spear emails) to evaluate recall.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Built by Muhsina — MSc Cyber Security | For research & education purposes.")
