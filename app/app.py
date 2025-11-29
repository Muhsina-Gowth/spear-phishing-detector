# app.py - SentinEL (Enhanced hybrid scoring for broad spear-phishing detection)
# UI-only changes: stronger, broader heuristics and presets (Normal/High/Aggressive).
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
# Page config
# ---------------------------
st.set_page_config(
    page_title="SentinEL | Spear Phishing Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---------------------------
# Global CSS (UI / accessibility)
# ---------------------------
st.markdown("""
<style>
/* --- Layout & page background --- */
.stApp {
  background: #f6f8fb;
  color: #0f172a;
  font-family: "Inter", "Segoe UI", Roboto, -apple-system, 'Helvetica Neue', Arial;
}

/* --- Card style --- */
.card {
  background: #ffffff;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 10px 30px rgba(14,30,37,0.06);
  margin-bottom: 18px;
}

/* --- Header / title --- */
.header-row { display:flex; align-items:center; gap:16px; }
.app-title { font-size:22px; font-weight:700; color:#0b1220; margin:0; }
.app-sub { color:#475569; font-size:13px; margin:0; }

/* --- Buttons --- */
.stButton>button {
  background: #0b61ff;
  color: #fff;
  border-radius: 10px;
  padding: 10px 14px;
  border: none;
  font-weight:600;
  box-shadow: 0 6px 18px rgba(11,97,255,0.12);
}
.stButton>button:focus { outline: 3px solid rgba(11,97,255,0.18); }

/* Danger & safe boxes */
.danger-box { background: #fff5f5; border-left: 6px solid #ef4444; padding:14px; border-radius:8px; color:#7f1d1d; }
.safe-box { background:#f0fdf4; border-left: 6px solid #10b981; padding:14px; border-radius:8px; color:#064e3b; }

/* Minor badges & hints */
.hint { color:#64748b; font-size:13px; }
.kv { font-weight:600; color:#0b1220; }

/* Focus and keyboard accessibility for interactive elements */
[data-baseweb="select"] { outline: none; }
a, button, input, textarea { -webkit-tap-highlight-color: rgba(0,0,0,0); }

/* Small responsive tweaks */
@media (max-width: 768px) {
  .app-title { font-size:18px; }
  .card { padding:14px; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utilities: hashing + safe extraction + redaction
# ---------------------------
def simple_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def safe_extract_text_from_eml(raw: str) -> str:
    """Small .eml/plaintext extractor; removes tags and returns text body if present."""
    parts = raw.split("\n\n", 1)
    candidate = parts[1].strip() if len(parts) > 1 else raw
    candidate = re.sub(r"<style[\\s\\S]*?</style>", "", candidate, flags=re.I)
    candidate = re.sub(r"<[^>]+>", "", candidate)
    candidate = re.sub(r"\r\n", "\n", candidate)
    return candidate.strip()

def redact_pii_for_export(text: str) -> str:
    """Simple redaction: emails and long digit sequences."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\b\d{6,}\b", "[REDACTED_NUMBER]", text)
    text = re.sub(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", "[REDACTED_IBAN]", text)
    return text

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
    except Exception:
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
    return float(probs[1])

# ---------------------------
# Trigger scoring functions (unchanged detection logic - expanded keywords)
# ---------------------------
URL_RE = re.compile(r"https?://[^\s)'\"]+", flags=re.I)
SHORT_URL_RE = re.compile(r"\b(?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|buff\.ly|rb\.gy|tiny\.cc)\b", flags=re.I)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+", flags=re.I)
CURRENCY_RE = re.compile(r'(\$\s?\d{1,3}(?:[,\d]{3})*(?:\.\d{1,2})?|\d+(?:\.\d{2})?\s?(?:usd|eur|aed|gbp)|(?:₹|£|€))', flags=re.I)

AMOUNT_WORDS = ["invoice", "wire transfer", "transfer", "amount", "pay", "payment", "payable", "bank", "account", "iban", "remit", "deposit"]
ACTION_WORDS = ["send", "transfer", "wire", "pay", "release", "approve", "reset", "click", "login", "provide", "share", "confirm", "verify"]
AUTH_WORDS = ["ceo", "cfo", "hr", "manager", "director", "cto", "chief", "vp", "president", "owner"]
CREDENTIAL_WORDS = ["password", "credentials", "login", "verify account", "reset password", "sign in", "sign-in", "authenticate"]
ATTACHMENT_WORDS = ["attached", "attachment", "invoice attached", "see attached", "download attachment"]
BRANDING_GREETINGS = ["dear", "hi", "hello", "greetings"]

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def likely_homoglyph(domain: str) -> bool:
    if not domain:
        return False
    if any(ch.isdigit() for ch in domain) and any(ch.isalpha() for ch in domain):
        return True
    if domain.count("-") >= 2:
        return True
    if len(domain.split(".")) > 3:
        return True
    # repeated characters or suspicious TLDs can also be red flags (basic)
    if re.search(r"[^\w\.]\1", domain):
        return True
    return False

def analyze_triggers_scored(text: str):
    """
    Heuristic detector collects many small signals and produces a trigger list + trigger_score.
    """
    text_lower = text.lower()
    triggers = []
    score = 0.0

    # urgency
    if any(w in text_lower for w in ["immediate", "urgent", "asap", "24 hours", "deadline", "now", "today", "urgent action required"]):
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

    # credential-related words
    if any(w in text_lower for w in CREDENTIAL_WORDS):
        triggers.append({"id":"credential","label":"Credential Request", "why":"Asks for credentials or login/reset actions"})
        score += 0.16

    # attachment mentions
    if any(w in text_lower for w in ATTACHMENT_WORDS):
        triggers.append({"id":"attachment","label":"Attachment Mention", "why":"References attachments which can be malicious"})
        score += 0.08

    # currency / amounts
    m = CURRENCY_RE.search(text)
    if m:
        amount_snip = m.group(0)
        triggers.append({"id":"amount","label":"Money Amount", "why":f"Detected amount/currency: {amount_snip}"})
        score += 0.18

    # action + financial
    if any(a in text_lower for a in ACTION_WORDS) and any(mw in text_lower for mw in AMOUNT_WORDS):
        triggers.append({"id":"action_fin","label":"Action + Financial", "why":"Explicit action requested for a payment or transfer"})
        score += 0.18

    # urls
    urls = URL_RE.findall(text)
    if urls:
        triggers.append({"id":"urls","label":"External URL(s)", "why":f"Found {len(urls)} link(s)"})
        score += 0.06
        # detect short link services inside text too
        if SHORT_URL_RE.search(text):
            triggers.append({"id":"short_url","label":"Shortened Link", "why":"Shortened URL found (often used to hide destinations)"})
            score += 0.12
        # homoglyph detection
        for u in urls[:5]:
            dom = extract_domain(u)
            if dom and likely_homoglyph(dom):
                triggers.append({"id":"homoglyph","label":"Suspicious Link Domain", "why":f"Suspicious domain pattern: {dom}"})
                score += 0.12

    # emails
    emails = EMAIL_RE.findall(text)
    if emails:
        triggers.append({"id":"emails","label":"Email Address(es)", "why":f"Found {len(emails)} address(es)"})
        if len(emails) > 1:
            score += 0.04

    # greeting vs branding mismatch / generic greetings often used in phishing
    # if opening has "Dear User" or "Dear Customer" etc, that's suspicious
    if re.search(r"\bdear (customer|user|client|employee|member)\b", text_lower):
        triggers.append({"id":"generic_greeting","label":"Generic Greeting", "why":"Generic salutations (e.g., 'Dear user' / 'Dear customer') used"})
        score += 0.08

    # sender/reply-to mismatch hints (best-effort: look for "From:" or "Reply-To:" lines in pasted raw text)
    m_from = re.search(r"from:\s*([^\n\r<]+@[\w\.-]+)", text, flags=re.I)
    m_reply = re.search(r"reply-?to:\s*([^\n\r<]+@[\w\.-]+)", text, flags=re.I)
    if m_from and m_reply and m_from.group(1).lower() != m_reply.group(1).lower():
        triggers.append({"id":"reply_mismatch","label":"Reply-To Mismatch", "why":"From: and Reply-To: differ (possible spoofing)"})
        score += 0.14

    # CEO-style pattern (authority + action + money) gets a notable boost but not forced override
    ceo_pattern = False
    if any(a in text_lower for a in AUTH_WORDS) and any(m in text_lower for m in AMOUNT_WORDS) and any(v in text_lower for v in ACTION_WORDS):
        triggers.append({"id":"ceo_fraud","label":"Executive Payment Pattern", "why":"Authority + Payment + Action detected — common in spear phishing"})
        score += 0.28
        ceo_pattern = True

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
# Legacy composite risk (kept)
# ---------------------------
def composite_risk(model_prob: float, trigger_score: float, alpha=0.7, beta=0.3):
    combined = alpha * float(model_prob) + beta * float(trigger_score)
    if trigger_score > 0.35 and model_prob < 0.6:
        combined = min(1.0, combined + 0.20 * trigger_score)
    return round(float(combined), 4)

# ---------------------------
# Enhanced rule-based final scoring (stronger, broader)
# ---------------------------
def get_presets(sensitivity: str):
    """
    Returns (alpha, beta, threshold, rule_boost_factor)
    More aggressive presets give heuristics more influence.
    """
    s = (sensitivity or "Normal").lower()
    if s == "aggressive":
        return 0.45, 0.55, 0.28, 0.60   # heuristics dominate strongly
    if s == "high":
        return 0.55, 0.45, 0.33, 0.45   # balanced, leaning heuristics
    # default Normal: model still matters but heuristics stronger than before
    return 0.62, 0.38, 0.36, 0.35

def _compute_trigger_severity(analysis: dict):
    """
    Convert analysis['triggers'] and other forensic signals into a 0..1 severity score.
    Uses many signals: amount, action, credential, short links, homoglyphs, attachments, greeting mismatch etc.
    """
    tlist = analysis.get("triggers", []) or []
    urls = analysis.get("urls", []) or []
    emails = analysis.get("emails", []) or []
    ceo = bool(analysis.get("ceo_pattern", False))

    base = float(analysis.get("trigger_score", 0.0))

    # link-related
    link_bonus = 0.0
    if urls:
        link_bonus += 0.10
        # extra if likely homoglyph domains or shortened URLs
        for u in urls[:6]:
            dom = extract_domain(u)
            if dom and likely_homoglyph(dom):
                link_bonus += 0.12
        # detect short link strings already flagged as triggers (additional)
        # this is partly handled in analyze_triggers_scored, but add small extra
        if any(SHORT_URL_RE.search(u) for u in urls):
            link_bonus += 0.08

    # multi-signal bonus (more triggers → much higher suspicion)
    multi_bonus = 0.0
    ntrig = len(tlist)
    if ntrig >= 2:
        multi_bonus += 0.07 * min(ntrig, 6)  # up to +0.42
    # additional bump for explicit money+action or credential flow
    if any(t.get("id") in ("action_fin", "amount", "credential") for t in tlist):
        multi_bonus += 0.18

    # attachments boost
    attach_bonus = 0.0
    if any(t.get("id") == "attachment" for t in tlist):
        attach_bonus += 0.10

    # CEO bump
    ceo_bonus = 0.18 if ceo else 0.0

    # multiple emails (BCC style) small bump
    email_bonus = 0.04 if len(emails) > 1 else 0.0

    severity = base + link_bonus + multi_bonus + attach_bonus + ceo_bonus + email_bonus
    if severity > 1.0:
        severity = 1.0
    return round(float(severity), 4)

def compute_final_score(model_prob: float, analysis: dict, sensitivity: str = "Normal"):
    """
    Returns (final_combined, model_prob, trigger_score, severity)
    Combines model_prob with structured heuristic severity and a sensitivity-based boost.
    """
    alpha, beta, base_threshold, rule_boost_factor = get_presets(sensitivity)

    model_prob = float(model_prob or 0.0)
    trigger_score = float(analysis.get("trigger_score", 0.0))

    base_combined = alpha * model_prob + beta * trigger_score

    severity = _compute_trigger_severity(analysis)

    # Stronger boost proportional to severity and preset factor
    if severity > 0:
        boost = rule_boost_factor * severity
        final = base_combined + boost
    else:
        final = base_combined

    # If model is near zero but severity extremely high, ensure a high floor
    if model_prob < 0.02 and severity >= 0.75:
        final = max(final, 0.85)

    # clamp
    final = min(0.995, max(0.0, final))

    return round(final, 4), round(model_prob, 4), round(trigger_score, 4), round(severity, 4)

# ---------------------------
# Gauge builder
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
# Sidebar (simplified)
# ---------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/000000/security-checked.png", width=48)
    st.markdown("## SentinEL Defense")
    st.markdown("A professional email risk scanner — simple for everyone, powerful for analysts.")
    st.markdown("---")
    st.caption("Model: DistilBERT (fine-tuned)")
    st.write(" ")

# ---------------------------
# Header (branding + onboarding)
# ---------------------------
st.markdown('<div class="card header-row">', unsafe_allow_html=True)
col_a, col_b = st.columns([0.08, 0.92])
with col_a:
    st.image("https://img.icons8.com/fluency/48/000000/security-checked.png", width=44)
with col_b:
    st.markdown('<p class="app-title">SentinEL — Email Risk Scanner</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-sub">Simple, professional, and privacy-aware. Paste an email and get a clear recommendation.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if "seen_onboard" not in st.session_state:
    st.session_state["seen_onboard"] = False

if not st.session_state["seen_onboard"]:
    st.info("Welcome — paste an email and click Scan. Use Advanced Tools only if you're comfortable viewing internal details.")
    if st.button("Don't show this again"):
        st.session_state["seen_onboard"] = True

# ---------------------------
# Top navigation
# ---------------------------
selected = option_menu(
    menu_title=None,
    options=["Simple Scanner", "Advanced Tools", "Batch Scan", "Docs"],
    icons=["search", "gear", "file-earmark-text", "book"],
    default_index=0,
    orientation="horizontal",
)

# ---------------------------
# SIMPLE SCANNER
# ---------------------------
if selected == "Simple Scanner":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Quick Scan")
    st.markdown('<p class="hint">Paste the email body below (do not paste attachments). Click <strong>Scan</strong>.</p>', unsafe_allow_html=True)

    example_col1, example_col2 = st.columns([1,1])
    with example_col1:
        if st.button("Try example: CEO fraud (demo)"):
            st.session_state['quick_example'] = ("Subject: URGENT - Wire Transfer\n\nHi, I'm in a meeting. Please wire $5,000 to account 123-456-789 now. Thanks.")
    with example_col2:
        if st.button("Try example: Credential phish (demo)"):
            st.session_state['quick_example'] = ("Subject: Action required: Verify your account\n\nDear user, please verify your account by visiting http://bit.ly/verify-now and entering your credentials.")
    email_text = st.text_area("Paste email content", value=st.session_state.get('quick_example', ""), height=300, placeholder="Subject: ...")
    scan_btn = st.button("🔎 Scan Email")
    st.markdown('</div>', unsafe_allow_html=True)

    if scan_btn:
        if not email_text or not email_text.strip():
            st.warning("Please paste an email to scan.")
        else:
            with st.spinner("Analyzing..."):
                text = safe_extract_text_from_eml(email_text)
                uid = simple_hash(text + str(datetime.utcnow()))
                try:
                    prob_model = predict_probability(text)
                except Exception as e:
                    st.error(f"Model unavailable: {e}")
                    prob_model = 0.0

                analysis = analyze_triggers_scored(text)
                combined, prob_model_r, trigger_score_r, severity = compute_final_score(prob_model, analysis, sensitivity="Normal")
                percent = round(combined * 100, 2)
                threshold = 0.36  # Normal quick-scan threshold (slightly more sensitive)

                st.plotly_chart(create_gauge(percent), use_container_width=True)
                st.markdown(f"**Model (raw):** {prob_model_r} &nbsp;&nbsp; **Trigger score:** {trigger_score_r} &nbsp;&nbsp; **Heuristic severity:** {severity}")

                if combined >= threshold:
                    st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                    st.markdown('<h3 style="margin:0">🚨 This email is suspicious — do NOT act</h3>', unsafe_allow_html=True)
                    st.markdown('<p class="hint" style="margin:6px 0 0 0">Short reason: <span class="kv">{}</span></p>'.format(
                        ", ".join([t['label'] for t in analysis['triggers'][:3]]) if analysis['triggers'] else "Suspicious patterns detected"), unsafe_allow_html=True)
                    st.markdown('<p style="margin-top:8px">Suggested next steps: <strong>Do not click links</strong>. Forward to your security team or download the scan report.</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-box">', unsafe_allow_html=True)
                    st.markdown('<h3 style="margin:0">✅ Likely Safe</h3>', unsafe_allow_html=True)
                    st.markdown('<p class="hint" style="margin-top:6px">Short reason: <span class="kv">{}</span></p>'.format(
                        ", ".join([t['label'] for t in analysis['triggers'][:2]]) if analysis['triggers'] else "No strong indicators"), unsafe_allow_html=True)
                    st.markdown('<p style="margin-top:8px">If unsure, open Advanced Tools to inspect links and headers.</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                if st.button("Download scan report (JSON)"):
                    report = {
                        "id": uid,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "model_prob": prob_model_r,
                        "trigger_score": trigger_score_r,
                        "heuristic_severity": severity,
                        "combined_prob": combined,
                        "triggers": analysis["triggers"],
                    }
                    b64 = base64.b64encode(json.dumps(report, indent=2).encode()).decode()
                    href = f"data:application/json;base64,{b64}"
                    st.markdown(f"[Download report]({href})", unsafe_allow_html=True)

                if st.button("Report a missed phishing (helps improve detection)"):
                    report = {
                        "text": text,
                        "model_prob": prob_model_r,
                        "trigger_score": trigger_score_r,
                        "heuristic_severity": severity,
                        "combined_prob": combined,
                        "triggers": analysis["triggers"],
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                    st.download_button("Download example JSON", data=json.dumps(report, indent=2), file_name="sentinel_mistake.json")
                    st.success("Thanks — example exported for retraining.")

# ---------------------------
# MERGED: ADVANCED TOOLS + FORENSICS
# ---------------------------
elif selected == "Advanced Tools":
    st.header("Advanced Tools — Expert Analysis")
    st.markdown("Use this page to upload emails, tune sensitivity, and inspect forensic details. Forensic sections are collapsed by default to avoid overwhelming non-technical users.")

    col_main, col_side = st.columns([2, 1])
    with col_main:
        with st.expander("Paste email text (recommended) or upload .eml/.txt", expanded=True):
            adv_raw = st.text_area("Email content or full .eml:", height=220)
            adv_uploaded = st.file_uploader("Upload file (.eml, .txt)", type=["eml", "txt"])
            if adv_uploaded is not None and (not adv_raw or not adv_raw.strip()):
                try:
                    adv_raw = adv_uploaded.getvalue().decode("utf-8", errors="replace")
                except Exception:
                    adv_raw = None
        run_btn = st.button("Run advanced analysis")
    with col_side:
        st.markdown("**Settings**")
        sensitivity = st.selectbox("Spear sensitivity", ["Normal", "High", "Aggressive"], index=0,
                                   help="High/Aggressive increase detection sensitivity (may increase false positives).")
        show_power = st.checkbox("Show model/heuristic weights (advanced)", value=False)
        if show_power:
            adv_alpha = st.slider("Model weight (alpha)", 0.0, 1.0, 0.62, 0.05)
            adv_threshold = st.slider("Risk threshold (combined)", 0.0, 1.0, 0.36, 0.01)
            adv_beta = round(1.0 - adv_alpha, 2)
        else:
            adv_alpha, adv_beta, adv_threshold, _ = get_presets(sensitivity)
        st.markdown(f"- Heuristic weight (beta): **{round(adv_beta,2)}**")
        st.checkbox("Show forensic traces (links, emails)", value=True, key="adv_show_traces")

    if run_btn:
        if not adv_raw or not adv_raw.strip():
            st.warning("Please paste or upload an email to analyze.")
        else:
            with st.spinner("Running advanced analysis..."):
                text = safe_extract_text_from_eml(adv_raw)
                try:
                    prob_model = predict_probability(text)
                except Exception as e:
                    st.error(f"Model error: {e}")
                    prob_model = 0.0

                analysis = analyze_triggers_scored(text)
                combined, prob_model_r, trigger_score_r, severity = compute_final_score(prob_model, analysis, sensitivity=sensitivity)
                percent = round(combined * 100, 2)

                st.plotly_chart(create_gauge(percent), use_container_width=True)
                st.markdown(f"**Combined probability:** {combined:.4f} &nbsp;&nbsp; **Model (raw):** {prob_model_r:.4f} &nbsp;&nbsp; **Trigger score:** {trigger_score_r:.4f} &nbsp;&nbsp; **Heuristic severity:** {severity}")

                if combined >= adv_threshold:
                    st.markdown('<div class="danger-box"><h3>🚨 PHISHING RISK — do not act</h3><p class="hint">This combined analysis indicates risk. Expand Forensic Details to see reasons.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-box"><h3>✅ Likely Safe</h3><p class="hint">No strong combined indicators detected. Expand Forensic Details for more information.</p></div>', unsafe_allow_html=True)

                with st.expander("Forensic details (analyst only)", expanded=False):
                    st.markdown("**Top heuristic triggers**")
                    if analysis["triggers"]:
                        for t in analysis["triggers"]:
                            st.write(f"- **{t['label']}**: {t['why']}")
                    else:
                        st.write("- None detected by heuristics.")

                    st.markdown("---")
                    st.markdown("**Forensic traces**")
                    st.write(f"- Links found: {len(analysis['urls'])}")
                    if st.session_state.get("adv_show_traces", True):
                        for u in analysis['urls'][:50]:
                            st.text(u)
                    st.write(f"- Emails found: {len(analysis['emails'])}")
                    for e in analysis['emails'][:50]:
                        st.text(e)

                    with st.expander("Token preview (first 120 tokens)", expanded=False):
                        try:
                            tokens = tokenizer.tokenize(text)[:120]
                            st.text(" ".join(tokens))
                        except Exception:
                            st.info("Token preview unavailable (tokenizer not loaded).")

                with st.expander("Export & active-learning (use carefully)", expanded=False):
                    st.markdown("You can export a scan report for retraining or sharing with security. For privacy, sensitive data is not shared automatically. Only export when appropriate.")
                    if st.button("Download detailed report (JSON)"):
                        report = {
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "model_prob": prob_model_r,
                            "trigger_score": trigger_score_r,
                            "heuristic_severity": severity,
                            "combined_prob": combined,
                            "triggers": analysis["triggers"]
                        }
                        b64 = base64.b64encode(json.dumps(report, indent=2).encode()).decode()
                        st.markdown(f"[Download report](data:application/json;base64,{b64})", unsafe_allow_html=True)

                    if st.button("Export example for retraining"):
                        report = {
                            "text": text,
                            "model_prob": prob_model_r,
                            "trigger_score": trigger_score_r,
                            "heuristic_severity": severity,
                            "combined_prob": combined,
                            "triggers": analysis["triggers"],
                            "timestamp": datetime.utcnow().isoformat() + "Z"
                        }
                        st.download_button("Download JSON", data=json.dumps(report, indent=2), file_name="sentinel_forensic_example.json")
                        st.success("Example exported (remember to remove sensitive PII if necessary).")

# ---------------------------
# BATCH SCAN (updated: preview + label + export redaction + sensitivity)
# ---------------------------
elif selected == "Batch Scan":
    st.header("Batch Scan — CSV Input")
    st.markdown("Upload a CSV with a column named `text` containing email bodies. The app will annotate each row with a phishing probability and a readable label.")

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
                batch_sensitivity = st.selectbox("Batch sensitivity", ["Normal", "High", "Aggressive"], index=0)
                preset_threshold = get_presets(batch_sensitivity)[2]
                threshold = st.slider("Label threshold (combined probability >= threshold → PHISHING)", 0.0, 1.0, float(preset_threshold), 0.01)

                st.info(f"Running predictions on {len(df)} rows. This may take time; progress updates appear below.")
                probs = []
                triggers_list = []
                labels = []

                progress_bar = st.progress(0)
                total = len(df)
                for i, row in df.iterrows():
                    txt = str(row["text"])[:4000]
                    try:
                        p_model = predict_probability(txt)
                    except Exception:
                        p_model = None

                    if p_model is None:
                        combined = None
                        triggers = None
                        label = "ERROR"
                    else:
                        analysis = analyze_triggers_scored(txt)
                        combined, _, _, _ = compute_final_score(p_model, analysis, sensitivity=batch_sensitivity)
                        triggers = analysis["triggers"]
                        label = "PHISHING" if (combined is not None and combined >= threshold) else "SAFE"

                    probs.append(combined)
                    triggers_list.append(triggers)
                    labels.append(label)

                    if (i + 1) % 5 == 0 or (i + 1) == total:
                        progress_bar.progress((i + 1) / total)

                df["phish_prob"] = probs
                df["label"] = labels
                df["triggers"] = [json.dumps(t) if t is not None else "" for t in triggers_list]

                st.success("Batch scan complete — preview below")

                if "phish_prob" in df.columns:
                    df["phish_prob"] = df["phish_prob"].apply(lambda x: (round(x, 4) if (x is not None and not pd.isna(x)) else ""))

                preview_cols = []
                if "text" in df.columns:
                    preview_cols.append("text")
                if "phish_prob" in df.columns:
                    preview_cols.append("phish_prob")
                if "label" in df.columns:
                    preview_cols.append("label")

                preview_df = df[preview_cols].copy()
                if "text" in preview_df.columns:
                    preview_df["text"] = preview_df["text"].apply(lambda t: (t[:250] + "…") if isinstance(t, str) and len(t) > 300 else t)

                def highlight_row(row):
                    lab = row.get("label", "")
                    if lab == "PHISHING":
                        return ['background-color: #fff1f1'] * len(row)
                    elif lab == "SAFE":
                        return ['background-color: #f0fdf4'] * len(row)
                    else:
                        return [''] * len(row)

                try:
                    styled = preview_df.style.apply(highlight_row, axis=1)
                    st.dataframe(styled, height=420)
                except Exception:
                    st.dataframe(preview_df.head(200))

                do_redact = st.checkbox("Redact emails and long numbers in exported CSV", value=True)

                export_df = df.copy()
                if "triggers" in export_df.columns:
                    export_df["triggers"] = export_df["triggers"].apply(lambda t: t if isinstance(t, str) else json.dumps(t))

                if do_redact and "text" in export_df.columns:
                    export_df["text_redacted"] = export_df["text"].apply(redact_pii_for_export)
                    export_cols = [c for c in export_df.columns if c != "text"]
                else:
                    export_cols = list(export_df.columns)

                preferred_order = []
                if "text_redacted" in export_cols:
                    preferred_order.append("text_redacted")
                elif "text" in export_cols:
                    preferred_order.append("text")
                if "phish_prob" in export_cols:
                    preferred_order.append("phish_prob")
                if "label" in export_cols:
                    preferred_order.append("label")
                if "triggers" in export_cols:
                    preferred_order.append("triggers")
                for c in export_cols:
                    if c not in preferred_order:
                        preferred_order.append(c)
                export_df = export_df[preferred_order]

                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                b64 = base64.b64encode(csv_bytes).decode()
                download_name = f"sentinel_batch_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
                st.markdown(f"[Download annotated CSV]({'data:text/csv;base64,' + b64})", unsafe_allow_html=True)
                st.download_button("Download CSV file", data=csv_bytes, file_name=download_name, mime="text/csv")

# ---------------------------
# DOCS
# ---------------------------
elif selected == "Docs":
    st.header("Documentation & Responsible Use")
    st.markdown("""
    **About SentinEL**

    SentinEL is a research-grade spear-phishing risk scanner built for demonstration and analysis.
    It combines a fine-tuned DistilBERT model with stronger, structured heuristics to better detect targeted attacks.

    **Important**
    - This tool is research/educational. For production use, integrate into a secure gateway with SPF/DKIM/DMARC checks and URL sandboxing.
    - Do not paste highly confidential information here in public environments.
    """)
    st.markdown("**Quick tips for analysts:**")
    st.markdown("- Start with `Normal` sensitivity. Use `High`/`Aggressive` only for red-team/hunt mode.")
    st.markdown("- Export missed examples to build a focused retraining set (active learning).")
    st.markdown("- Maintain a red-team test set of real spear examples to measure recall.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Built by Muhsina — MSc Cyber Security | For research & education purposes.")
