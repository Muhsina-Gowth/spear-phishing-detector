import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SentinEL | Spear Phishing Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (MODERN UI & ACCESSIBILITY) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Card Design */
    .css-card {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #0f172a; /* High contrast dark blue */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    p, li {
        color: #334155; /* Readable slate gray */
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Status Boxes (Accessible Colors) */
    .safe-box {
        background-color: #ecfdf5; 
        border-left: 5px solid #10b981; /* Strong Green */
        padding: 15px;
        border-radius: 5px;
        color: #064e3b;
    }
    .danger-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444; /* Strong Red */
        padding: 15px;
        border-radius: 5px;
        color: #7f1d1d;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_ai_model():
    model_id = "iammuhsina/spear-phishing-bert"
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_id)
        model = DistilBertForSequenceClassification.from_pretrained(model_id)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

tokenizer, model, device = load_ai_model()

# --- HELPER FUNCTIONS ---
def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Phishing Probability (%)", 'font': {'size': 18, 'color': "#334155"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': "#ef4444" if score > 50 else "#10b981"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 50], 'color': "#ecfdf5"},
                {'range': [50, 80], 'color': "#fff7ed"},
                {'range': [80, 100], 'color': "#fef2f2"}],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def analyze_triggers(text):
    triggers = []
    text_lower = text.lower()
    
    # Heuristic keywords
    urgency = ['immediate', 'urgent', 'asap', '24 hours', 'deadline', 'suspend', 'lock', 'expire']
    money = ['wire transfer', 'invoice', 'payment', 'bank', 'gift card', 'payroll', 'deposit']
    auth = ['ceo', 'admin', 'security', 'hr', 'manager', 'password', 'credential']
    
    if any(w in text_lower for w in urgency):
        triggers.append("⚠️ **Urgency & Pressure:** Uses words designed to force a quick, unthinking reaction.")
    if any(w in text_lower for w in money):
        triggers.append("💰 **Financial Request:** Mentions payments, banking, or money transfers.")
    if any(w in text_lower for w in auth):
        triggers.append("👔 **Authority / Credentials:** Mentions high-status roles or sensitive login data.")
        
    if not triggers:
        triggers.append("✅ **No Keyword Triggers:** The model relied purely on deep contextual patterns.")
    return triggers

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/grand-master-key.png", width=60)
    st.markdown("### SentinEL Defense")
    
    # Modern Option Menu
    selected = option_menu(
        menu_title=None,
        options=["Scanner Tool", "Safety Guide", "How it Works", "About"],
        icons=["shield-check", "lock", "cpu", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#1e3a8a", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#e0e7ff", "color": "#1e3a8a", "font-weight": "bold"},
        }
    )
    
    st.markdown("---")
    st.caption("Developed for MSc Cyber Security")
    st.caption(f"Engine: **Fine-Tuned BERT**")
    st.caption("v1.0.4 | Stable")

# --- PAGE: SCANNER ---
if selected == "Scanner Tool":
    st.title("🛡️ SentinEL Email Scanner")
    st.markdown("Analyze suspicious emails for **spear phishing indicators** using advanced AI.")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        # Input Card
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### 📧 Input Analysis")
        email_text = st.text_area(
            "Paste Email Content:", 
            height=300, 
            placeholder="Subject: Urgent Invoice...\n\nHi Sarah, please process this payment...",
            help="Copy and paste the body of the email here. Do not include attachments."
        )
        analyze_btn = st.button("🚀 Scan for Threats", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Results Card
        if analyze_btn and email_text.strip():
            with st.spinner("🔍 Analyzing semantic patterns..."):
                # AI Inference
                inputs = tokenizer(email_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1)
                
                phishing_prob = probs[0][1].item() * 100
                prediction = torch.argmax(probs, dim=1).item()
                
                st.markdown('<div class="css-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Scan Results")
                
                # Gauge Chart
                st.plotly_chart(create_gauge(phishing_prob), use_container_width=True)
                
                # Verdict Box
                if prediction == 1:
                    st.markdown("""
                    <div class="danger-box">
                        <h3>🚨 PHISHING DETECTED</h3>
                        <strong>Critical Threat Level</strong><br>
                        This email contains high-risk patterns consistent with spear phishing.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="safe-box">
                        <h3>✅ APPEARS LEGITIMATE</h3>
                        <strong>Low Risk Level</strong><br>
                        The language and context appear consistent with normal business communication.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("**🔎 Forensic Indicators:**")
                triggers = analyze_triggers(email_text)
                for t in triggers:
                    st.markdown(f"- {t}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        elif analyze_btn:
            st.warning("⚠️ Please enter text to scan.")
        
        else:
            # Placeholder State
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.info("👈 Paste an email and click 'Scan' to see the AI analysis here.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: SAFETY GUIDE ---
elif selected == "Safety Guide":
    st.title("🔒 Security Best Practices")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("### 🛑 Before You Scan")
    st.markdown("""
    Handling live phishing emails is dangerous. Follow these rules to stay safe:
    
    1.  **Never Click Links:** Even clicking a link to "check" it can compromise your browser.
    2.  **Use 'View Source':** In Outlook/Gmail, view the "Original Message" to copy the text safely without rendering malicious images.
    3.  **Anonymize Data:** If the email contains real names or account numbers, replace them (e.g., `[Name]`) before pasting.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: HOW IT WORKS ---
elif selected == "How it Works":
    st.title("🧠 The AI Architecture")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("""
    ### From Dictionary to Brain
    
    Traditional spam filters look for **"Bad Words"** (like "Lottery" or "Viagra"). Spear phishing bypasses this by using professional language.
    
    **SentinEL** uses a **Fine-Tuned BERT Model**.
    1.  **Pre-training:** It read Wikipedia to understand English grammar and context.
    2.  **Fine-Tuning:** We trained it on **14,000+ emails** (Enron + Phishing) to learn the difference between "Safe Business" and "Malicious Business."
    3.  **Active Learning:** We specifically taught it to catch **CEO Fraud** by generating 1,000+ targeted attack simulations.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: ABOUT ---
elif selected == "About":
    st.title("ℹ️ About the Project")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("### Detecting Spear Phishing with NLP")
    st.markdown("This tool was developed as part of a Master's Thesis to demonstrate the effectiveness of **Contextual Embeddings** over traditional sentiment analysis.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "99.21%")
    c2.metric("False Positive Rate", "< 1.0%")
    c3.metric("Architecture", "DistilBERT")
    
    st.markdown("---")
    st.markdown("**Developer:** Muhsina Gowth")
    st.markdown("**Institution:** Middlesex University")
    st.markdown("**Research Area:** Applied AI in Cyber Security")
    st.markdown('</div>', unsafe_allow_html=True)