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
  box-shadow: 0 10px 30px rgba(14,
