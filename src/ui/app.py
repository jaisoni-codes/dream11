import streamlit as st
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ui.product_page import render_product_page
from src.ui.model_page import render_model_page

st.set_page_config(page_title="Dream11 Optimizer", layout="wide", initial_sidebar_state="expanded")

def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .main-header {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        background: linear-gradient(90deg, #FF3B44 0%, #b91c1c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .player-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
        border-left: 5px solid #FF3B44;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        transition: transform 0.2s;
    }
    .player-card:hover {
        transform: translateY(-2px);
    }
    .player-card.captain {
        border-left: 5px solid #fbbf24;
        background-color: #332b13;
    }
    .player-card.vice-captain {
        border-left: 5px solid #38bdf8;
        background-color: #162633;
    }
    .badge-role {
        background-color: #334155;
        color: #cbd5e1;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-pts {
        background-color: #FF3B44;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 800;
    }
    .kpi-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #334155;
    }
    .kpi-val {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
    }
    .kpi-lbl {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 5px;
    }
    /* Hide some default Streamlit elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def main():
    inject_custom_css()
    st.sidebar.markdown('<div style="font-size: 24px; font-weight: 800; color: #FF3B44; margin-bottom: 20px;">DREAM11 AI</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", ["🏏 Product UI (Prediction)", "📈 Model UI (Evaluation)"])

    if page == "🏏 Product UI (Prediction)":
        render_product_page()
    else:
        render_model_page()

if __name__ == "__main__":
    main()
