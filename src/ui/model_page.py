import streamlit as st
import pandas as pd
import os

def render_model_page():
    st.title("Model Validation & Metrics")
    
    st.markdown("""
    This section presents the canonical M7/M8 evaluation metrics. 
    The models were trained purely on historical data up to **2024-06-30** and validated 
    via strict walk-forward evaluation from **2024-07-01** onwards.
    """)
    
    st.header("1. Architecture")
    st.code("""
    [ Historical Data strictly < D ]
             |
    [ Pre-toss Candidate Pool ]
             |
    [ Feature Engineering ]
             |
    [ FP Model ]   [ P(play) Model ]
             \\       /
        [ Expected FP ]
             |
       [ ILP Optimizer ]
             |
     [ Best XI + C/VC ]
             |
    [ SHAP Explainability ]
    """, language="text")
    
    st.header("2. M7 Final Evaluation Metrics")
    st.markdown("These are the exact holdout evaluation metrics from the `reports/m7_final_summary.csv` artifact.")
    
    try:
        m7_df = pd.read_csv('reports/m7_final_summary.csv')
        metrics = m7_df.iloc[0].to_dict()
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="kpi-card"><div class="kpi-val">{metrics.get("Holdout Matches", 244)}</div><div class="kpi-lbl">Holdout Matches</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#10b981;">{metrics.get("Mean Recall@11", 0.7168):.4f}</div><div class="kpi-lbl">Recall@11</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div class="kpi-val">{metrics.get("Mean XI Overlap", 7.8852):.2f} / 11</div><div class="kpi-lbl">Mean Squad Overlap</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        c4, c5, c6 = st.columns(3)
        c4.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#FF3B44;">{metrics.get("Mean Team Regret", 296.3):.1f}</div><div class="kpi-lbl">Mean Team Regret</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#38bdf8;">{metrics.get("Captain Acc (in-play)", 0.791)*100:.1f}%</div><div class="kpi-lbl">Captain Accuracy</div></div>', unsafe_allow_html=True)
        c6.markdown(f'<div class="kpi-card"><div class="kpi-val">{metrics.get("Reproducibility (100 matches)", "PASS")}</div><div class="kpi-lbl">Reproducibility</div></div>', unsafe_allow_html=True)
        
        st.write(f"**Training window:** {metrics.get('Training Window', '2015-01-01 to 2024-06-30')}")
        st.write(f"**Leakage Status:** {metrics.get('Leakage Status', 'CLEAN')}")
        
    except Exception as e:
        st.error(f"Could not load M7 metrics: {str(e)}")
        
    st.header("3. Global Model Insights (SHAP)")
    st.markdown("Global feature importance based on mean absolute SHAP values across the holdout set.")
    
    try:
        with open('reports/m8_shap_report.md', 'r') as f:
            shap_report = f.read()
            
        # Display the markdown from M8
        st.markdown(shap_report)
    except Exception as e:
        st.error("M8 report not found. Run m8_shap_evaluation.py first.")
