import streamlit as st
import pandas as pd
from src.inference.pipeline import PreTossInferencePipeline
import time

@st.cache_resource
def load_pipeline():
    return PreTossInferencePipeline(
        fp_model_path='models/model_bundle.pkl',
        play_model_path='models/play_probability_bundle.pkl'
    )

@st.cache_data
def load_data():
    pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
    feats_df = pd.read_parquet('data/processed/player_features.parquet')
    dels_df = pd.read_parquet('data/processed/deliveries.parquet')
    return pms_df, feats_df, dels_df

def run_prediction(target_date, team_a, team_b, explain=True):
    pipeline = load_pipeline()
    pms_df, feats_df, dels_df = load_data()
    
    t0 = time.time()
    pool, dream_team, latencies = pipeline.predict(
        target_date, team_a, team_b, pms_df, feats_df, explain=explain
    )
    ui_overhead = (time.time() - t0) - latencies['total_inference']
    if 'shap_explanation' in latencies:
        ui_overhead -= latencies['shap_explanation']
        
    latencies['ui_overhead'] = max(0, ui_overhead)
    return pool, dream_team, latencies
