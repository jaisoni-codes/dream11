from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import time
import os

import math
def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

from src.inference.pipeline import PreTossInferencePipeline
from src.inference.genai_explainer import GroundedGenAIExplainer

app = FastAPI(title="Dream11 AI Team Builder API")

# Load models exactly like Streamlit did
pipeline = PreTossInferencePipeline(
    fp_model_path='models/model_bundle.pkl',
    play_model_path='models/play_probability_bundle.pkl'
)

# Load data
pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
feats_df = pd.read_parquet('data/processed/player_features.parquet')
dels_df = pd.read_parquet('data/processed/deliveries.parquet')

all_teams = sorted(list(dels_df['batting_team'].dropna().unique()))

genai_layer = GroundedGenAIExplainer()

class PredictionRequest(BaseModel):
    target_date: str
    team_a: str
    team_b: str
    use_genai: bool = False

@app.get("/api/teams")
def get_teams():
    return {"teams": all_teams}

@app.post("/api/predict")
def predict(req: PredictionRequest):
    try:
        t0 = time.time()
        pool, dream_team, latencies = pipeline.predict(
            req.target_date, req.team_a, req.team_b, pms_df, feats_df, explain=True
        )
        
        if req.use_genai:
            genai_texts, genai_lat = genai_layer.explain(dream_team, dream_team.get('explanations', []))
            if genai_texts:
                dream_team['genai_explanations'] = genai_texts
            latencies['genai_explanation'] = genai_lat
            
        # Clean up dataframes for JSON serialization
        dream_team['team_df'] = dream_team['team_df'].to_dict(orient='records')
        
        return clean_nans({
            "success": True,
            "dream_team": dream_team,
            "latencies": latencies
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount static files
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")
