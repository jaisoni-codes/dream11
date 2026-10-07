import pytest
import pandas as pd
from src.training.walk_forward import load_data
from src.models.lgbm_fp import LGBMFantasyPointsModel

def test_target_leakage():
    # Load combined dataset
    df = load_data()
    
    # We instantiate the LGBM wrapper and prepare features
    model_wrapper = LGBMFantasyPointsModel({})
    X_feats = model_wrapper._prepare_features(df)
    
    # Ensure forbidden columns do not exist
    forbidden = ['fantasy_points', 'match_id', 'player', 'date', 'runs', 'wickets', 'catches']
    for f in forbidden:
        assert f not in X_feats.columns, f"LEAKAGE DETECTED: '{f}' is inside the feature matrix!"
