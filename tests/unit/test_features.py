import pytest
import pandas as pd
import numpy as np
from src.features.rolling import compute_rolling_features
from src.features.form import build_form_features

def test_ewma_calculation():
    pms = pd.DataFrame([
        {'match_id': f'm{i}', 'player': 'P1', 'date': f'2024-01-{10+i}', 'format': 'T20', 'fantasy_points': i * 10}
        for i in range(1, 6) # m1=10, m2=20, m3=30, m4=40, m5=50
    ])
    
    features = compute_rolling_features(pms)
    
    # For m5 (i=5), the history is m1..m4
    m5_feats = features[features['match_id'] == 'm5'].iloc[0]
    
    assert m5_feats['fantasy_points_last_1'] == 40
    assert m5_feats['fantasy_points_last_3'] == (20 + 30 + 40) / 3
    
    # EWMA short should be calculated and exist
    assert 'fantasy_points_ewma_short' in m5_feats
    assert not pd.isna(m5_feats['fantasy_points_ewma_short'])

def test_form_volatility():
    pms = pd.DataFrame([
        {'match_id': f'm{i}', 'player': 'P1', 'date': f'2024-01-{10+i}', 'format': 'T20', 'fantasy_points': val}
        for i, val in enumerate([10, 10, 10, 10, 10, 10]) # Stable 10 pts
    ])
    
    rolling = compute_rolling_features(pms)
    form = build_form_features(pms, rolling)
    
    # After 5 matches (so m5/m6), std deviation should be 0 because it's always 10
    m5_form = form[form['match_id'] == 'm5'].iloc[0]
    
    assert m5_form['fantasy_points_std_career'] == 0.0
