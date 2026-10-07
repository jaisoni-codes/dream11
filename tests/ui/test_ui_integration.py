import pytest
import pandas as pd
from src.ui.utils import run_prediction, load_data

def test_ui_data_loading():
    pms_df, feats_df, dels_df = load_data()
    assert not pms_df.empty
    assert not feats_df.empty
    assert not dels_df.empty

def test_ui_prediction_wrapper():
    from src.optimization.pool_builder import _load_dels_cache
    _load_dels_cache.cache_clear()
    
    # Use the demo match
    target_date = "2024-07-01"
    team_a = "Malawi"
    team_b = "Kenya"
    
    pool, dream_team, latencies = run_prediction(target_date, team_a, team_b, explain=True)
    
    # Verify outputs
    assert not pool.empty
    assert len(dream_team['team_df']) == 11
    assert 'explanations' in dream_team
    
    # Verify explainability structure
    exp = dream_team['explanations'][0]
    assert 'player' in exp
    assert 'p_play' in exp
    assert 'expected_fp' in exp
    assert 'human_readable_explanation' in exp
    
    # Verify latencies
    assert 'total_inference' in latencies
    assert 'shap_explanation' in latencies
    assert 'ui_overhead' in latencies
    
    # C/VC present
    assert dream_team['captain'] in dream_team['team_df']['player'].values
    assert dream_team['vice_captain'] in dream_team['team_df']['player'].values

def test_ui_prediction_no_explain():
    from src.optimization.pool_builder import _load_dels_cache
    _load_dels_cache.cache_clear()
    
    target_date = "2024-07-01"
    team_a = "Malawi"
    team_b = "Kenya"
    
    pool, dream_team, latencies = run_prediction(target_date, team_a, team_b, explain=False)
    
    # Shouldn't have explanations if explain=False
    assert 'explanations' not in dream_team
    assert 'shap_explanation' not in latencies
