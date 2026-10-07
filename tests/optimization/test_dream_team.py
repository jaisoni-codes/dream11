import pytest
import pandas as pd
from src.optimization.dream_team import generate_dream_team
from tests.optimization.test_ilp import _create_synthetic_pool

def test_dream_team_generation():
    df = _create_synthetic_pool()
    result = generate_dream_team(df)
    
    assert result['status'] == 'Optimal'
    
    team_df = result['team_df']
    assert len(team_df) == 11
    
    # Check C and VC logic
    assert team_df['is_captain'].sum() == 1
    assert team_df['is_vice_captain'].sum() == 1
    
    c = result['captain']
    vc = result['vice_captain']
    
    assert c != vc
    assert c in team_df['player'].values
    assert vc in team_df['player'].values
    
    # B_0 has 100 pts, should be captain
    assert c == 'B_0'
    
    c_row = team_df[team_df['player'] == c].iloc[0]
    # Points multiplier: 100 * 2.0 = 200.0
    assert c_row['optimized_points'] == 200.0

def test_actual_points_ignored():
    # If a candidate pool has 'actual_target_points', it shouldn't affect selection.
    df = _create_synthetic_pool()
    df['actual_target_points'] = 500.0 # dummy
    
    # B_0 is 100 predicted, A_10 is 60 predicted
    # Let's make B_1 actual 1000. It still shouldn't be picked over someone with higher predicted if constrained.
    
    result = generate_dream_team(df)
    team_df = result['team_df']
    
    # B_0 still captain because we only use 'predicted_fantasy_points'
    assert result['captain'] == 'B_0'
