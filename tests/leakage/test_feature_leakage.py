import pytest
import pandas as pd
from datetime import date
from src.features.rolling import compute_rolling_features
from src.features.role_opp import infer_roles_and_opportunity

def test_same_match_leakage():
    # Target match performance cannot appear in its own rolling features.
    pms = pd.DataFrame([
        {'match_id': 'm1', 'player': 'P1', 'date': '2024-01-01', 'format': 'T20', 'fantasy_points': 100},
        {'match_id': 'm2', 'player': 'P1', 'date': '2024-01-10', 'format': 'T20', 'fantasy_points': 50}
    ])
    
    features = compute_rolling_features(pms)
    
    m1_feats = features[features['match_id'] == 'm1'].iloc[0]
    # P1's first match, should have no history
    assert m1_feats['has_history'] == 0
    assert pd.isna(m1_feats['fantasy_points_last_1'])
    
    m2_feats = features[features['match_id'] == 'm2'].iloc[0]
    # P1's second match, history should ONLY be m1 (100 pts), NOT m2 (50 pts)
    assert m2_feats['has_history'] == 1
    assert m2_feats['fantasy_points_last_1'] == 100
    assert m2_feats['fantasy_points_career_mean'] == 100

def test_insufficient_history():
    pms = pd.DataFrame([
        {'match_id': 'm1', 'player': 'P1', 'date': '2024-01-01', 'format': 'T20', 'fantasy_points': 10},
        {'match_id': 'm2', 'player': 'P1', 'date': '2024-01-10', 'format': 'T20', 'fantasy_points': 20}
    ])
    
    features = compute_rolling_features(pms)
    m2_feats = features[features['match_id'] == 'm2'].iloc[0]
    
    # Check 10-match rolling for a player with 1 match of history
    # The mean should just be 10 (the single available match), not NaN (because min_periods=1)
    assert m2_feats['fantasy_points_last_10'] == 10

def test_role_temporal_leakage():
    # A future role-changing match must not alter an earlier target-date role.
    pms = pd.DataFrame([
        # Match 1: Just bats
        {'match_id': 'm1', 'player': 'P1', 'date': '2024-01-01', 'format': 'T20', 'fantasy_points': 50, 'balls_bowled': 0, 'stumpings': 0},
        # Match 2: Still bats
        {'match_id': 'm2', 'player': 'P1', 'date': '2024-01-10', 'format': 'T20', 'fantasy_points': 50, 'balls_bowled': 0, 'stumpings': 0},
        # Match 3: Keeps wicket (stumping = 1)
        {'match_id': 'm3', 'player': 'P1', 'date': '2024-02-01', 'format': 'T20', 'fantasy_points': 50, 'balls_bowled': 0, 'stumpings': 1},
        # Match 4: Next match
        {'match_id': 'm4', 'player': 'P1', 'date': '2024-02-10', 'format': 'T20', 'fantasy_points': 50, 'balls_bowled': 0, 'stumpings': 0},
    ])
    
    # We need to simulate how pipeline runs: pms -> rolling -> form -> role
    # Since rolling doesn't calculate 'stumpings_career_sum' by default unless we specify it in configs,
    # let's just manually inject it as if rolling did it, or configure it.
    # The rolling module dynamically reads configs/features.yaml, which does contain 'stumpings'.
    
    rolling = compute_rolling_features(pms)
    roles = infer_roles_and_opportunity(pms, rolling)
    
    # In match 2 (before keeping), they should NOT be WK
    m2_role = roles[roles['match_id'] == 'm2'].iloc[0]['role']
    assert m2_role != 'WK'
    
    # In match 4 (after keeping in match 3), they SHOULD be WK
    m4_role = roles[roles['match_id'] == 'm4'].iloc[0]['role']
    assert m4_role == 'WK'
