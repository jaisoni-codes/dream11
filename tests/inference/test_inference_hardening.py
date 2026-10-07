import pytest
import pandas as pd
import numpy as np
import os
import sys
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.inference.pipeline import PreTossInferencePipeline

@pytest.fixture(scope="module")
def pipeline():
    return PreTossInferencePipeline()

@pytest.fixture(scope="module")
def base_data():
    pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
    feats_df = pd.read_parquet('data/processed/player_features.parquet')
    return pms_df, feats_df

def test_reproducibility_deterministic_inference(pipeline, base_data):
    pms_df, feats_df = base_data
    
    # Pick a random target match date
    target_date = '2023-01-15'
    team_a = 'India'
    team_b = 'Sri Lanka'
    
    # Run twice
    pool_1, dt_1, _ = pipeline.predict(target_date, team_a, team_b, pms_df, feats_df)
    pool_2, dt_2, _ = pipeline.predict(target_date, team_a, team_b, pms_df, feats_df)
    
    # 1. Candidate pool identical
    assert pool_1['player'].equals(pool_2['player']), "Candidate pool players differ across runs!"
    
    # 2. Features and probabilities identical
    assert np.allclose(pool_1['predicted_fp_raw'], pool_2['predicted_fp_raw']), "FP predictions differ!"
    assert np.allclose(pool_1['play_prob'], pool_2['play_prob']), "P(play) differs!"
    assert np.allclose(pool_1['expected_fp'], pool_2['expected_fp']), "Expected FP differs!"
    
    # 3. Best 11 identical
    xi_1 = set(dt_1['team_df']['player'])
    xi_2 = set(dt_2['team_df']['player'])
    assert xi_1 == xi_2, "Predicted XI differs across runs!"
    
    # 4. C/VC identical
    assert dt_1['captain'] == dt_2['captain'], "Captain differs!"
    assert dt_1['vice_captain'] == dt_2['vice_captain'], "Vice Captain differs!"

def test_target_poisoning_does_not_affect_inference(pipeline, base_data):
    pms_df, feats_df = base_data
    
    # Identify a target match that exists in the dataset
    target_match = '1298150' # Example match ID
    if target_match not in pms_df['match_id'].values:
        target_match = pms_df['match_id'].iloc[-1]
        
    target_date = pms_df[pms_df['match_id'] == target_match]['date'].iloc[0]
    
    # Find teams
    target_pms = pms_df[pms_df['match_id'] == target_match]
    team_a = 'Australia' # We can just test a known team combination or just mock it.
    team_b = 'England'
    # Wait, lets pick teams dynamically so we guarantee a result
    team_a, team_b = "South Africa", "India"
    
    # Control Run
    pool_ctrl, dt_ctrl, _ = pipeline.predict(target_date, team_a, team_b, pms_df, feats_df)
    
    # Poison target match in pms_df
    pms_poisoned = pms_df.copy()
    mask = pms_poisoned['match_id'] == target_match
    # Give everyone 500 fantasy points
    pms_poisoned.loc[mask, 'fantasy_points'] = 500
    # Drop some actual players from the scorecard
    drop_idx = pms_poisoned[mask].head(3).index
    pms_poisoned = pms_poisoned.drop(drop_idx)
    
    # Poison target match in features_df (if it exists on that exact date)
    feats_poisoned = feats_df.copy()
    feats_poisoned.loc[feats_poisoned['match_id'] == target_match, 'fantasy_points_last_1'] = 999
    
    # Poisoned Run
    pool_pois, dt_pois, _ = pipeline.predict(target_date, team_a, team_b, pms_poisoned, feats_poisoned)
    
    assert pool_ctrl['player'].equals(pool_pois['player']), "Poisoning target match altered pre-toss candidates!"
    assert np.allclose(pool_ctrl['expected_fp'], pool_pois['expected_fp']), "Poisoning target match altered expected points!"
    assert set(dt_ctrl['team_df']['player']) == set(dt_pois['team_df']['player']), "Poisoning target match altered playing XI prediction!"

def test_train_inference_feature_parity(pipeline):
    fp_expected = pipeline.fp_model.features
    prob_expected = pipeline.prob_features
    
    # Ensure they are non-empty
    assert len(fp_expected) > 0
    assert len(prob_expected) > 0
    
    # Ensure no future leak features are present
    forbidden = ['actual_fantasy_points', 'actual_play', 'match_id_target']
    for f in forbidden:
        assert f not in fp_expected
        assert f not in prob_expected
