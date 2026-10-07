import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.optimization.pool_builder import build_reconstructed_pool
from src.evaluation.evaluate_m6 import evaluate_m6

@pytest.fixture
def base_data():
    pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
    feats_df = pd.read_parquet('data/processed/player_features.parquet')
    dels_df = pd.read_parquet('data/processed/deliveries.parquet')
    return pms_df, feats_df, dels_df

def test_1_and_6_participation_and_fp_labels_do_not_change_features(base_data):
    pms_df, feats_df, _ = base_data
    
    # Pick a random target match and team
    target_match = '1286978'
    target_date = pms_df[pms_df['match_id'] == target_match]['date'].iloc[0]
    team_a = 'Australia'
    team_b = 'Sri Lanka'
    
    # Baseline pool
    pool_base = build_reconstructed_pool(target_date, team_a, team_b, pms_df, feats_df)
    
    # Modify participation and fantasy points in PMS
    pms_mod = pms_df.copy()
    mask = pms_mod['match_id'] == target_match
    # Drop half the players
    pms_mod = pms_mod.drop(pms_mod[mask].head(5).index)
    # Change fantasy points of the rest
    pms_mod.loc[pms_mod['match_id'] == target_match, 'fantasy_points'] = 999
    
    pool_mod = build_reconstructed_pool(target_date, team_a, team_b, pms_mod, feats_df)
    
    assert pool_base.equals(pool_mod), "Modifying target match labels altered pre-toss features/pool"

def test_2_target_only_player_not_in_pool(base_data):
    pms_df, feats_df, _ = base_data
    target_match = '1286978'
    target_date = pms_df[pms_df['match_id'] == target_match]['date'].iloc[0]
    
    pms_mod = pms_df.copy()
    # Add a fake player ONLY to the target match
    new_row = pms_mod[pms_mod['match_id'] == target_match].iloc[0].copy()
    new_row['player'] = 'GHOST_PLAYER_999'
    pms_mod = pd.concat([pms_mod, pd.DataFrame([new_row])])
    
    pool = build_reconstructed_pool(target_date, 'Australia', 'Sri Lanka', pms_mod, feats_df)
    assert 'GHOST_PLAYER_999' not in pool['player'].values, "Future player leaked into pre-toss pool"

def test_3_no_feature_row_date_gte_target(base_data):
    pms_df, feats_df, _ = base_data
    target_match = '1286978'
    target_date = pms_df[pms_df['match_id'] == target_match]['date'].iloc[0]
    
    pool = build_reconstructed_pool(target_date, 'Australia', 'Sri Lanka', pms_df, feats_df)
    
    # Every row in pool is drawn from features_df, we can check if its 'date' is < target_date
    assert (pd.to_datetime(pool['date']) < pd.to_datetime(target_date)).all(), "Feature date >= target date found!"

def test_4_training_data_cutoff():
    bundle = joblib.load('models/play_probability_bundle.pkl')
    assert bundle['cutoff'] == '2024-06-30', "Model trained on invalid cutoff!"

def test_5_target_fp_not_in_play_prob_features():
    bundle = joblib.load('models/play_probability_bundle.pkl')
    features = bundle['features']
    assert 'fantasy_points' not in features, "fantasy_points leaked into prob model features!"
    assert 'actual_fantasy_points' not in features, "actual_fantasy_points leaked into prob model features!"

def test_7_8_deterministic_and_identical_pools():
    pms_df, feats_df, _ = pd.read_parquet('data/processed/player_match_stats.parquet'), pd.read_parquet('data/processed/player_features.parquet'), None
    target_date = '2022-02-15'
    pool_1 = build_reconstructed_pool(target_date, 'Australia', 'Sri Lanka', pms_df, feats_df)
    pool_2 = build_reconstructed_pool(target_date, 'Australia', 'Sri Lanka', pms_df, feats_df)
    
    assert pool_1.equals(pool_2), "Pool building is non-deterministic!"
