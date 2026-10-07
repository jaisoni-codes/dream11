import pytest
import pandas as pd
import numpy as np
from src.inference.pipeline import PreTossInferencePipeline

@pytest.fixture(scope="module")
def pipeline():
    return PreTossInferencePipeline(
        fp_model_path='models/model_bundle.pkl',
        play_model_path='models/play_probability_bundle.pkl'
    )

@pytest.fixture(scope="module")
def sample_data():
    pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
    feats_df = pd.read_parquet('data/processed/player_features.parquet')
    dels_df = pd.read_parquet('data/processed/deliveries.parquet')
    
    # Get a valid evaluation match
    eval_matches = pms_df[pms_df['date'] >= '2024-07-01'][['match_id', 'date']].drop_duplicates().sort_values('date')
    mid = eval_matches.iloc[0]['match_id']
    mdate = eval_matches.iloc[0]['date']
    
    teams = dels_df[dels_df['match_id'] == mid]['batting_team'].unique()
    return mdate, teams[0], teams[1], pms_df, feats_df

def test_shap_explanation_generation(pipeline, sample_data):
    mdate, ta, tb, pms_df, feats_df = sample_data
    
    # Run with explain=True
    pool, dream_team, latencies = pipeline.predict(mdate, ta, tb, pms_df, feats_df, explain=True)
    
    assert 'explanations' in dream_team
    explanations = dream_team['explanations']
    
    assert len(explanations) == len(pool)
    
    # Verify structure of first explanation
    exp = explanations[0]
    expected_fields = [
        "player", "team", "role", "p_play", "predicted_fp", "expected_fp",
        "top_positive_fp_factors", "top_negative_fp_factors",
        "top_positive_play_factors", "top_negative_play_factors",
        "human_readable_explanation"
    ]
    for field in expected_fields:
        assert field in exp
        
    # Verify SHAP latency is tracked
    assert 'shap_explanation' in latencies

def test_shap_explanation_determinism(pipeline, sample_data):
    mdate, ta, tb, pms_df, feats_df = sample_data
    
    # Run twice
    _, dt1, _ = pipeline.predict(mdate, ta, tb, pms_df, feats_df, explain=True)
    _, dt2, _ = pipeline.predict(mdate, ta, tb, pms_df, feats_df, explain=True)
    
    exp1 = dt1['explanations'][0]
    exp2 = dt2['explanations'][0]
    
    assert exp1['human_readable_explanation'] == exp2['human_readable_explanation']
    assert exp1['p_play'] == exp2['p_play']
    assert exp1['expected_fp'] == exp2['expected_fp']

def test_no_side_effects_on_predictions(pipeline, sample_data):
    mdate, ta, tb, pms_df, feats_df = sample_data
    
    # Run WITHOUT explain
    pool_no_exp, dt_no_exp, _ = pipeline.predict(mdate, ta, tb, pms_df, feats_df, explain=False)
    
    # Run WITH explain
    pool_exp, dt_exp, _ = pipeline.predict(mdate, ta, tb, pms_df, feats_df, explain=True)
    
    # Verify pool sizes are identical
    assert len(pool_no_exp) == len(pool_exp)
    
    # Verify expected FP is identical
    assert np.allclose(pool_no_exp['expected_fp'].values, pool_exp['expected_fp'].values)
    
    # Verify Dream Team is identical
    assert set(dt_no_exp['team_df']['player']) == set(dt_exp['team_df']['player'])
    assert dt_no_exp['captain'] == dt_exp['captain']
    assert dt_no_exp['vice_captain'] == dt_exp['vice_captain']

def test_human_readable_content(pipeline, sample_data):
    mdate, ta, tb, pms_df, feats_df = sample_data
    _, dt, _ = pipeline.predict(mdate, ta, tb, pms_df, feats_df, explain=True)
    
    exp = dt['explanations'][0]
    text = exp['human_readable_explanation']
    
    # Text should mention the models and values
    assert exp['player'] in text
    assert "predicted FP of" in text
    assert "probability of playing" in text
    assert "Final Expected FP:" in text
