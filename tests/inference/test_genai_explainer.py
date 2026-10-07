import pytest
import os
from unittest.mock import patch, MagicMock
from src.inference.genai_explainer import GroundedGenAIExplainer
import pandas as pd

def test_genai_explainer_fallback_when_no_api_key():
    explainer = GroundedGenAIExplainer()
    explainer.model = None
    
    dream_team = {'team_df': MagicMock(), 'captain': 'A', 'vice_captain': 'B', 'total_predicted_score': 100}
    shap = [{'player': 'A', 'human_readable_explanation': 'Test M8 text'}]
    
    output, lat = explainer.explain(dream_team, shap)
    assert output is None
    assert lat == 0.0

@patch('src.inference.genai_explainer.genai.GenerativeModel')
def test_genai_explainer_hallucination_guardrails(mock_model_class):
    mock_model = MagicMock()
    explainer = GroundedGenAIExplainer()
    explainer.model = mock_model
    
    dream_team = {
        'team_df': pd.DataFrame({'player': ['Player 1', 'Player 2', 'Player 3']}),
        'captain': 'Player 1',
        'vice_captain': 'Player 2',
        'total_predicted_score': 100
    }
    
    shap = [
        {'player': 'Player 1', 'team': 'T1', 'role': 'BAT', 'p_play': 0.85, 'expected_fp': 42.5, 'human_readable_explanation': 'Fallback 1', 'top_positive_fp_factors': [], 'top_negative_fp_factors': [], 'top_positive_play_factors': []},
        {'player': 'Player 2', 'team': 'T1', 'role': 'BAT', 'p_play': 0.99, 'expected_fp': 42.5, 'human_readable_explanation': 'Fallback 2', 'top_positive_fp_factors': [], 'top_negative_fp_factors': [], 'top_positive_play_factors': []},
        {'player': 'Player 3', 'team': 'T1', 'role': 'BAT', 'p_play': 0.99, 'expected_fp': 42.5, 'human_readable_explanation': 'Fallback 3', 'top_positive_fp_factors': [], 'top_negative_fp_factors': [], 'top_positive_play_factors': []}
    ]
    
    # 1. Test "confirmed" on low p_play
    mock_model.generate_content.return_value = MagicMock(text='{"players": {"Player 1": {"summary": "Confirmed"}}}')
    out, _ = explainer.explain(dream_team, shap)
    assert out['Player 1'] == 'Fallback 1'
    
    # 2. Test forbidden word (pitch)
    mock_model.generate_content.return_value = MagicMock(text='{"players": {"Player 2": {"summary": "Great pitch today."}}}')
    out, _ = explainer.explain(dream_team, shap)
    assert out['Player 2'] == 'Fallback 2'
    
    # 3. Test missing expected FP match
    mock_model.generate_content.return_value = MagicMock(text='{"players": {"Player 3": {"fp_explanation": "expected fp is 100"}}}')
    out, _ = explainer.explain(dream_team, shap)
    assert out['Player 3'] == 'Fallback 3'
    
    # 4. Test malformed JSON
    mock_model.generate_content.return_value = MagicMock(text='{malformed_json:')
    out, _ = explainer.explain(dream_team, shap)
    assert out is None

@patch('src.inference.genai_explainer.genai.GenerativeModel')
def test_genai_explainer_valid_response(mock_model_class):
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '''{
        "team_explanation": "Good team.",
        "players": {
            "Player 1": {
                "summary": "Solid pick.",
                "why_selected": "High points.",
                "p_play_explanation": "Highly likely to play (99%).",
                "fp_explanation": "Expected FP of 42.5."
            }
        }
    }'''
    mock_model.generate_content.return_value = mock_response
    
    explainer = GroundedGenAIExplainer()
    explainer.model = mock_model
    
    dream_team = {
        'team_df': pd.DataFrame({'player': ['Player 1']}),
        'captain': 'Player 1',
        'vice_captain': 'Player 2',
        'total_predicted_score': 100
    }
    
    shap = [{
        'player': 'Player 1',
        'team': 'T1',
        'role': 'BAT',
        'p_play': 0.99,
        'expected_fp': 42.5,
        'human_readable_explanation': 'Fallback Text',
        'top_positive_fp_factors': [{'feature': 'f1', 'contribution': 1}],
        'top_negative_fp_factors': [{'feature': 'f2', 'contribution': -1}],
        'top_positive_play_factors': [{'feature': 'f3', 'contribution': 1}]
    }]
    
    output, lat = explainer.explain(dream_team, shap)
    assert "Solid pick" in output['Player 1']
    assert "Fallback Text" not in output['Player 1']
