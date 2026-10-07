import os
import pytest
from src.models.bundle import load_model_bundle

def test_model_bundle_feature_count():
    bundle_path = 'models/model_bundle.pkl'
    if not os.path.exists(bundle_path):
        pytest.skip(f"{bundle_path} not found. Run training first.")
        
    bundle = load_model_bundle(bundle_path)
    
    # The feature pipeline generates exactly 180 columns.
    # When merged with the target 'fantasy_points' for training, the DataFrame has 181 columns.
    # We exclude 4 columns: ['match_id', 'player', 'date', 'fantasy_points']
    # 181 - 4 = 177 training features.
    
    features = bundle['features']
    assert len(features) == 177, f"Expected exactly 177 features, but got {len(features)}. If the feature pipeline was intentionally modified, update this test."
    
    # Ensure no leakage columns
    forbidden = ['fantasy_points', 'match_id', 'player', 'date']
    for f in forbidden:
        assert f not in features, f"Forbidden column {f} found in model bundle feature list!"
