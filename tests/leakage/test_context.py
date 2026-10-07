import pytest
import pandas as pd
from datetime import date
from src.features.context import FeatureContext

def test_feature_context_leakage():
    # Setup historical data
    df = pd.DataFrame([
        {'id': 1, 'date': '2024-01-01', 'val': 10},
        {'id': 2, 'date': '2024-02-01', 'val': 20},
        {'id': 3, 'date': '2024-03-01', 'val': 30},
    ])
    
    # Case 1: Target date is after all -> all allowed
    ctx = FeatureContext('2024-04-01')
    safe_df = ctx.filter_historical_data(df)
    assert len(safe_df) == 3
    
    # Case 2: Historical match ON target date -> rejected
    ctx2 = FeatureContext('2024-03-01')
    safe_df2 = ctx2.filter_historical_data(df)
    assert len(safe_df2) == 2
    assert 3 not in safe_df2['id'].values
    
    # Case 3: Historical match AFTER target date -> rejected
    ctx3 = FeatureContext('2024-01-15')
    safe_df3 = ctx3.filter_historical_data(df)
    assert len(safe_df3) == 1
    assert 1 in safe_df3['id'].values

def test_poisoned_future():
    # Case 5: Future match must not alter rolling historical stats
    df = pd.DataFrame([
        {'id': 1, 'date': '2024-01-01', 'val': 10},
        {'id': 2, 'date': '2024-02-01', 'val': 20},
    ])
    
    ctx = FeatureContext('2024-03-01')
    features_before = ctx.filter_historical_data(df)
    
    # Poison the future
    poisoned_df = pd.concat([df, pd.DataFrame([{'id': 99, 'date': '2024-04-01', 'val': 99999}])])
    
    features_after = ctx.filter_historical_data(poisoned_df)
    
    # Assert features for '2024-03-01' remain exactly the same
    pd.testing.assert_frame_equal(features_before, features_after)

from src.utils.config import validate_cutoff, HARD_CUTOFF

def test_hard_cutoff_validation():
    # 2024-06-30 is included
    assert validate_cutoff('2024-06-30') is True
    assert validate_cutoff(date(2024, 6, 30)) is True
    
    # 2024-07-01 is excluded
    with pytest.raises(ValueError, match="Data leak: date 2024-07-01 is strictly after HARD_CUTOFF 2024-06-30"):
        validate_cutoff('2024-07-01')
        
    with pytest.raises(ValueError, match="Data leak: date 2024-07-01 is strictly after HARD_CUTOFF 2024-06-30"):
        validate_cutoff(date(2024, 7, 1))
