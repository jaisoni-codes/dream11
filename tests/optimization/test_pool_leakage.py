import pytest
import pandas as pd
from src.optimization.pool_builder import build_reconstructed_pool, _load_dels_cache

def test_pool_leakage(monkeypatch):
    # Clear any cached deliveries from prior tests so monkeypatch takes effect
    _load_dels_cache.cache_clear()

    # Create fake pms_df and features_df
    target_date = "2024-01-05"
    
    pms_df = pd.DataFrame([
        # Historical matches
        {'match_id': 'm1', 'player': 'P1', 'date': '2024-01-01', 'fantasy_points': 50},
        # TARGET match (Leaked info)
        {'match_id': 'm2', 'player': 'P2', 'date': '2024-01-05', 'fantasy_points': 100},
    ])
    
    features_df = pd.DataFrame([
        {'match_id': 'm1', 'player': 'P1', 'date': '2024-01-01', 'role': 'BAT'},
        {'match_id': 'm2', 'player': 'P2', 'date': '2024-01-05', 'role': 'BOWL'},
    ])
    
    # Fake deliveries to map team
    dels = pd.DataFrame([
        {'match_id': 'm1', 'batting_team': 'Team A', 'batter': 'P1'},
        {'match_id': 'm2', 'batting_team': 'Team A', 'batter': 'P2'},
    ])
    dels.to_parquet('tests/optimization/fake_deliveries.parquet', index=False)
    
    # We need to mock the read_parquet call inside pool_builder
    original_read = pd.read_parquet
    
    def mocked_read(path, **kwargs):
        if 'deliveries.parquet' in str(path):
            return dels
        return original_read(path, **kwargs)
        
    monkeypatch.setattr(pd, 'read_parquet', mocked_read)
    
    pool = build_reconstructed_pool(target_date, 'Team A', 'Team B', pms_df, features_df)
    
    # P1 should be in the pool (played before 2024-01-05)
    assert 'P1' in pool['player'].values
    
    # P2 played ONLY on 2024-01-05, so they must NOT appear in the pre-toss pool
    assert 'P2' not in pool['player'].values
