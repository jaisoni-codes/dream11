import pytest
import pandas as pd
from src.optimization.ilp import optimize_team

def _create_synthetic_pool():
    # Create 22 player pool with clear predicted points
    # 11 players Team A, 11 players Team B
    
    roles = ['BAT', 'BAT', 'BAT', 'BOWL', 'BOWL', 'BOWL', 'AR', 'AR', 'WK', 'WK', 'BAT']
    
    data = []
    for i in range(11):
        data.append({
            'player': f'A_{i}',
            'team': 'Team A',
            'role': roles[i],
            'predicted_fantasy_points': 50.0 + i  # 50 to 60
        })
        data.append({
            'player': f'B_{i}',
            'team': 'Team B',
            'role': roles[i],
            'predicted_fantasy_points': 20.0 + i  # 20 to 30
        })
        
    df = pd.DataFrame(data)
    # Set one player artificially high to test deterministic objective maximization
    df.loc[df['player'] == 'B_0', 'predicted_fantasy_points'] = 100.0
    return df

def test_ilp_constraints():
    df = _create_synthetic_pool()
    result = optimize_team(df)
    
    # 1. Exactly 11 players
    assert len(result) == 11
    
    # 2. Both teams represented, max 7 per team
    team_counts = result['team'].value_counts()
    assert len(team_counts) == 2
    assert team_counts.max() <= 7
    assert team_counts.min() >= 1
    
    # 3. Role bounds
    role_counts = result['role'].value_counts()
    for r in ['BAT', 'BOWL', 'AR', 'WK']:
        assert role_counts.get(r, 0) >= 1
        assert role_counts.get(r, 0) <= 8
        
    # 4. Optimizer chooses highest predicted points
    # B_0 has 100 points, MUST be selected.
    assert 'B_0' in result['player'].values

def test_infeasible_pool_too_few():
    df = _create_synthetic_pool().head(10) # only 10 players
    with pytest.raises(ValueError, match="Minimum 11 required"):
        optimize_team(df)
        
def test_infeasible_pool_one_team():
    df = _create_synthetic_pool()
    df['team'] = 'Team A' # all same team
    with pytest.raises(ValueError, match="exactly 2 teams"):
        optimize_team(df)

def test_infeasible_pool_missing_role():
    df = _create_synthetic_pool()
    df.loc[df['role'] == 'WK', 'role'] = 'BAT' # erase WK
    with pytest.raises(ValueError, match="No players available for role WK"):
        optimize_team(df)
