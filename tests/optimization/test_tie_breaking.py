import pytest
import pandas as pd
from src.optimization.captain import select_captain_and_vc

def test_tie_breaking():
    # Tie breaking: predicted points DESC, role ASC, player name ASC
    df = pd.DataFrame([
        {'player': 'Zebra', 'role': 'BAT', 'predicted_fantasy_points': 50.0},
        {'player': 'Alpha', 'role': 'BAT', 'predicted_fantasy_points': 50.0},
        {'player': 'Beta', 'role': 'AR', 'predicted_fantasy_points': 50.0},
    ])
    
    result = select_captain_and_vc(df)
    
    # Expected:
    # 1. AR comes before BAT alphabetically
    # So 'Beta' (AR) should be first (Captain).
    # Next tie is between 'Alpha' and 'Zebra' (both BAT). Alpha comes first (Vice-Captain).
    
    c = result[result['is_captain'] == 1]['player'].iloc[0]
    vc = result[result['is_vice_captain'] == 1]['player'].iloc[0]
    
    assert c == 'Beta'
    assert vc == 'Alpha'
