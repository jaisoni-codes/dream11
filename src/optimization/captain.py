import pandas as pd

def select_captain_and_vc(selected_df):
    """
    Selects captain and vice-captain deterministically.
    Captain receives 2.0x points.
    Vice-Captain receives 1.5x points.
    Tie breaking:
    1. Higher predicted points
    2. Alphabetical ordering by player name
    """
    df = selected_df.copy()
    
    # Sort deterministically
    # Tie breaking: predicted points DESC, role ASC, player name ASC
    sorted_df = df.sort_values(by=['predicted_fantasy_points', 'role', 'player'], ascending=[False, True, True])
    
    if len(sorted_df) < 2:
        raise ValueError("Must have at least 2 players to select C and VC.")
        
    captain = sorted_df.iloc[0]['player']
    vice_captain = sorted_df.iloc[1]['player']
    
    df['is_captain'] = (df['player'] == captain).astype(int)
    df['is_vice_captain'] = (df['player'] == vice_captain).astype(int)
    
    # Recalculate points with multipliers
    # Base is 1.0. C adds 1.0 (total 2.0), VC adds 0.5 (total 1.5)
    df['optimized_points'] = df['predicted_fantasy_points'] * (1.0 + df['is_captain'] * 1.0 + df['is_vice_captain'] * 0.5)
    
    return df
