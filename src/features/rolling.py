import pandas as pd
import yaml

def load_feature_config():
    with open('configs/features.yaml', 'r') as f:
        return yaml.safe_load(f)

def compute_rolling_features(pms_df):
    """
    Computes rolling and EWMA features.
    CRITICAL LEAKAGE RULE: Must only use matches BEFORE the current match date.
    Achieved by sorting by player, date, and using shift(1).
    """
    config = load_feature_config()
    windows = config.get('rolling_windows', [1, 3, 5, 10, 20])
    ewma_spans = config.get('ewma_spans', {'short': 5, 'long': 20})
    cols = config.get('columns_to_roll', ['fantasy_points'])
    
    # Sort strictly by date
    df = pms_df.sort_values(by=['player', 'date', 'match_id']).copy()
    
    # Filter cols to those that actually exist in the dataframe
    cols = [c for c in cols if c in df.columns]
    
    # 1. Create a shifted dataframe to prevent same-match leakage
    # For every row, the 'shifted' values represent the state of the player strictly BEFORE this match.
    shifted = df.groupby('player')[cols].shift(1)
    
    result = df[['match_id', 'player', 'date', 'format']].copy()
    
    # Create an indicator for whether the player has ANY history
    # If shifted is null for fantasy points, they have no history prior to this match.
    result['has_history'] = shifted['fantasy_points'].notnull().astype(int)
    result['matches_played_before'] = df.groupby('player').cumcount()
    
    for w in windows:
        # We roll over the shifted data
        rolled = shifted.groupby(df['player']).rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        for col in cols:
            result[f'{col}_last_{w}'] = rolled[col]
            
        # For sums (like runs, wickets, matches)
        rolled_sum = shifted.groupby(df['player']).rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        result[f'matches_played_last_{w}'] = shifted['fantasy_points'].notnull().groupby(df['player']).rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)

    for span_name, span_val in ewma_spans.items():
        ewma = shifted.groupby(df['player']).ewm(span=span_val, min_periods=1).mean().reset_index(level=0, drop=True)
        for col in cols:
            result[f'{col}_ewma_{span_name}'] = ewma[col]
            
    # Also add career means (expanding window on shifted)
    expanding = shifted.groupby(df['player']).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    for col in cols:
        result[f'{col}_career_mean'] = expanding[col]
        
    expanding_sum = shifted.groupby(df['player']).expanding(min_periods=1).sum().reset_index(level=0, drop=True)
    for col in cols:
        result[f'{col}_career_sum'] = expanding_sum[col]

    return result
