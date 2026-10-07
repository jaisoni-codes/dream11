import pandas as pd
import numpy as np

def compute_form_features(rolling_df):
    """
    Computes form, momentum, volatility, and deviation features 
    using the purely historical rolling features.
    """
    df = rolling_df.copy()
    
    # Recent vs Career Delta
    # e.g., how is their last 5 matches compared to their career baseline
    if 'fantasy_points_last_5' in df.columns and 'fantasy_points_career_mean' in df.columns:
        df['fantasy_recent_vs_career_delta'] = df['fantasy_points_last_5'] - df['fantasy_points_career_mean']
        
    # Momentum (Z-style: short EWMA vs long EWMA)
    if 'fantasy_points_ewma_short' in df.columns and 'fantasy_points_ewma_long' in df.columns:
        df['fantasy_points_momentum'] = df['fantasy_points_ewma_short'] - df['fantasy_points_ewma_long']
        
    return df
    # but the prompt requires standard deviation / volatility.
    # We will compute it inside a dedicated function passing the shifted data,
    # but for simplicity, let's assume we can merge it or we compute it if we pass the shifted data.
    # Actually, we should just pass the original PMS here to compute standard deviations properly.
    pass

def compute_volatility(pms_df):
    df = pms_df.sort_values(by=['player', 'date', 'match_id']).copy()
    shifted = df.groupby('player')['fantasy_points'].shift(1)
    
    vol = df[['match_id', 'player', 'date']].copy()
    vol['fantasy_points_std_last_10'] = shifted.groupby(df['player']).rolling(window=10, min_periods=3).std().reset_index(level=0, drop=True)
    vol['fantasy_points_std_career'] = shifted.groupby(df['player']).expanding(min_periods=3).std().reset_index(level=0, drop=True)
    return vol

def build_form_features(pms_df, rolling_df):
    vol_df = compute_volatility(pms_df)
    
    # Merge
    merged = pd.merge(rolling_df, vol_df, on=['match_id', 'player', 'date'], how='left')
    merged = compute_form_features(merged)
    
    return merged
