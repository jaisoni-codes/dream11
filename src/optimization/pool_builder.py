import pandas as pd
import logging
import functools

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=1)
def _load_dels_cache():
    """Load and cache the deliveries lookup table once per process."""
    df = pd.read_parquet("data/processed/deliveries.parquet")[
        ['match_id', 'batting_team', 'batter']
    ].drop_duplicates()
    return df.rename(columns={'batter': 'player', 'batting_team': 'team'})

def build_reconstructed_pool(target_date, team_A, team_B, pms_df, features_df, days_lookback=365):
    """
    Constructs a pre-toss candidate pool purely from historical data.
    Rule: Any player whose most recent team assignment in the last `days_lookback` 
    matches either team_A or team_B.
    
    This ensures no target-match (playing XI) leakage occurs.
    """
    target_date = pd.to_datetime(target_date).date()
    
    # 1. Filter history strictly before target date
    pms_df['date_obj'] = pd.to_datetime(pms_df['date']).dt.date
    history = pms_df[pms_df['date_obj'] < target_date].copy()
    
    cutoff_date = target_date - pd.Timedelta(days=days_lookback)
    recent_history = history[history['date_obj'] >= cutoff_date].copy()
    
    if recent_history.empty:
        logger.warning(f"No history found before {target_date} within {days_lookback} days.")
        return pd.DataFrame()
        
    # Load deliveries via cached loader (reads from disk only once per process)
    dels_dates = _load_dels_cache()
    
    # Get match dates from history
    match_dates = history[['match_id', 'date_obj']].drop_duplicates()
    
    merged = pd.merge(dels_dates, match_dates, on='match_id', how='inner')
    
    if merged.empty:
        logger.warning("Could not map teams from deliveries.")
        return pd.DataFrame()
        
    last_teams = merged.sort_values('date_obj').groupby('player')['team'].last().reset_index()
        
    # Filter candidates belonging to Team A or Team B
    candidates = last_teams[last_teams['team'].isin([team_A, team_B])].copy()
    
    if candidates.empty:
        return pd.DataFrame()
        
    # 2. Get latest features strictly before target date
    features_df['date_obj'] = pd.to_datetime(features_df['date']).dt.date
    hist_feats = features_df[features_df['date_obj'] < target_date].copy()
    
    # Sort and get the last feature row for each candidate
    latest_feats = hist_feats.sort_values('date_obj').groupby('player').last().reset_index()
    
    # 3. Merge candidates with their latest features
    pool = pd.merge(candidates, latest_feats, on='player', how='inner')
    
    # Note: 'role' is present in features_df
    # Add source tracking
    pool['eligibility_source'] = 'reconstructed_historical_pool'
    
    return pool
