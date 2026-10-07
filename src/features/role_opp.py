import pandas as pd
import os

def load_overrides():
    if os.path.exists('configs/role_overrides.csv'):
        return pd.read_csv('configs/role_overrides.csv')
    return pd.DataFrame(columns=['player', 'override_role'])

def infer_roles_and_opportunity(pms_df, features_df):
    """
    Infers role (BAT, BOWL, AR, WK) from strictly historical data (features_df).
    Also generates opportunity proxies (e.g. historical batting involvement).
    """
    df = features_df.copy()
    
    # Defaults
    df['role'] = 'BAT' # default fallback
    df['role_confidence'] = 'low'
    df['role_source'] = 'inferred'
    
    # Opportunity features
    if 'balls_career_mean' in df.columns and 'balls_bowled_career_mean' in df.columns:
        df['batting_opportunity_rate'] = df['balls_career_mean'] # proxy for balls faced per match
        df['bowling_opportunity_rate'] = df['balls_bowled_career_mean'] / 6.0 # overs per match
        
        # Inference Logic
        # Meaningful batting: faces > 10 balls per match on average
        # Meaningful bowling: bowls > 1.5 overs per match on average
        
        is_regular_bowler = df['bowling_opportunity_rate'] >= 1.5
        is_regular_batter = df['batting_opportunity_rate'] >= 10.0
        
        # AR: both
        mask_ar = is_regular_bowler & is_regular_batter
        # BOWL: bowls but doesn't bat much
        mask_bowl = is_regular_bowler & ~is_regular_batter
        # BAT: everything else (defaults to BAT)
        
        df.loc[mask_bowl, 'role'] = 'BOWL'
        df.loc[mask_ar, 'role'] = 'AR'
        
        # Confidence
        # If they have > 5 matches, confidence is medium/high
        if 'matches_played_before' in df.columns:
            df.loc[df['matches_played_before'] > 5, 'role_confidence'] = 'medium'
            df.loc[df['matches_played_before'] > 20, 'role_confidence'] = 'high'
            
    # WK Inference
    # We can infer WK if their historical stumping rate is high.
    if 'stumpings_career_sum' in df.columns:
        # A keeper usually has >0 stumpings over a career. Let's say > 2 stumpings = WK
        # Alternatively, if they have many catches but 0 bowling, could be WK, but hard to tell from pure outfielders.
        # We'll use stumpings > 0 as a strong indicator of WK (or part-time WK).
        mask_wk = df['stumpings_career_sum'] >= 1
        df.loc[mask_wk, 'role'] = 'WK'
        
    # Apply Overrides (post-inference)
    overrides = load_overrides()
    if not overrides.empty:
        # merge
        df = df.merge(overrides, on='player', how='left')
        mask_override = df['override_role'].notnull()
        df.loc[mask_override, 'role'] = df.loc[mask_override, 'override_role']
        df.loc[mask_override, 'role_source'] = 'override'
        df.loc[mask_override, 'role_confidence'] = 'high'
        df.drop(columns=['override_role'], inplace=True)
        
    return df
