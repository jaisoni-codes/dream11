import pandas as pd
import numpy as np

def build_contextual_features(pms_df, matches_df, deliveries_df):
    """
    Computes expanding opponent and venue contextual features purely pre-toss.
    Returns a DataFrame with ['match_id', 'player', 'opp_avg_pts', 'venue_avg_pts']
    """
    # 1. Figure out Player's Team and Opponent for each match
    del_match = deliveries_df.merge(matches_df[['match_id', 'team1', 'team2']], on='match_id', how='left')
    del_match['bowling_team'] = np.where(del_match['batting_team'] == del_match['team1'], del_match['team2'], del_match['team1'])
    
    team_mapping = del_match[['match_id', 'batting_team', 'batter']].drop_duplicates()
    team_mapping.columns = ['match_id', 'team', 'player']
    
    bowling_team_mapping = del_match[['match_id', 'bowling_team', 'bowler']].drop_duplicates()
    bowling_team_mapping.columns = ['match_id', 'team', 'player']
    
    player_teams = pd.concat([team_mapping, bowling_team_mapping]).drop_duplicates(subset=['match_id', 'player'])
    
    # 2. Join matches info
    df = pms_df[['match_id', 'player', 'date', 'fantasy_points']].copy()
    df = df.merge(player_teams, on=['match_id', 'player'], how='left')
    df = df.merge(matches_df[['match_id', 'team1', 'team2', 'venue']], on='match_id', how='left')
    
    # Determine opponent
    df['opponent'] = np.where(df['team'] == df['team1'], df['team2'], df['team1'])
    # If team is unknown, opponent is unknown
    df.loc[df['team'].isna(), 'opponent'] = 'Unknown'
    
    # Sort strictly chronologically
    df = df.sort_values(['player', 'date', 'match_id']).reset_index(drop=True)
    
    # 3. Expanding Average by (Player, Opponent) - STRICTLY SHIFTED
    # We group by player and opponent, and take the expanding mean, then shift(1)
    df['opp_expanding'] = df.groupby(['player', 'opponent'])['fantasy_points'].transform(lambda x: x.expanding().mean().shift(1))
    
    # 4. Expanding Average by (Player, Venue) - STRICTLY SHIFTED
    df['venue_expanding'] = df.groupby(['player', 'venue'])['fantasy_points'].transform(lambda x: x.expanding().mean().shift(1))
    
    # Fill NAs for players with no history against this opponent or at this venue
    # with their global expanding mean
    df['global_expanding'] = df.groupby('player')['fantasy_points'].transform(lambda x: x.expanding().mean().shift(1))
    
    df['opp_avg_pts'] = df['opp_expanding'].fillna(df['global_expanding']).fillna(0)
    df['venue_avg_pts'] = df['venue_expanding'].fillna(df['global_expanding']).fillna(0)
    
    return df[['match_id', 'player', 'opp_avg_pts', 'venue_avg_pts']]
