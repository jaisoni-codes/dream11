import pytest
import pandas as pd
from src.features.fantasy_points import calculate_fantasy_points

def test_fantasy_scoring():
    matches_df = pd.DataFrame([{
        'match_id': 'm1',
        'date': '2024-01-01',
        'format': 'T20'
    }])
    
    # 1. Batter scoring 50 off 30 balls (SR 166.6)
    # runs: 50*1 = 50
    # 30 bonus = 4, 50 bonus = 8 -> 12
    # 5 fours = 5*1 = 5
    # 2 sixes = 2*2 = 4
    # SR bonus = 166.6 -> between 150-170 -> 4
    # Total bat pts = 50 + 12 + 5 + 4 + 4 = 75
    # Played 11 pts = 4
    # Total expected = 79
    
    deliveries = []
    for i in range(30):
        runs = 0
        if i < 5: runs = 4 # 5 fours
        elif i < 7: runs = 6 # 2 sixes
        elif i < 25: runs = 1 # 18 singles
        
        deliveries.append({
            'match_id': 'm1',
            'innings': 1,
            'over': i // 6,
            'ball': (i % 6) + 1,
            'batting_team': 'Team A',
            'batter': 'Player A',
            'bowler': 'Player B',
            'non_striker': 'Player C',
            'batter_runs': runs,
            'extras': 0,
            'total_runs': runs,
            'is_wicket': 0,
            'dismissal_kind': None,
            'player_out': None,
            'fielder': None,
            'is_wide': 0,
            'is_noball': 0,
            'is_bye': 0,
            'is_legbye': 0
        })
        
    deliveries_df = pd.DataFrame(deliveries)
    
    pms = calculate_fantasy_points(deliveries_df, matches_df)
    
    player_a_stats = pms[pms['player'] == 'Player A'].iloc[0]
    
    assert player_a_stats['runs'] == 50
    assert player_a_stats['balls'] == 30
    assert player_a_stats['bat_pts'] == 75
    assert player_a_stats['fantasy_points'] == 79


def test_extras_handling():
    matches_df = pd.DataFrame([{'match_id': 'm2', 'date': '2024-01-01', 'format': 'T20'}])
    
    deliveries = [
        # Normal ball: 1 run
        {'match_id': 'm2', 'innings': 1, 'over': 0, 'ball': 1, 'batting_team': 'Team A', 'batter': 'B1', 'bowler': 'BW1', 'non_striker': 'B2',
         'batter_runs': 1, 'extras': 0, 'total_runs': 1, 'is_wicket': 0, 'dismissal_kind': None, 'player_out': None, 'fielder': None,
         'is_wide': 0, 'is_noball': 0, 'is_bye': 0, 'is_legbye': 0},
        
        # Wide: 1 wide run. Does not count as ball faced, does not count as ball bowled. 
        {'match_id': 'm2', 'innings': 1, 'over': 0, 'ball': 2, 'batting_team': 'Team A', 'batter': 'B1', 'bowler': 'BW1', 'non_striker': 'B2',
         'batter_runs': 0, 'extras': 1, 'total_runs': 1, 'is_wicket': 0, 'dismissal_kind': None, 'player_out': None, 'fielder': None,
         'is_wide': 1, 'is_noball': 0, 'is_bye': 0, 'is_legbye': 0},
         
        # No-ball + 1 batter run: 2 total runs. Counts as ball faced, NOT ball bowled.
        {'match_id': 'm2', 'innings': 1, 'over': 0, 'ball': 2, 'batting_team': 'Team A', 'batter': 'B1', 'bowler': 'BW1', 'non_striker': 'B2',
         'batter_runs': 1, 'extras': 1, 'total_runs': 2, 'is_wicket': 0, 'dismissal_kind': None, 'player_out': None, 'fielder': None,
         'is_wide': 0, 'is_noball': 1, 'is_bye': 0, 'is_legbye': 0},
         
        # Bye: 4 byes. Counts as ball faced, counts as ball bowled. Does NOT count towards batter runs or bowler runs.
        {'match_id': 'm2', 'innings': 1, 'over': 0, 'ball': 3, 'batting_team': 'Team A', 'batter': 'B1', 'bowler': 'BW1', 'non_striker': 'B2',
         'batter_runs': 0, 'extras': 4, 'total_runs': 4, 'is_wicket': 0, 'dismissal_kind': None, 'player_out': None, 'fielder': None,
         'is_wide': 0, 'is_noball': 0, 'is_bye': 1, 'is_legbye': 0},
         
        # Leg-bye: 1 leg-bye. Counts as ball faced, counts as ball bowled. Does NOT count towards batter runs or bowler runs.
        {'match_id': 'm2', 'innings': 1, 'over': 0, 'ball': 4, 'batting_team': 'Team A', 'batter': 'B1', 'bowler': 'BW1', 'non_striker': 'B2',
         'batter_runs': 0, 'extras': 1, 'total_runs': 1, 'is_wicket': 0, 'dismissal_kind': None, 'player_out': None, 'fielder': None,
         'is_wide': 0, 'is_noball': 0, 'is_bye': 0, 'is_legbye': 1},
         
        # Legal ball, Wicket.
        {'match_id': 'm2', 'innings': 1, 'over': 0, 'ball': 5, 'batting_team': 'Team A', 'batter': 'B1', 'bowler': 'BW1', 'non_striker': 'B2',
         'batter_runs': 0, 'extras': 0, 'total_runs': 0, 'is_wicket': 1, 'dismissal_kind': 'bowled', 'player_out': 'B1', 'fielder': None,
         'is_wide': 0, 'is_noball': 0, 'is_bye': 0, 'is_legbye': 0},
    ]
    
    deliveries_df = pd.DataFrame(deliveries)
    pms = calculate_fantasy_points(deliveries_df, matches_df)
    
    b1_stats = pms[pms['player'] == 'B1'].iloc[0]
    
    # Batter assertions
    # Balls faced: Legal (1) + No-ball (1) + Bye (1) + Leg-bye (1) + Wicket (1) = 5
    assert b1_stats['balls'] == 5
    # Batter runs: Legal (1) + No-ball (1) = 2
    assert b1_stats['runs'] == 2
    # SR: 2/5 * 100 = 40.0
    assert b1_stats['sr'] == 40.0
    
    bw1_stats = pms[pms['player'] == 'BW1'].iloc[0]
    
    # Bowler assertions
    # Balls bowled: Legal (1) + Bye (1) + Leg-bye (1) + Wicket (1) = 4
    assert bw1_stats['balls_bowled'] == 4
    # Runs conceded: Legal (1) + Wide (1) + No-ball (2) + Bye (0) + Leg-bye (0) = 4
    assert bw1_stats['runs_conceded'] == 4
    
