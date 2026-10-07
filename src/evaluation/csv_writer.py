import os
import pandas as pd

def write_prediction_csv(dream_team_result, match_date, team_A, team_B, output_path="reports/dream_team_prediction.csv"):
    """
    Writes the predicted dream team to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = dream_team_result['team_df']
    
    out_df = pd.DataFrame({
        'Match Date': match_date,
        'Team 1': team_A,
        'Team 2': team_B,
        'Player': df['player'],
        'Role': df['role'],
        'Predicted Fantasy Points': df['predicted_fantasy_points'].round(2),
        'Optimized Points': df['optimized_points'].round(2),
        'Captain': df['is_captain'].map({1: 'Yes', 0: 'No'}),
        'Vice Captain': df['is_vice_captain'].map({1: 'Yes', 0: 'No'})
    })
    
    out_df.to_csv(output_path, index=False)
    return output_path
