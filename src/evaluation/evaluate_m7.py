import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.inference.pipeline import PreTossInferencePipeline
from src.optimization.dream_team import generate_dream_team

def evaluate_m7():
    print("Loading data and pipeline...")
    pipeline = PreTossInferencePipeline()
    
    pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
    feats_df = pd.read_parquet('data/processed/player_features.parquet')
    dels_df = pd.read_parquet('data/processed/deliveries.parquet')
    
    pms_df['date'] = pd.to_datetime(pms_df['date'])
    feats_df['date'] = pd.to_datetime(feats_df['date'])
    
    eval_matches = pms_df[pms_df['date'] >= '2024-07-01'][['match_id', 'date']].drop_duplicates().sort_values('date')
    
    print(f"Evaluating {len(eval_matches)} matches...", flush=True)
    results = []
    
    latencies = {
        'pool': [],
        'fp_inf': [],
        'play_inf': [],
        'ilp': [],
        'total': []
    }
    
    count = 0
    # Evaluate 250 matches to stay consistent and fast
    for idx, row in eval_matches.iterrows():
        count += 1
        if count > 250:
            break
            
        match_id = row['match_id']
        match_date = row['date']
        
        match_dels = dels_df[dels_df['match_id'] == match_id]
        teams = match_dels['batting_team'].unique()
        if len(teams) != 2:
            continue
            
        team_a, team_b = teams[0], teams[1]
        
        # INFERENCE
        try:
            pool, dt, lats = pipeline.predict(match_date, team_a, team_b, pms_df, feats_df)
        except ValueError:
            continue
            
        print(".", end="", flush=True)
        pred_xi = set(dt['team_df']['player'])
        pred_c = dt['captain']
        pred_vc = dt['vice_captain']
        
        # ACTUAL GROUND TRUTH
        actual_stats = pms_df[pms_df['match_id'] == match_id].copy()
        actual_players = set(actual_stats['player'])
        
        pred_actual_pts = sum(
            actual_stats[actual_stats['player'] == p]['fantasy_points'].iloc[0] 
            for p in pred_xi if p in actual_players
        )
        
        # Calculate Dream XI
        actual_feats = feats_df[feats_df['match_id'] == match_id][['player', 'role']]
        actual_df = pd.merge(actual_stats, actual_feats, on='player', how='inner')
        batter_teams = match_dels[['batter', 'batting_team']].drop_duplicates().rename(columns={'batter': 'player', 'batting_team': 'team'})
        actual_df = pd.merge(actual_df, batter_teams, on='player', how='left')
        actual_df['team'] = actual_df['team'].fillna(team_a)
        actual_df['predicted_fantasy_points'] = actual_df['fantasy_points']
        
        try:
            actual_dt = generate_dream_team(actual_df)
            actual_xi = set(actual_dt['team_df']['player'])
            actual_c = actual_dt['captain']
            actual_vc = actual_dt['vice_captain']
            actual_base_points = actual_dt['team_df']['predicted_fantasy_points'].sum()
        except ValueError:
            continue
            
        # METRICS
        overlap = len(pred_xi.intersection(actual_xi))
        recall = overlap / 11.0
        precision = overlap / len(pred_xi) if len(pred_xi) > 0 else 0
        jaccard = overlap / (len(pred_xi) + 11 - overlap)
        regret = actual_base_points - pred_actual_pts
        
        c_acc = 1 if pred_c == actual_c else 0
        vc_acc = 1 if pred_vc == actual_vc else 0
        
        latencies['pool'].append(lats['pool_construction'])
        latencies['fp_inf'].append(lats['m_fp_inference'])
        latencies['play_inf'].append(lats['m_play_inference'])
        latencies['ilp'].append(lats['ilp_optimization'])
        latencies['total'].append(lats['total_inference'])
        
        results.append({
            'Match Date': match_date,
            'Match ID': match_id,
            'Team 1': team_a,
            'Team 2': team_b,
            'Overlap': overlap,
            'Recall@11': recall,
            'Precision@11': precision,
            'Jaccard': jaccard,
            'Predicted XI Actual Pts': pred_actual_pts,
            'Dream XI Pts': actual_base_points,
            'Team Regret': regret,
            'Captain Correct': c_acc,
            'VC Correct': vc_acc
        })
        
    res_df = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    res_df.to_csv('reports/milestone7_evaluation.csv', index=False)
    
    # Summarize
    def p50(x): return np.percentile(x, 50)
    def p90(x): return np.percentile(x, 90)
    
    summary = {
        'Matches Evaluated': len(res_df),
        'Mean Overlap': res_df['Overlap'].mean(),
        'Mean Recall@11': res_df['Recall@11'].mean(),
        'Mean Precision@11': res_df['Precision@11'].mean(),
        'Mean Jaccard': res_df['Jaccard'].mean(),
        'Mean Team Regret': res_df['Team Regret'].mean(),
        'Mean Captain Acc': res_df['Captain Correct'].mean(),
        'Mean VC Acc': res_df['VC Correct'].mean(),
        'Latency Pool (mean)': np.mean(latencies['pool']),
        'Latency Pool (p50)': p50(latencies['pool']),
        'Latency Pool (p90)': p90(latencies['pool']),
        'Latency FP Inf (mean)': np.mean(latencies['fp_inf']),
        'Latency Play Inf (mean)': np.mean(latencies['play_inf']),
        'Latency ILP (mean)': np.mean(latencies['ilp']),
        'Latency Total (mean)': np.mean(latencies['total']),
        'Latency Total (p90)': p90(latencies['total']),
    }
    
    pd.DataFrame([summary]).to_csv('reports/milestone7_summary.csv', index=False)
    
    with open('reports/milestone7_report.md', 'w') as f:
        f.write("# Milestone 7: Final Pre-Toss Inference Hardening\n\n")
        f.write("## End-to-End Metrics\n")
        for k, v in summary.items():
            if isinstance(v, float):
                f.write(f"- **{k}**: {v:.4f}\n")
            else:
                f.write(f"- **{k}**: {v}\n")
                
    print("Milestone 7 complete!")

if __name__ == '__main__':
    evaluate_m7()
