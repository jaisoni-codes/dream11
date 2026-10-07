import pandas as pd
import numpy as np
import yaml
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.lgbm_fp import LGBMFantasyPointsModel
from src.models.bundle import save_model_bundle
import lightgbm as lgb
import joblib

from src.inference.pipeline import PreTossInferencePipeline

def train_ablation_fp(start_date, end_date='2024-06-30'):
    X_df = pd.read_parquet('data/processed/player_features.parquet')
    y_df = pd.read_parquet('data/processed/player_match_stats.parquet')[['match_id', 'player', 'fantasy_points']]
    df = X_df.merge(y_df, on=['match_id', 'player'], how='inner')
    
    df['date_obj'] = pd.to_datetime(df['date']).dt.date
    mask = (df['date_obj'] >= pd.to_datetime(start_date).date()) & (df['date_obj'] <= pd.to_datetime(end_date).date())
    train_df = df[mask].copy()
    
    # We will use the last 20% of the training window as validation for early stopping
    train_df = train_df.sort_values('date_obj')
    val_idx = int(len(train_df) * 0.8)
    
    t_df = train_df.iloc[:val_idx]
    v_df = train_df.iloc[val_idx:]
    
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model = LGBMFantasyPointsModel(config['lgbm_params'])
    # Drop non-feature columns that would confuse LightGBM
    for col in ['date_obj', 'date']:
        if col in t_df.columns: t_df = t_df.drop(columns=[col])
        if col in v_df.columns: v_df = v_df.drop(columns=[col])
    model.fit(t_df, t_df['fantasy_points'], v_df, v_df['fantasy_points'])
    
    path = f'models/ablation_fp_{start_date[:4]}.pkl'
    save_model_bundle(
        model=model.model,
        features=model.features,
        config=config,
        cutoff=end_date,
        save_path=path
    )
    return path, len(train_df)

def train_ablation_play(start_date, end_date='2024-06-30'):
    df = pd.read_parquet('data/processed/play_prob_dataset_full.parquet')
    df['date_obj'] = pd.to_datetime(df['target_match_date']).dt.date
    
    mask = (df['date_obj'] >= pd.to_datetime(start_date).date()) & (df['date_obj'] <= pd.to_datetime(end_date).date())
    train_df = df[mask].copy()
    
    if 'role_cat' not in train_df.columns:
        train_df['role_cat'] = train_df['role'].astype('category').cat.codes
    if 'role_confidence' in train_df.columns and train_df['role_confidence'].dtype == 'O':
        train_df['role_confidence'] = train_df['role_confidence'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1)
        
    features = [
        'days_since_last_match',
        'fantasy_points_ewma_short',
        'fantasy_points_career_mean',
        'matches_played_before',
        'batting_opportunity_rate',
        'bowling_opportunity_rate',
        'fantasy_points_last_1',
        'fantasy_points_ewma_long',
        'role_cat',
        'role_confidence',
        'batting_strike_rate_career',
        'bowling_strike_rate_career',
        'economy_rate_career',
        'runs_career_mean',
        'wickets_career_mean',
        'recent_fantasy_points_var',
        'fantasy_points_career_sum'
    ]
    
    for f in features:
        if f not in train_df.columns:
            train_df[f] = 0
            
    train_df = train_df.sort_values('date_obj')
    val_idx = int(len(train_df) * 0.8)
    
    t_df = train_df.iloc[:val_idx]
    v_df = train_df.iloc[val_idx:]
    
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 300,
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**lgbm_params)
    
    # Corrected early stopping usage
    model.fit(
        t_df[features], t_df['y_play'],
        eval_set=[(v_df[features], v_df['y_play'])],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    path = f'models/ablation_play_{start_date[:4]}.pkl'
    bundle = {
        'model': model,
        'features': features,
        'cutoff': end_date
    }
    joblib.dump(bundle, path)
    return path, len(train_df)

def run_ablation():
    windows = ['2005-01-01', '2015-01-01', '2018-01-01']
    
    pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
    feats_df = pd.read_parquet('data/processed/player_features.parquet')
    dels_df = pd.read_parquet('data/processed/deliveries.parquet')
    
    pms_df['date'] = pd.to_datetime(pms_df['date'])
    eval_matches = pms_df[pms_df['date'] >= '2024-07-01'][['match_id', 'date']].drop_duplicates().sort_values('date').head(100)
    
    results = []
    
    for w in windows:
        print(f"--- Ablation Window: {w} ---")
        fp_path, fp_rows = train_ablation_fp(w)
        play_path, play_rows = train_ablation_play(w)
        
        pipeline = PreTossInferencePipeline(fp_model_path=fp_path, play_model_path=play_path)
        
        # Run Evaluation
        lat_list = []
        overlaps = []
        recalls = []
        jaccards = []
        regrets = []
        c_accs = []
        vc_accs = []
        
        for idx, row in eval_matches.iterrows():
            match_id = row['match_id']
            match_date = row['date']
            
            teams = dels_df[dels_df['match_id'] == match_id]['batting_team'].unique()
            if len(teams) != 2: continue
            
            try:
                pool, dt, lats = pipeline.predict(match_date, teams[0], teams[1], pms_df, feats_df)
            except ValueError:
                continue
                
            pred_xi = set(dt['team_df']['player'])
            
            actual_stats = pms_df[pms_df['match_id'] == match_id].copy()
            actual_players = set(actual_stats['player'])
            
            pred_actual_pts = sum(
                actual_stats[actual_stats['player'] == p]['fantasy_points'].iloc[0] 
                for p in pred_xi if p in actual_players
            )
            
            # Simple Regret
            actual_best = actual_stats.sort_values('fantasy_points', ascending=False).head(11)['fantasy_points'].sum()
            regret = actual_best - pred_actual_pts
            
            overlap = len(pred_xi.intersection(actual_players))
            recall = overlap / 11.0
            
            overlaps.append(overlap)
            recalls.append(recall)
            regrets.append(regret)
            
            # We skip jaccard/C/VC strict calc for brief ablation since it's an estimate
            c_accs.append(1 if actual_stats[actual_stats['player'] == dt['captain']]['fantasy_points'].sum() > 0 else 0)
            lat_list.append(lats['total_inference'])
            
        results.append({
            'Training Window': f"{w[:4]}-2024",
            'Training Rows (FP)': fp_rows,
            'Training Rows (Play)': play_rows,
            'Recall@11': np.mean(recalls),
            'Mean XI Overlap': np.mean(overlaps),
            'Team Regret': np.mean(regrets),
            'Captain Acc': np.mean(c_accs), # Roughly checking if they played
            'Mean Inf Latency (s)': np.mean(lat_list)
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv('reports/ablation_results.csv', index=False)
    print(df_res.to_markdown())

if __name__ == '__main__':
    run_ablation()
