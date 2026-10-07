import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss
from src.models.lgbm_fp import LGBMFantasyPointsModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"

def run_m10_ablation():
    logger.info("Loading M10 Features (with Context)...")
    feats_df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'player_features_m10.parquet'))
    pms_df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'player_match_stats.parquet'))
    feats_df = feats_df.merge(pms_df[['match_id', 'player', 'fantasy_points']], on=['match_id', 'player'], how='inner')
    
    play_df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'play_prob_dataset.parquet'))
    
    # Context features to ablate
    context_features = ["opp_avg_pts", "venue_avg_pts"]
    
    experiments = {
        "Baseline": [],
        "Opponent Only": ["opp_avg_pts"],
        "Venue Only": ["venue_avg_pts"],
        "Opponent + Venue": ["opp_avg_pts", "venue_avg_pts"]
    }
    
    feats_df['date_obj'] = pd.to_datetime(feats_df['date']).dt.date
    train_end = pd.to_datetime('2023-06-30').date()
    val_end = pd.to_datetime('2024-06-30').date()
    
    train_df = feats_df[(feats_df['date_obj'] >= pd.to_datetime('2015-01-01').date()) & (feats_df['date_obj'] <= train_end)].copy()
    val_df = feats_df[(feats_df['date_obj'] > train_end) & (feats_df['date_obj'] <= val_end)].copy()
    
    play_train = play_df[(pd.to_datetime(play_df['date']).dt.date >= pd.to_datetime('2015-01-01').date()) & (pd.to_datetime(play_df['date']).dt.date <= train_end)].copy()
    play_val = play_df[(pd.to_datetime(play_df['date']).dt.date > train_end) & (pd.to_datetime(play_df['date']).dt.date <= val_end)].copy()
    
    # Merge contextual features into play dataset
    play_train = play_train.merge(feats_df[['match_id', 'player'] + context_features], on=['match_id', 'player'], how='left').fillna(0)
    play_val = play_val.merge(feats_df[['match_id', 'player'] + context_features], on=['match_id', 'player'], how='left').fillna(0)
    
    results = []
    
    for name, keep_features in experiments.items():
        logger.info(f"=== Running Experiment: {name} ===")
        drop_features = [f for f in context_features if f not in keep_features]
        
        # Train FP Model
        fp_train_sub = train_df.drop(columns=drop_features + ['date_obj'])
        fp_val_sub = val_df.drop(columns=drop_features + ['date_obj'])
        
        fp_params = {
            'objective': 'regression', 'metric': 'mae',
            'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31,
            'verbose': -1, 'seed': 42
        }
        fp_model = LGBMFantasyPointsModel(params=fp_params)
        fp_model.fit(fp_train_sub, fp_train_sub['fantasy_points'], fp_val_sub, fp_val_sub['fantasy_points'])
        
        preds = fp_model.predict(fp_val_sub)
        fp_mae = np.mean(np.abs(preds - fp_val_sub['fantasy_points']))
        
        # Train Play Model
        play_features = [c for c in play_train.columns if c not in ['match_id', 'player', 'date', 'y_play', 'target_match_id', 'target_match_date', 'team', 'date_obj', 'eligibility_source'] + drop_features]
        
        # Fix categorical types
        cat_cols = ['role', 'role_confidence', 'role_source', 'format']
        for col in cat_cols:
            if col in play_train.columns:
                play_train[col] = play_train[col].astype('category')
                play_val[col] = play_val[col].astype('category')
                
        dtrain = lgb.Dataset(play_train[play_features], label=play_train['y_play'])
        dval = lgb.Dataset(play_val[play_features], label=play_val['y_play'], reference=dtrain)
        
        play_params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31,
            'verbose': -1, 'seed': 42
        }
        
        play_model_lgb = lgb.train(play_params, dtrain, valid_sets=[dval], num_boost_round=200, callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
        play_preds = play_model_lgb.predict(play_val[play_features])
        play_ll = log_loss(play_val['y_play'], play_preds)
        
        logger.info(f"[{name}] FP MAE: {fp_mae:.4f} | P(play) LogLoss: {play_ll:.4f}")
        
        results.append({
            'Feature Set': name,
            'FP MAE': fp_mae,
            'P(play) LogLoss': play_ll
        })
        
    res_df = pd.DataFrame(results)
    print("\n=== M10 Validation Ablation Results ===")
    print(res_df.to_markdown(index=False))
    res_df.to_csv('reports/m10_validation_ablation.csv', index=False)

if __name__ == '__main__':
    run_m10_ablation()
