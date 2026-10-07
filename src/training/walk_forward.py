import os
import yaml
import logging
import pandas as pd
import numpy as np

from src.models.baselines import BaselineB1, BaselineB2, BaselineB3, BaselineB4, BaselineB5
from src.models.lgbm_fp import LGBMFantasyPointsModel
from src.models.bundle import save_model_bundle
from src.evaluation.metrics import calculate_metrics, print_metrics_table
from src.utils.config import HARD_CUTOFF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    features_path = "data/processed/player_features.parquet"
    stats_path = "data/processed/player_match_stats.parquet"
    
    if not os.path.exists(features_path) or not os.path.exists(stats_path):
        raise FileNotFoundError("Run milestone 1 and 2 pipelines first.")
        
    X_df = pd.read_parquet(features_path)
    y_df = pd.read_parquet(stats_path)[['match_id', 'player', 'fantasy_points']]
    
    # Merge target
    # The target fantasy_points comes ONLY from the current match result.
    # The features in X_df are strictly computed before the current match.
    df = X_df.merge(y_df, on=['match_id', 'player'], how='inner')
    
    # Strictly enforce HARD_CUTOFF
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[df['date'] <= HARD_CUTOFF].copy()
    
    # Leakage check: target 'fantasy_points' should only be the last column
    # Ensure no other target proxies leaked
    leakage_cols = ['runs', 'wickets', 'catches']
    for c in leakage_cols:
        assert c not in df.columns, f"Leakage detected: {c} found in features!"
        
    return df

def run_walk_forward():
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    df = load_data()
    
    folds = config['folds']
    lgbm_params = config['lgbm_params']
    
    results = []
    
    # For final model
    best_lgbm = None
    
    for i, fold in enumerate(folds):
        train_end = pd.to_datetime(fold['train_end']).date()
        val_start = pd.to_datetime(fold['val_start']).date()
        val_end = pd.to_datetime(fold['val_end']).date()
        
        train_mask = df['date'] <= train_end
        val_mask = (df['date'] >= val_start) & (df['date'] <= val_end)
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        
        if len(train_df) == 0 or len(val_df) == 0:
            logger.warning(f"Fold {i+1} has no data. Skipping.")
            continue
            
        y_train = train_df['fantasy_points']
        y_val = val_df['fantasy_points']
        
        logger.info(f"Fold {i+1}: Train rows: {len(train_df)}, Val rows: {len(val_df)}")
        
        # Evaluate Baselines
        baselines = {
            'B1': BaselineB1(),
            'B2': BaselineB2(),
            'B3': BaselineB3(),
            'B4': BaselineB4(),
            'B5': BaselineB5()
        }
        
        for b_name, b_model in baselines.items():
            b_model.fit(train_df, y_train)
            preds = b_model.predict(val_df)
            metrics = calculate_metrics(y_val, preds)
            results.append({
                'Model': b_name,
                'Fold': i + 1,
                **metrics
            })
            
        # Evaluate LightGBM
        lgbm = LGBMFantasyPointsModel(lgbm_params)
        lgbm.fit(train_df, y_train, val_df, y_val)
        lgbm_preds = lgbm.predict(val_df)
        lgbm_metrics = calculate_metrics(y_val, lgbm_preds)
        results.append({
            'Model': 'LightGBM',
            'Fold': i + 1,
            **lgbm_metrics
        })
        
        # Save feature importance of the last fold
        if i == len(folds) - 1:
            best_lgbm = lgbm
            fi = lgbm.get_feature_importance()
            os.makedirs("reports", exist_ok=True)
            fi.to_csv("reports/feature_importance.csv", index=False)
            logger.info("Saved feature importance to reports/feature_importance.csv")
            
    # Print comparison
    print_metrics_table(results)
    
    # Save the final bundle
    if best_lgbm is not None:
        save_model_bundle(
            model=best_lgbm.model,
            features=best_lgbm.features,
            config=config,
            cutoff=HARD_CUTOFF
        )

if __name__ == '__main__':
    run_walk_forward()
