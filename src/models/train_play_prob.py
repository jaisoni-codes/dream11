import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_calibration(y_true, y_prob, n_bins=5):
    bins = np.linspace(0, 1, n_bins + 1)
    bucket_indices = np.digitize(y_prob, bins) - 1
    
    metrics = []
    for i in range(n_bins):
        mask = bucket_indices == i
        if not np.any(mask):
            continue
        
        bucket_true = y_true[mask]
        bucket_prob = y_prob[mask]
        
        metrics.append({
            'bucket': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            'count': len(bucket_true),
            'mean_pred_prob': np.mean(bucket_prob),
            'actual_play_rate': np.mean(bucket_true)
        })
        
    return pd.DataFrame(metrics)

def train_model():
    logger.info("Loading play probability dataset...")
    df = pd.read_parquet('data/processed/play_prob_dataset.parquet')
    
    df['target_match_date'] = pd.to_datetime(df['target_match_date'])
    df = df.sort_values('target_match_date').reset_index(drop=True)
    
    features = [
        'days_since_last_match',
        'matches_played_before',
        'fantasy_points_last_1', 'fantasy_points_last_5', 'fantasy_points_last_10',
        'fantasy_points_ewma_short', 'fantasy_points_ewma_long',
        'fantasy_points_career_mean', 'fantasy_points_career_sum',
        'fantasy_recent_vs_career_delta', 'fantasy_points_momentum',
        'runs_career_mean', 'wickets_career_mean',
        'batting_opportunity_rate', 'bowling_opportunity_rate',
        'role_confidence'
    ]
    
    # role is categorical string, let's encode it quickly
    df['role_cat'] = df['role'].astype('category').cat.codes
    features.append('role_cat')
    
    # role_confidence is string
    if 'role_confidence' in df.columns:
        df['role_confidence'] = df['role_confidence'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1)
    
    # Ensure all features exist
    available_features = [f for f in features if f in df.columns]
    logger.info(f"Using {len(available_features)} features.")
    
    target = 'y_play'
    
    folds = [
        ('Fold 1', '2021-12-31', '2022-01-01', '2022-12-31'),
        ('Fold 2', '2022-12-31', '2023-01-01', '2023-12-31'),
        ('Fold 3', '2023-12-31', '2024-01-01', '2024-06-30'),
    ]
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'random_state': 42,
        'is_unbalance': True,  # automatically handle class imbalance
        'verbose': -1
    }
    
    all_metrics = []
    
    for fold_name, train_end, val_start, val_end in folds:
        logger.info(f"\n--- {fold_name} ---")
        train_df = df[df['target_match_date'] <= train_end]
        val_df = df[(df['target_match_date'] >= val_start) & (df['target_match_date'] <= val_end)]
        
        if val_df.empty:
            continue
            
        X_train, y_train = train_df[available_features], train_df[target]
        X_val, y_val = val_df[available_features], val_df[target]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        y_prob = model.predict(X_val)
        y_pred = (y_prob >= 0.5).astype(int)
        
        roc_auc = roc_auc_score(y_val, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        brier = brier_score_loss(y_val, y_prob)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        logger.info(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
        logger.info(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        calib_df = evaluate_calibration(y_val.values, y_prob)
        logger.info(f"Calibration Buckets:\n{calib_df.to_string(index=False)}")
        
    # Final Frozen Model
    logger.info("\n--- Training Final Frozen Model (Cutoff 2024-06-30) ---")
    final_df = df[df['target_match_date'] <= '2024-06-30']
    X_final, y_final = final_df[available_features], final_df[target]
    final_data = lgb.Dataset(X_final, label=y_final)
    
    final_model = lgb.train(
        lgb_params,
        final_data,
        num_boost_round=150,  # approximate from CV
    )
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': final_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Top 5 Features:\n{importance.head(5).to_string(index=False)}")
    
    bundle = {
        'model': final_model,
        'features': available_features,
        'cutoff': '2024-06-30',
        'version': '1.0',
        'random_seed': 42,
        'config': lgb_params
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(bundle, 'models/play_probability_bundle.pkl')
    logger.info("Saved final model to models/play_probability_bundle.pkl")

if __name__ == '__main__':
    train_model()
