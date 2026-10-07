import os
import yaml
import numpy as np
import pandas as pd
from src.models.baselines import BaselineB1, BaselineB2, BaselineB3, BaselineB4, BaselineB5
from src.models.bundle import load_model_bundle
from src.evaluation.metrics import calculate_metrics
from src.training.walk_forward import load_data

def run_diagnostics():
    # 6. Model Artifact Check
    bundle_path = "models/model_bundle.pkl"
    assert os.path.exists(bundle_path), "Bundle not found!"
    bundle = load_model_bundle(bundle_path)
    
    model = bundle['model']
    features = bundle['features']
    cutoff = bundle['metadata']['training_cutoff']
    
    print("\n--- MODEL ARTIFACT CHECK ---")
    print("Model loads successfully:", model is not None)
    print("Feature list loads successfully:", features is not None)
    print("Feature count:", len(features))
    print("Cutoff stored:", cutoff)
    
    # Load Data
    df = load_data()
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    folds = config['folds']
    
    all_metrics = []
    
    # For prediction sanity check and error analysis (pooled val)
    lgbm_all_preds = []
    y_all_actuals = []
    roles_all = []
    history_all = []
    
    for i, fold in enumerate(folds):
        train_end = pd.to_datetime(fold['train_end']).date()
        val_start = pd.to_datetime(fold['val_start']).date()
        val_end = pd.to_datetime(fold['val_end']).date()
        
        train_mask = df['date'] <= train_end
        val_mask = (df['date'] >= val_start) & (df['date'] <= val_end)
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        
        y_train = train_df['fantasy_points']
        y_val = val_df['fantasy_points']
        
        # Baselines
        baselines = {'B1': BaselineB1(), 'B2': BaselineB2(), 'B3': BaselineB3(), 'B4': BaselineB4(), 'B5': BaselineB5()}
        for b_name, b_model in baselines.items():
            b_model.fit(train_df, y_train)
            preds = b_model.predict(val_df)
            m = calculate_metrics(y_val, preds)
            all_metrics.append({'Fold': i+1, 'Model': b_name, **m})
            
        # LightGBM (Using the already loaded bundle model to predict on the folds)
        # Note: In a true walk-forward we'd retrain per fold, but for purely evaluating the metrics
        # across folds, we'll re-train it properly here to get true fold-by-fold results as requested.
        from src.models.lgbm_fp import LGBMFantasyPointsModel
        lgbm = LGBMFantasyPointsModel(config['lgbm_params'])
        lgbm.features = features # force same feature ordering
        lgbm.fit(train_df, y_train, val_df, y_val)
        lgbm_preds = lgbm.predict(val_df)
        m = calculate_metrics(y_val, lgbm_preds)
        all_metrics.append({'Fold': i+1, 'Model': 'LightGBM', **m})
        
        # Accumulate validation info
        lgbm_all_preds.extend(lgbm_preds)
        y_all_actuals.extend(y_val.values)
        roles_all.extend(val_df['role'].values)
        history_all.extend(val_df['matches_played_before'].values)
        
    metrics_df = pd.DataFrame(all_metrics)
    
    print("\n--- 1. COMPLETE FOLD-BY-FOLD METRICS ---")
    print(metrics_df.to_string(index=False))
    
    print("\n--- AVERAGE METRICS ACROSS FOLDS ---")
    avg_metrics = metrics_df.groupby('Model')[['MAE', 'RMSE', 'R2']].mean().reset_index()
    print(avg_metrics.to_string(index=False))
    
    print("\n--- 2. LIGHTGBM VS BEST BASELINE ---")
    for i in range(1, len(folds)+1):
        fold_m = metrics_df[metrics_df['Fold'] == i]
        baselines_only = fold_m[fold_m['Model'].str.startswith('B')]
        best_b = baselines_only.loc[baselines_only['MAE'].idxmin()]
        lgbm_m = fold_m[fold_m['Model'] == 'LightGBM'].iloc[0]
        
        mae_imp = best_b['MAE'] - lgbm_m['MAE']
        pct_imp = (mae_imp / best_b['MAE']) * 100
        
        print(f"Fold {i}: Best Baseline is {best_b['Model']} (MAE {best_b['MAE']:.2f})")
        print(f"        LightGBM MAE {lgbm_m['MAE']:.2f}")
        verb = "Outperformed" if mae_imp > 0 else "Did not outperform"
        print(f"        {verb} by {mae_imp:.2f} absolute, {pct_imp:.2f}% improvement")
        
    print("\n--- 3. PREDICTION SANITY CHECK ---")
    lgbm_arr = np.array(lgbm_all_preds)
    actuals_arr = np.array(y_all_actuals)
    
    print(f"Min Prediction:     {np.min(lgbm_arr):.2f}")
    print(f"Max Prediction:     {np.max(lgbm_arr):.2f}")
    print(f"Mean Prediction:    {np.mean(lgbm_arr):.2f}")
    print(f"Median Prediction:  {np.median(lgbm_arr):.2f}")
    print(f"Actual Mean FP:     {np.mean(actuals_arr):.2f}")
    print(f"Actual Median FP:   {np.median(actuals_arr):.2f}")
    
    print("\n--- 4. ERROR ANALYSIS ---")
    error_df = pd.DataFrame({
        'Actual': actuals_arr,
        'Pred': lgbm_arr,
        'Role': roles_all,
        'History': history_all
    })
    error_df['MAE'] = np.abs(error_df['Actual'] - error_df['Pred'])
    
    print("\nMAE by Role:")
    print(error_df.groupby('Role')['MAE'].mean().to_string())
    
    def hist_bucket(x):
        if x == 0: return "Debut / 0 matches"
        elif 3 <= x <= 9: return "3-9 matches"
        elif x >= 10: return ">= 10 matches"
        else: return "1-2 matches"
        
    error_df['History_Bucket'] = error_df['History'].apply(hist_bucket)
    print("\nMAE by History Depth:")
    print(error_df.groupby('History_Bucket')['MAE'].mean().to_string())

if __name__ == '__main__':
    run_diagnostics()
