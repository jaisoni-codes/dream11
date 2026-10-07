import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

def print_metrics_table(results_list):
    df = pd.DataFrame(results_list)
    print("\n" + "="*60)
    print("MODEL VALIDATION METRICS")
    print("="*60)
    if not df.empty:
        # Reorder columns
        cols = ['Model', 'Fold', 'MAE', 'RMSE', 'R2']
        print(df[cols].to_string(index=False))
    else:
        print("No results to display.")
    print("="*60 + "\n")
