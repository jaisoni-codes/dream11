import os
import logging
import pandas as pd
from src.features.rolling import compute_rolling_features
from src.features.form import build_form_features
from src.features.role_opp import infer_roles_and_opportunity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"

def run_feature_pipeline():
    pms_path = os.path.join(PROCESSED_DIR, 'player_match_stats.parquet')
    if not os.path.exists(pms_path):
        logger.error(f"{pms_path} not found.")
        return
        
    logger.info("Loading player match stats...")
    pms_df = pd.read_parquet(pms_path)
    
    logger.info("Computing rolling features...")
    rolling_df = compute_rolling_features(pms_df)
    
    logger.info("Computing form features...")
    form_df = build_form_features(pms_df, rolling_df)
    
    logger.info("Inferring roles and opportunity...")
    final_df = infer_roles_and_opportunity(pms_df, form_df)
    
    out_path = os.path.join(PROCESSED_DIR, 'player_features.parquet')
    final_df.to_parquet(out_path, index=False)
    
    logger.info(f"Feature dataset saved to {out_path} with shape {final_df.shape}")

if __name__ == '__main__':
    run_feature_pipeline()
