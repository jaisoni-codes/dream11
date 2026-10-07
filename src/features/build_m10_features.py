import os
import logging
import pandas as pd
from src.features.contextual import build_contextual_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"

def run_m10_feature_pipeline():
    feats_path = os.path.join(PROCESSED_DIR, 'player_features.parquet')
    pms_path = os.path.join(PROCESSED_DIR, 'player_match_stats.parquet')
    matches_path = os.path.join(PROCESSED_DIR, 'matches.parquet')
    dels_path = os.path.join(PROCESSED_DIR, 'deliveries.parquet')
    
    logger.info("Loading baseline features...")
    baseline_df = pd.read_parquet(feats_path)
    pms_df = pd.read_parquet(pms_path)
    matches_df = pd.read_parquet(matches_path)
    dels_df = pd.read_parquet(dels_path)
    
    logger.info("Computing M10 Contextual Features...")
    ctx_df = build_contextual_features(pms_df, matches_df, dels_df)
    
    final_df = baseline_df.merge(ctx_df, on=['match_id', 'player'], how='left')
    
    # Fill any remaining NAs with 0
    final_df['opp_avg_pts'] = final_df['opp_avg_pts'].fillna(0)
    final_df['venue_avg_pts'] = final_df['venue_avg_pts'].fillna(0)
    
    out_path = os.path.join(PROCESSED_DIR, 'player_features_m10.parquet')
    final_df.to_parquet(out_path, index=False)
    logger.info(f"M10 Feature dataset saved to {out_path} with shape {final_df.shape}")

if __name__ == '__main__':
    run_m10_feature_pipeline()
