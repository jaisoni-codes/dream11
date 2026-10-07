import pandas as pd
import numpy as np
from datetime import timedelta
import time
import logging
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.optimization.pool_builder import build_reconstructed_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_match(args):
    match_id, match_date, team_A, team_B, pms_subset, feats_subset = args
    try:
        # Reconstruct pool
        pool = build_reconstructed_pool(match_date, team_A, team_B, pms_subset, feats_subset)
        if pool.empty:
            return None
            
        # Actual players
        actual_players = set(pms_subset[pms_subset['match_id'] == match_id]['player'])
        
        # Assign target
        pool['y_play'] = pool['player'].apply(lambda x: 1 if x in actual_players else 0)
        
        # Compute days since last match for the player
        if 'date' in pool.columns:
            pool['days_since_last_match'] = (pd.to_datetime(match_date) - pd.to_datetime(pool['date'])).dt.days
        else:
            pool['days_since_last_match'] = -1
            
        pool['target_match_id'] = match_id
        pool['target_match_date'] = match_date
        
        return pool
    except Exception as e:
        logger.error(f"Error on {match_id}: {e}")
        return None

def build_dataset():
    logger.info("Loading base data...")
    pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
    dels_df = pd.read_parquet('data/processed/deliveries.parquet')
    features_df = pd.read_parquet('data/processed/player_features.parquet')
    
    pms_df['date'] = pd.to_datetime(pms_df['date'])
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    # Select matches to process (training data < 2024-07-01)
    matches = pms_df[(pms_df['date'] >= '2021-01-01') & (pms_df['date'] < '2024-07-01')][['match_id', 'date']].drop_duplicates().sort_values('date')
    
    logger.info(f"Generating candidate pools for {len(matches)} matches...")
    
    tasks = []
    for idx, row in matches.iterrows():
        match_id = row['match_id']
        match_date = row['date']
        
        match_dels = dels_df[dels_df['match_id'] == match_id]
        playing_teams = match_dels['batting_team'].unique()
        if len(playing_teams) == 2:
            tasks.append((match_id, match_date, playing_teams[0], playing_teams[1], pms_df, features_df))
            
    logger.info(f"Prepared {len(tasks)} tasks.")
    
    results = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        res = process_match(task)
        if res is not None:
            results.append(res)
        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(f"Processed {i+1}/{len(tasks)} matches in {elapsed:.1f}s")
            
    final_df = pd.concat(results, ignore_index=True)
    logger.info(f"Dataset generated with shape {final_df.shape}")
    logger.info(f"Target distribution: {final_df['y_play'].value_counts(normalize=True).to_dict()}")
    
    final_df.to_parquet('data/processed/play_prob_dataset.parquet')
    logger.info("Saved to data/processed/play_prob_dataset.parquet")

if __name__ == '__main__':
    build_dataset()
