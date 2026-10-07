import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.optimization.pool_builder import build_reconstructed_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals for workers
_pms_df = None
_dels_df = None
_features_df = None

def init_worker():
    global _pms_df, _dels_df, _features_df
    _pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
    _dels_df = pd.read_parquet('data/processed/deliveries.parquet')
    _features_df = pd.read_parquet('data/processed/player_features.parquet')
    _pms_df['date'] = pd.to_datetime(_pms_df['date'])
    _features_df['date'] = pd.to_datetime(_features_df['date'])

def process_match(task):
    match_id, match_date, team_A, team_B = task
    try:
        pool = build_reconstructed_pool(match_date, team_A, team_B, _pms_df, _features_df)
        if pool.empty:
            return None
            
        actual_players = set(_pms_df[_pms_df['match_id'] == match_id]['player'])
        pool['y_play'] = pool['player'].apply(lambda x: 1 if x in actual_players else 0)
        
        if 'date' in pool.columns:
            pool['days_since_last_match'] = (pd.to_datetime(match_date) - pd.to_datetime(pool['date'])).dt.days
        else:
            pool['days_since_last_match'] = -1
            
        pool['target_match_id'] = match_id
        pool['target_match_date'] = match_date
        return pool
    except Exception as e:
        return None

def build_dataset_parallel():
    logger.info("Loading base data for main process to get matches...")
    pms_df = pd.read_parquet('data/processed/player_match_stats.parquet')
    dels_df = pd.read_parquet('data/processed/deliveries.parquet')
    pms_df['date'] = pd.to_datetime(pms_df['date'])
    
    # Ablation requires back to 2005. Get ALL matches < 2024-07-01
    matches = pms_df[pms_df['date'] < '2024-07-01'][['match_id', 'date']].drop_duplicates().sort_values('date')
    
    # Group deliveries by match_id for fast lookup
    dels_grouped = dels_df.groupby('match_id')['batting_team'].unique().to_dict()
    
    tasks = []
    for idx, row in matches.iterrows():
        match_id = row['match_id']
        match_date = row['date']
        teams = dels_grouped.get(match_id, [])
        if len(teams) == 2:
            tasks.append((match_id, match_date, teams[0], teams[1]))
            
    logger.info(f"Prepared {len(tasks)} tasks.")
    
    t0 = time.time()
    num_cpus = cpu_count()
    logger.info(f"Using {num_cpus} CPUs")
    
    # We will use imap_unordered for progress
    results = []
    with Pool(processes=num_cpus, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(process_match, tasks)):
            if res is not None:
                results.append(res)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                logger.info(f"Processed {i+1}/{len(tasks)} matches in {elapsed:.1f}s")
                
    final_df = pd.concat(results, ignore_index=True)
    logger.info(f"Dataset generated with shape {final_df.shape}")
    final_df.to_parquet('data/processed/play_prob_dataset_full.parquet')
    logger.info("Saved to play_prob_dataset_full.parquet")

if __name__ == '__main__':
    build_dataset_parallel()
