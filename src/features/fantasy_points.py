import os
import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
CONFIG_PATH = "configs/scoring.yaml"

def load_scoring_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def calculate_fantasy_points(deliveries_df, matches_df):
    config = load_scoring_config().get('t20', {})
    bat_cfg = config.get('batting', {})
    bowl_cfg = config.get('bowling', {})
    field_cfg = config.get('fielding', {})
    econ_cfg = config.get('economy', {})
    sr_cfg = config.get('strike_rate', {})
    
    # 1. Batting Stats
    # Deliveries where batter faced a ball (exclude wides)
    deliveries_df['is_legal_ball_faced'] = (deliveries_df['is_wide'] == 0).astype(int)
    
    # Deliveries where bowler bowled a legal ball (exclude wides and no-balls)
    deliveries_df['is_legal_ball_bowled'] = ((deliveries_df['is_wide'] == 0) & (deliveries_df['is_noball'] == 0)).astype(int)
    
    bat_stats = deliveries_df.groupby(['match_id', 'batter']).agg(
        runs=pd.NamedAgg(column='batter_runs', aggfunc='sum'),
        balls=pd.NamedAgg(column='is_legal_ball_faced', aggfunc='sum'), # Wides don't count
        fours=pd.NamedAgg(column='batter_runs', aggfunc=lambda x: (x == 4).sum()),
        sixes=pd.NamedAgg(column='batter_runs', aggfunc=lambda x: (x == 6).sum())
    ).reset_index()
    
    bat_stats['bat_pts'] = bat_stats['runs'] * bat_cfg.get('run', 1)
    bat_stats['bat_pts'] += bat_stats['fours'] * bat_cfg.get('boundary_bonus', 1)
    bat_stats['bat_pts'] += bat_stats['sixes'] * bat_cfg.get('six_bonus', 2)
    bat_stats['bat_pts'] += (bat_stats['runs'] >= 30).astype(int) * bat_cfg.get('thirty_run_bonus', 4)
    bat_stats['bat_pts'] += (bat_stats['runs'] >= 50).astype(int) * bat_cfg.get('fifty_run_bonus', 8)
    bat_stats['bat_pts'] += (bat_stats['runs'] >= 100).astype(int) * bat_cfg.get('hundred_run_bonus', 16)
    
    # Strike rate
    bat_stats['sr'] = 0.0
    mask_balls = bat_stats['balls'] > 0
    bat_stats.loc[mask_balls, 'sr'] = (bat_stats.loc[mask_balls, 'runs'] / bat_stats.loc[mask_balls, 'balls']) * 100
    
    def get_sr_bonus(row):
        if row['balls'] < 10: return 0
        sr = row['sr']
        if sr > 170: return sr_cfg.get('above_170', 6)
        elif 150 < sr <= 170: return sr_cfg.get('between_150_and_170', 4)
        elif 130 <= sr <= 150: return sr_cfg.get('between_130_and_150', 2)
        elif 60 <= sr <= 70: return sr_cfg.get('between_60_and_70', -2)
        elif 50 <= sr < 60: return sr_cfg.get('between_50_and_59_99', -4)
        elif sr < 50: return sr_cfg.get('below_50', -6)
        return 0
        
    bat_stats['sr_pts'] = bat_stats.apply(get_sr_bonus, axis=1)
    bat_stats['bat_pts'] += bat_stats['sr_pts']
    
    # Duck penalty (Needs role info to exclude pure bowlers, ignoring for MVP as per instructions)
    bat_stats.loc[(bat_stats['runs'] == 0) & (bat_stats['balls'] > 0), 'bat_pts'] += bat_cfg.get('duck_penalty', -2)
    
    # 2. Bowling Stats
    # Bowler is not debited for byes and legbyes.
    # Total runs - byes - legbyes = bowler runs conceded.
    deliveries_df['bowler_runs_conceded'] = deliveries_df['total_runs'] - (deliveries_df['is_bye'] * deliveries_df['extras']) - (deliveries_df['is_legbye'] * deliveries_df['extras'])
    
    bowl_stats = deliveries_df.groupby(['match_id', 'bowler']).agg(
        runs_conceded=pd.NamedAgg(column='bowler_runs_conceded', aggfunc='sum'),
        balls_bowled=pd.NamedAgg(column='is_legal_ball_bowled', aggfunc='sum'),
        wickets=pd.NamedAgg(column='is_wicket', aggfunc='sum')
    ).reset_index()

    
    bowl_stats['overs'] = bowl_stats['balls_bowled'] / 6
    bowl_stats['bowl_pts'] = bowl_stats['wickets'] * bowl_cfg.get('wicket', 25)
    bowl_stats['bowl_pts'] += (bowl_stats['wickets'] == 4).astype(int) * bowl_cfg.get('four_wicket_bonus', 8)
    bowl_stats['bowl_pts'] += (bowl_stats['wickets'] >= 5).astype(int) * bowl_cfg.get('five_wicket_bonus', 16)
    
    # Economy
    bowl_stats['econ'] = bowl_stats['runs_conceded'] / bowl_stats['overs']
    
    def get_econ_bonus(row):
        if row['overs'] < 2: return 0
        econ = row['econ']
        if econ < 5: return econ_cfg.get('below_5', 6)
        elif 5 <= econ < 6: return econ_cfg.get('between_5_and_5_99', 4)
        elif 6 <= econ <= 7: return econ_cfg.get('between_6_and_7', 2)
        elif 10 <= econ <= 11: return econ_cfg.get('between_10_and_11', -2)
        elif 11 < econ <= 12: return econ_cfg.get('between_11_and_12', -4)
        elif econ > 12: return econ_cfg.get('above_12', -6)
        return 0
        
    bowl_stats['econ_pts'] = bowl_stats.apply(get_econ_bonus, axis=1)
    bowl_stats['bowl_pts'] += bowl_stats['econ_pts']
    
    # 3. Fielding Stats
    fielding_df = deliveries_df[deliveries_df['fielder'].notnull()]
    field_stats = fielding_df.groupby(['match_id', 'fielder']).agg(
        catches=pd.NamedAgg(column='dismissal_kind', aggfunc=lambda x: (x == 'caught').sum()),
        stumpings=pd.NamedAgg(column='dismissal_kind', aggfunc=lambda x: (x == 'stumped').sum()),
        runouts=pd.NamedAgg(column='dismissal_kind', aggfunc=lambda x: (x == 'run out').sum())
    ).reset_index()
    
    field_stats['field_pts'] = field_stats['catches'] * field_cfg.get('catch', 8)
    field_stats['field_pts'] += (field_stats['catches'] >= 3).astype(int) * field_cfg.get('three_catch_bonus', 4)
    field_stats['field_pts'] += field_stats['stumpings'] * field_cfg.get('stumping', 12)
    field_stats['field_pts'] += field_stats['runouts'] * field_cfg.get('run_out_direct', 12) # Simplified
    
    # Merge all
    # We need a master list of players per match
    all_players = set()
    for df, col in [(bat_stats, 'batter'), (bowl_stats, 'bowler'), (field_stats, 'fielder')]:
        for _, row in df.iterrows():
            all_players.add((row['match_id'], row[col]))
            
    pms = pd.DataFrame(list(all_players), columns=['match_id', 'player'])
    pms = pms.merge(matches_df[['match_id', 'date', 'format']], on='match_id', how='left')
    
    pms = pms.merge(bat_stats, left_on=['match_id', 'player'], right_on=['match_id', 'batter'], how='left').drop(columns=['batter'])
    pms = pms.merge(bowl_stats, left_on=['match_id', 'player'], right_on=['match_id', 'bowler'], how='left').drop(columns=['bowler'])
    pms = pms.merge(field_stats, left_on=['match_id', 'player'], right_on=['match_id', 'fielder'], how='left').drop(columns=['fielder'])
    
    pms.fillna(0, inplace=True)
    
    # Total FP
    pms['fantasy_points'] = pms['bat_pts'] + pms['bowl_pts'] + pms['field_pts'] + config.get('playing_xi', {}).get('played', 4)
    
    return pms

def generate_player_match_stats():
    matches_path = os.path.join(PROCESSED_DIR, 'matches.parquet')
    deliveries_path = os.path.join(PROCESSED_DIR, 'deliveries.parquet')
    
    if not os.path.exists(matches_path): return
    
    matches_df = pd.read_parquet(matches_path)
    deliveries_df = pd.read_parquet(deliveries_path)
    
    pms = calculate_fantasy_points(deliveries_df, matches_df)
    
    pms.to_parquet(os.path.join(PROCESSED_DIR, 'player_match_stats.parquet'), index=False)
    logger.info(f"Generated player_match_stats.parquet with {len(pms)} rows.")

if __name__ == "__main__":
    generate_player_match_stats()
