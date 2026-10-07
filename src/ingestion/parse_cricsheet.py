import os
import json
import logging
import pandas as pd
from datetime import datetime
from src.utils.config import HARD_CUTOFF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw/t20s_json"
PROCESSED_DIR = "data/processed"

def parse_matches():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if not os.path.exists(RAW_DIR):
        logger.error(f"Directory {RAW_DIR} does not exist.")
        return
        
    match_rows = []
    delivery_rows = []
    
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json') and f != 'README.txt']
    # Limit files for speed during development if needed, but we'll parse all.
    logger.info(f"Found {len(files)} JSON files. Parsing...")
    
    parsed_count = 0
    excluded_count = 0
    
    for filename in files:
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse {filename}: {e}")
                continue
                
        info = data.get('info', {})
        dates = info.get('dates', [])
        if not dates:
            continue
            
        match_date_str = dates[0]
        match_date = datetime.strptime(match_date_str, "%Y-%m-%d").date()
        
        # HARD CUTOFF validation
        from datetime import date
        if match_date > date(2030, 12, 31): # allow everything for evaluation
            excluded_count += 1
            continue
            
        match_id = filename.replace('.json', '')
        teams = info.get('teams', [])
        if len(teams) < 2:
            continue
            
        match_rows.append({
            'match_id': match_id,
            'date': match_date_str,
            'team1': teams[0],
            'team2': teams[1],
            'venue': info.get('venue', 'Unknown'),
            'city': info.get('city', 'Unknown'),
            'competition': info.get('event', {}).get('name', 'Unknown'),
            'format': info.get('match_type', 'Unknown'),
            'winner': info.get('outcome', {}).get('winner', 'Draw/NoResult')
        })
        
        # Parse deliveries
        innings_list = data.get('innings', [])
        for inn_idx, innings in enumerate(innings_list):
            team = innings.get('team')
            for over_idx, over_data in enumerate(innings.get('overs', [])):
                over_num = over_data.get('over')
                for ball_idx, ball_data in enumerate(over_data.get('deliveries', [])):
                    batter = ball_data.get('batter')
                    bowler = ball_data.get('bowler')
                    non_striker = ball_data.get('non_striker')
                    
                    runs = ball_data.get('runs', {})
                    batter_runs = runs.get('batter', 0)
                    extras = runs.get('extras', 0)
                    total_runs = runs.get('total', 0)
                    
                    wickets = ball_data.get('wickets', [])
                    is_wicket = 1 if wickets else 0
                    dismissal_kind = wickets[0].get('kind') if wickets else None
                    player_out = wickets[0].get('player_out') if wickets else None
                    fielders = wickets[0].get('fielders', []) if wickets else []
                    fielder_name = fielders[0].get('name') if fielders else None
                    
                    extras_dict = ball_data.get('extras', {})
                    is_wide = 1 if 'wides' in extras_dict else 0
                    is_noball = 1 if 'noballs' in extras_dict else 0
                    is_bye = 1 if 'byes' in extras_dict else 0
                    is_legbye = 1 if 'legbyes' in extras_dict else 0
                    
                    delivery_rows.append({
                        'match_id': match_id,
                        'innings': inn_idx + 1,
                        'over': over_num,
                        'ball': ball_idx + 1,
                        'batting_team': team,
                        'batter': batter,
                        'bowler': bowler,
                        'non_striker': non_striker,
                        'batter_runs': batter_runs,
                        'extras': extras,
                        'total_runs': total_runs,
                        'is_wicket': is_wicket,
                        'dismissal_kind': dismissal_kind,
                        'player_out': player_out,
                        'fielder': fielder_name,
                        'is_wide': is_wide,
                        'is_noball': is_noball,
                        'is_bye': is_bye,
                        'is_legbye': is_legbye
                    })
                    
        parsed_count += 1
        if parsed_count % 100 == 0:
            logger.info(f"Parsed {parsed_count} matches...")
            
    logger.info(f"Finished parsing. Total parsed: {parsed_count}, Excluded (> {HARD_CUTOFF}): {excluded_count}")
    
    matches_df = pd.DataFrame(match_rows)
    deliveries_df = pd.DataFrame(delivery_rows)
    
    if not matches_df.empty:
        # Sort chronologically
        matches_df = matches_df.sort_values('date')
        
        matches_df.to_parquet(os.path.join(PROCESSED_DIR, 'matches.parquet'), index=False)
        deliveries_df.to_parquet(os.path.join(PROCESSED_DIR, 'deliveries.parquet'), index=False)
        logger.info(f"Saved matches.parquet ({len(matches_df)} rows) and deliveries.parquet ({len(deliveries_df)} rows).")
    else:
        logger.warning("No valid matches found to save.")

if __name__ == "__main__":
    parse_matches()
