import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"

# Example mapping of known aliases (in a real system, this would be a large file)
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad", # Technically different franchises, but grouped often. Let's keep them distinct for safety.
}

def normalise_data():
    matches_path = os.path.join(PROCESSED_DIR, 'matches.parquet')
    deliveries_path = os.path.join(PROCESSED_DIR, 'deliveries.parquet')
    
    if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
        logger.error("Parsed data not found. Run parse_cricsheet.py first.")
        return
        
    matches_df = pd.read_parquet(matches_path)
    deliveries_df = pd.read_parquet(deliveries_path)
    
    # Clean whitespace and case
    for col in ['team1', 'team2', 'venue', 'city']:
        matches_df[col] = matches_df[col].str.strip()
    
    # Apply aliases safely
    matches_df['team1'] = matches_df['team1'].replace(TEAM_ALIASES)
    matches_df['team2'] = matches_df['team2'].replace(TEAM_ALIASES)
    
    deliveries_df['batting_team'] = deliveries_df['batting_team'].replace(TEAM_ALIASES)
    
    # Save back
    matches_df.to_parquet(matches_path, index=False)
    deliveries_df.to_parquet(deliveries_path, index=False)
    logger.info("Normalisation complete.")

if __name__ == "__main__":
    normalise_data()
