import pandas as pd
import logging
from src.optimization.ilp import optimize_team
from src.optimization.captain import select_captain_and_vc
from src.evaluation.csv_writer import write_prediction_csv

logger = logging.getLogger(__name__)

def generate_dream_team(candidate_players):
    """
    Orchestrates the selection of the optimal Dream11 team.
    Expects candidate_players to have: 'player', 'team', 'role', 'predicted_fantasy_points'
    """
    logger.info(f"Received {len(candidate_players)} candidates.")
    
    # 1. ILP Optimization
    selected_xi = optimize_team(candidate_players)
    
    # 2. Select C and VC
    final_xi = select_captain_and_vc(selected_xi)
    
    # 3. Post-Optimization Validation
    if len(final_xi) != 11:
        raise ValueError(f"Validation Error: Expected 11 players, got {len(final_xi)}")
        
    teams = final_xi['team'].value_counts()
    if len(teams) != 2:
        raise ValueError(f"Validation Error: Team representation must be exactly 2. Got: {teams.to_dict()}")
        
    if teams.max() > 7:
        raise ValueError(f"Validation Error: A single team has more than 7 players: {teams.to_dict()}")
        
    roles = final_xi['role'].value_counts()
    for r in ['BAT', 'BOWL', 'AR', 'WK']:
        count = roles.get(r, 0)
        if not (1 <= count <= 8):
            raise ValueError(f"Validation Error: Role {r} count {count} is out of bounds (1-8).")
            
    if final_xi['is_captain'].sum() != 1:
        raise ValueError("Validation Error: Must have exactly 1 Captain.")
        
    if final_xi['is_vice_captain'].sum() != 1:
        raise ValueError("Validation Error: Must have exactly 1 Vice-Captain.")
        
    if final_xi['player'].duplicated().any():
        raise ValueError("Validation Error: Duplicate players found in final XI.")
        
    c = final_xi[final_xi['is_captain'] == 1]['player'].iloc[0]
    vc = final_xi[final_xi['is_vice_captain'] == 1]['player'].iloc[0]
    if c == vc:
        raise ValueError("Validation Error: Captain and Vice-Captain cannot be the same player.")
        
    total_score = final_xi['optimized_points'].sum()
    
    result = {
        'team_df': final_xi.sort_values(by=['is_captain', 'is_vice_captain', 'role', 'player'], ascending=[False, False, True, True]),
        'total_predicted_score': total_score,
        'captain': c,
        'vice_captain': vc,
        'status': 'Optimal'
    }
    
    return result

if __name__ == '__main__':
    # Real-Data Integration Test
    logging.basicConfig(level=logging.INFO)
    from src.models.bundle import load_model_bundle
    from src.models.lgbm_fp import LGBMFantasyPointsModel
    
    # 1. Load data and model
    bundle = load_model_bundle("models/model_bundle.pkl")
    features_df = pd.read_parquet("data/processed/player_features.parquet")
    
    from src.optimization.pool_builder import build_reconstructed_pool
    
    # Pick a real match (e.g., a match from late 2023 or 2024 fold)
    match_counts = features_df['match_id'].value_counts()
    target_match = match_counts[match_counts == 22].index[-1] # Pick a recent match
    
    logger.info(f"Integration Test on Match ID: {target_match}")
    
    target_date = features_df[features_df['match_id'] == target_match]['date'].iloc[0]
    stats_df = pd.read_parquet("data/processed/player_match_stats.parquet")
    
    # For this target match, let's identify the actual teams to reconstruct
    # Note: We only look at the actual match to GET the teams playing that day.
    # We do NOT use the actual playing XI to build the pool.
    dels_df = pd.read_parquet("data/processed/deliveries.parquet")
    playing_teams = dels_df[dels_df['match_id'] == target_match]['batting_team'].unique()
    if len(playing_teams) != 2:
        logger.error(f"Found {len(playing_teams)} teams in match {target_match}: {playing_teams}")
        exit(1)
        
    team_A, team_B = playing_teams[0], playing_teams[1]
    
    logger.info(f"Reconstructing pre-toss candidate pool for {team_A} vs {team_B} on {target_date}...")
    candidate_pool = build_reconstructed_pool(target_date, team_A, team_B, stats_df, features_df)
    
    if candidate_pool.empty:
        logger.error("Reconstructed pool is empty. Please check data/dates.")
        exit(1)
        
    logger.info(f"Reconstructed pool size: {len(candidate_pool)}")
    
    # 2. Generate predictions using the loaded model
    model_wrapper = LGBMFantasyPointsModel(bundle['config']['lgbm_params'])
    model_wrapper.model = bundle['model']
    model_wrapper.features = bundle['features']
    
    candidate_pool['predicted_fantasy_points'] = model_wrapper.predict(candidate_pool)
    
    # 3. Optimize Dream Team
    candidates = candidate_pool[['player', 'team', 'role', 'predicted_fantasy_points']].copy()
    dt = generate_dream_team(candidates)
    
    logger.info(f"Status: {dt['status']}")
    logger.info(f"Captain: {dt['captain']}, VC: {dt['vice_captain']}")
    logger.info(f"Total Predicted Points: {dt['total_predicted_score']:.2f}")
    
    # 4. Save to CSV
    csv_path = write_prediction_csv(dt, match_date=target_date, team_A=team_A, team_B=team_B)
    logger.info(f"Saved predicted team to {csv_path}")
