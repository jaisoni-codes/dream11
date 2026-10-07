import pulp
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def optimize_team(candidates_df):
    """
    Optimizes Dream11 team selection using ILP.
    Accepts a candidate pool DataFrame.
    """
    required_cols = {'player', 'team', 'role', 'predicted_fantasy_points'}
    missing = required_cols - set(candidates_df.columns)
    if missing:
        raise ValueError(f"Candidate pool missing required columns: {missing}")
        
    # Validate pool size
    if len(candidates_df) < 11:
        raise ValueError(f"Infeasible: Only {len(candidates_df)} players available. Minimum 11 required.")
        
    # Validate teams
    teams = candidates_df['team'].unique()
    if len(teams) != 2:
        raise ValueError(f"Infeasible: Candidate pool must contain exactly 2 teams. Found {len(teams)}: {teams}")
        
    team_A, team_B = teams[0], teams[1]
    
    # Validate roles
    valid_roles = {'WK', 'BAT', 'AR', 'BOWL'}
    invalid_roles = set(candidates_df['role'].unique()) - valid_roles
    if invalid_roles:
        raise ValueError(f"Invalid roles detected: {invalid_roles}. Only WK, BAT, AR, BOWL are permitted.")
        
    for r in valid_roles:
        if len(candidates_df[candidates_df['role'] == r]) < 1:
            raise ValueError(f"Infeasible: No players available for role {r}.")

    # Create problem
    prob = pulp.LpProblem("Dream11_Optimization", pulp.LpMaximize)
    
    # Decision variables
    player_vars = pulp.LpVariable.dicts("player", candidates_df.index, cat='Binary')
    
    # Objective: Maximize predicted fantasy points
    prob += pulp.lpSum([candidates_df.loc[i, 'predicted_fantasy_points'] * player_vars[i] for i in candidates_df.index]), "Total_Predicted_Points"
    
    # Constraint 1: Exactly 11 players
    prob += pulp.lpSum([player_vars[i] for i in candidates_df.index]) == 11, "Total_Players"
    
    # Constraint 2: Teams (Max 7, Min 1 per team)
    prob += pulp.lpSum([player_vars[i] for i in candidates_df.index if candidates_df.loc[i, 'team'] == team_A]) <= 7, f"Max_{team_A}"
    prob += pulp.lpSum([player_vars[i] for i in candidates_df.index if candidates_df.loc[i, 'team'] == team_A]) >= 1, f"Min_{team_A}"
    
    prob += pulp.lpSum([player_vars[i] for i in candidates_df.index if candidates_df.loc[i, 'team'] == team_B]) <= 7, f"Max_{team_B}"
    prob += pulp.lpSum([player_vars[i] for i in candidates_df.index if candidates_df.loc[i, 'team'] == team_B]) >= 1, f"Min_{team_B}"
    
    # Constraint 3: Roles (1 to 8 per role)
    for r in valid_roles:
        prob += pulp.lpSum([player_vars[i] for i in candidates_df.index if candidates_df.loc[i, 'role'] == r]) >= 1, f"Min_{r}"
        prob += pulp.lpSum([player_vars[i] for i in candidates_df.index if candidates_df.loc[i, 'role'] == r]) <= 8, f"Max_{r}"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise ValueError(f"Infeasible: Solver could not find an optimal solution. Status: {pulp.LpStatus[prob.status]}")
        
    selected_indices = [i for i in candidates_df.index if player_vars[i].varValue == 1.0]
    
    # Validation
    if len(selected_indices) != 11:
        raise ValueError(f"Post-solve validation failed: Selected {len(selected_indices)} players instead of 11.")
        
    return candidates_df.loc[selected_indices].copy()
