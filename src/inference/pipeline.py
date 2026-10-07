import pandas as pd
import numpy as np
import time
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.optimization.pool_builder import build_reconstructed_pool
from src.optimization.dream_team import generate_dream_team
from src.models.lgbm_fp import LGBMFantasyPointsModel

class PreTossInferencePipeline:
    def __init__(self, fp_model_path='models/model_bundle.pkl', play_model_path='models/play_probability_bundle.pkl'):
        # Load FP Model - handle both standard and ablation bundle formats
        self.fp_bundle = joblib.load(fp_model_path)
        # Ablation bundles may not have 'config'; use empty dict as fallback
        fp_config = self.fp_bundle.get('config', {})
        lgbm_params = fp_config.get('lgbm_params', {}) if fp_config else {}
        self.fp_model = LGBMFantasyPointsModel(lgbm_params)
        self.fp_model.model = self.fp_bundle['model']
        self.fp_model.features = self.fp_bundle['features']
        
        # Load Play Prob Model
        self.prob_bundle = joblib.load(play_model_path)
        self.prob_model = self.prob_bundle['model']
        self.prob_features = self.prob_bundle['features']
        
    def predict(self, target_date, team_a, team_b, pms_df, features_df, explain=False):
        """
        Single end-to-end inference path.
        Strictly requires no target-match outcome data.
        """
        latencies = {}
        
        # 1. Candidate Pool Construction
        t0 = time.time()
        pool = build_reconstructed_pool(target_date, team_a, team_b, pms_df, features_df)
        latencies['pool_construction'] = time.time() - t0
        
        if pool.empty or len(pool) < 11:
            raise ValueError("Insufficient candidates in reconstructed pool.")
            
        # 2. M_fp Prediction
        t0 = time.time()
        pool['predicted_fp_raw'] = self.fp_model.predict(pool)
        latencies['m_fp_inference'] = time.time() - t0
        
        # 3. M_play Prediction
        t0 = time.time()
        prob_pool = pool.copy()
        if 'date' in prob_pool.columns:
            prob_pool['days_since_last_match'] = (pd.to_datetime(target_date) - pd.to_datetime(prob_pool['date'])).dt.days
        else:
            prob_pool['days_since_last_match'] = -1
            
        if 'role' in prob_pool.columns:
            prob_pool['role_cat'] = prob_pool['role'].astype('category').cat.codes
        if 'role_confidence' in prob_pool.columns:
            prob_pool['role_confidence'] = prob_pool['role_confidence'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1)
            
        for f in self.prob_features:
            if f not in prob_pool.columns:
                prob_pool[f] = 0
                
        # Ensure parity: correct order of features
        X_prob = prob_pool[self.prob_features]
        # Use predict_proba[:, 1] to get P(play=1), not hard class labels
        if hasattr(self.prob_model, 'predict_proba'):
            pool['play_prob'] = self.prob_model.predict_proba(X_prob)[:, 1]
        else:
            pool['play_prob'] = self.prob_model.predict(X_prob)
        latencies['m_play_inference'] = time.time() - t0
        
        # 4. Expected FP
        pool['expected_fp'] = pool['predicted_fp_raw'] * pool['play_prob']
        
        # 5. ILP Optimization
        t0 = time.time()
        # We temporarily alias expected_fp to predicted_fantasy_points for the optimizer
        pool['predicted_fantasy_points'] = pool['expected_fp']
        dream_team = generate_dream_team(pool)
        latencies['ilp_optimization'] = time.time() - t0
        
        # 6. SHAP Explanation (Optional)
        if explain:
            from src.inference.explainer import SHAPExplainer
            if not hasattr(self, 'explainer'):
                self.explainer = SHAPExplainer(
                    fp_model=self.fp_model.model,
                    fp_features=self.fp_model.features,
                    play_model=self.prob_model,
                    play_features=self.prob_features
                )
            # Prepare exact X_fp and X_play that models saw
            X_fp = self.fp_model._prepare_features(pool.copy())[self.fp_model.features]
            X_play = prob_pool[self.prob_features]
            
            explanations, lat_shap = self.explainer.explain(pool, X_fp, X_play)
            latencies['shap_explanation'] = lat_shap
            dream_team['explanations'] = explanations
            
        latencies['total_inference'] = sum(latencies.values())
        
        return pool, dream_team, latencies
