import shap
import pandas as pd
import numpy as np
import time

class SHAPExplainer:
    def __init__(self, fp_model, fp_features, play_model, play_features):
        """
        Initializes the SHAP explainers for both models.
        fp_model, play_model: Underling lightgbm booster/model objects.
        """
        # For lightgbm, if it's a Booster or LGBMRegressor/Classifier, TreeExplainer works.
        self.fp_explainer = shap.TreeExplainer(fp_model)
        self.play_explainer = shap.TreeExplainer(play_model)
        
        self.fp_features = fp_features
        self.play_features = play_features

    def _get_top_factors(self, shap_vals, feature_names, feature_values, top_n=3):
        factors = []
        for val, name, f_val in zip(shap_vals, feature_names, feature_values):
            factors.append({
                "feature": name,
                "value": float(f_val) if isinstance(f_val, (int, float, np.number)) else str(f_val),
                "contribution": float(val)
            })
        
        # Sort by absolute contribution for finding the most impactful
        factors = sorted(factors, key=lambda x: abs(x["contribution"]), reverse=True)
        
        positive_factors = [f for f in factors if f["contribution"] > 0][:top_n]
        negative_factors = [f for f in factors if f["contribution"] < 0][:top_n]
        
        return positive_factors, negative_factors

    def generate_human_readable(self, exp):
        """
        Deterministic template-based explanation generation.
        """
        player = exp['player']
        fp_pos = exp['top_positive_fp_factors']
        fp_neg = exp['top_negative_fp_factors']
        play_pos = exp['top_positive_play_factors']
        play_neg = exp['top_negative_play_factors']
        
        fp_reasons = []
        if fp_pos:
            fp_reasons.append(f"strong positive indicators like {fp_pos[0]['feature']} (contrib: {fp_pos[0]['contribution']:.1f})")
        if fp_neg:
            fp_reasons.append(f"offsets from {fp_neg[0]['feature']} (contrib: {fp_neg[0]['contribution']:.1f})")
            
        fp_text = " and ".join(fp_reasons) if fp_reasons else "average baseline features"
        
        play_reasons = []
        if play_pos:
            play_reasons.append(f"{play_pos[0]['feature']}")
        if play_neg:
            play_reasons.append(f"negative pressure from {play_neg[0]['feature']}")
            
        play_text = ", mainly due to " + " and ".join(play_reasons) if play_reasons else ""
        
        text = f"{player} has a predicted FP of {exp['predicted_fp']:.1f} because of {fp_text}. "
        text += f"The play model assigns a {exp['p_play']:.2f} probability of playing{play_text}. "
        text += f"Final Expected FP: {exp['expected_fp']:.1f}."
        
        return text

    def explain(self, pool_df, X_fp, X_play):
        """
        Generates explanations for the provided candidate pool.
        Returns a list of dictionary explanations.
        """
        start_time = time.time()
        
        # Calculate SHAP values
        fp_shap_values = self.fp_explainer.shap_values(X_fp)
        play_shap_values = self.play_explainer.shap_values(X_play)
        
        # Handle lightgbm multi-class/binary format (list of arrays or single array)
        if isinstance(play_shap_values, list):
            play_shap_values = play_shap_values[1] # Take positive class for binary classification
            
        if isinstance(fp_shap_values, list):
            fp_shap_values = fp_shap_values[0] # Usually regressor is a single array, but just in case
            
        explanations = []
        for i, row in pool_df.iterrows():
            fp_pos, fp_neg = self._get_top_factors(
                fp_shap_values[i], self.fp_features, X_fp.iloc[i].values
            )
            play_pos, play_neg = self._get_top_factors(
                play_shap_values[i], self.play_features, X_play.iloc[i].values
            )
            
            exp = {
                "player": row['player'],
                "team": row.get('team', 'Unknown'),
                "role": row.get('role', 'Unknown'),
                "p_play": float(row['play_prob']),
                "predicted_fp": float(row['predicted_fp_raw']),
                "expected_fp": float(row['expected_fp']),
                "top_positive_fp_factors": fp_pos,
                "top_negative_fp_factors": fp_neg,
                "top_positive_play_factors": play_pos,
                "top_negative_play_factors": play_neg
            }
            exp["human_readable_explanation"] = self.generate_human_readable(exp)
            explanations.append(exp)
            
        latency = time.time() - start_time
        return explanations, latency
