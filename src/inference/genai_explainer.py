import os
import json
import logging
import time

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

logger = logging.getLogger(__name__)

class GroundedGenAIExplainer:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = None
        if HAS_GENAI and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash', 
                generation_config={"response_mime_type": "application/json"})

    def _build_input_contract(self, dream_team, shap_explanations):
        selected_players = dream_team['team_df']['player'].tolist()
        c = dream_team['captain']
        vc = dream_team['vice_captain']
        
        contract = {
            "team_summary": {
                "total_expected_fp": dream_team['total_predicted_score'],
                "captain": c,
                "vice_captain": vc
            },
            "players": {}
        }
        
        for exp in shap_explanations:
            p = exp['player']
            if p in selected_players:
                contract["players"][p] = {
                    "player": p,
                    "team": exp['team'],
                    "role": exp['role'],
                    "p_play": round(exp['p_play'], 4),
                    "expected_fp": round(exp['expected_fp'], 2),
                    "is_captain": p == c,
                    "is_vice_captain": p == vc,
                    "top_positive_fp_factors": [f['feature'] for f in exp['top_positive_fp_factors'][:2]],
                    "top_negative_fp_factors": [f['feature'] for f in exp['top_negative_fp_factors'][:2]],
                    "top_positive_play_factors": [f['feature'] for f in exp['top_positive_play_factors'][:2]]
                }
        return contract

    def _build_prompt(self, contract):
        return f"""
        You are a strict, grounded AI assistant for a Dream11 fantasy cricket platform.
        Your ONLY job is to translate the provided structured ML pipeline output into readable natural language.
        
        CRITICAL RULES:
        1. GROUNDING: Do NOT invent any facts. You cannot mention pitch conditions, weather, form, or opponent match-ups UNLESS they are explicitly listed in the "factors" arrays below.
        2. NUMERICAL FAITHFULNESS: You MUST NOT change any numbers. P(play) and expected_fp must match exactly or be appropriately rounded (e.g. 0.85 -> 85%).
        3. NO PREDICTIONS: Do not predict match outcomes.
        
        INPUT DATA:
        {json.dumps(contract, indent=2)}
        
        OUTPUT FORMAT (JSON ONLY):
        Return a JSON object with this exact structure:
        {{
            "team_explanation": "Concise summary of the team's expected value and optimization.",
            "players": {{
                "Player Name": {{
                    "summary": "1 sentence overview.",
                    "why_selected": "Why the ILP optimizer selected them (based on expected FP and role).",
                    "p_play_explanation": "Translate p_play factors.",
                    "fp_explanation": "Translate FP factors.",
                    "expected_fp_explanation": "Expected FP = P(play) * FP if plays."
                }}
            }}
        }}
        """

    def _validate_and_fallback(self, contract, genai_output, shap_explanations):
        """
        Guardrail validation layer.
        Checks if the generated text contradicts the numerical inputs or hallucinates contexts.
        If hallucination is detected, falls back to deterministic.
        """
        validated = {}
        fallback_dict = {exp['player']: exp for exp in shap_explanations}
        
        forbidden_words = ['pitch', 'weather', 'rain', 'injury', 'injured', 'batting order', 'toss']
        
        for p, data in contract["players"].items():
            if not isinstance(genai_output, dict) or "players" not in genai_output or p not in genai_output["players"]:
                validated[p] = fallback_dict[p].get('human_readable_explanation', '')
                continue
                
            genai_text_dict = genai_output["players"][p]
            if not isinstance(genai_text_dict, dict):
                validated[p] = fallback_dict[p].get('human_readable_explanation', '')
                continue
                
            combined_text = " ".join([str(v) for v in genai_text_dict.values()]).lower()
            
            # Guardrail 1: Forbidden words
            if any(w in combined_text for w in forbidden_words):
                logger.warning(f"Hallucination (forbidden word) detected for {p}. Falling back.")
                validated[p] = fallback_dict[p].get('human_readable_explanation', '')
                continue
                
            # Guardrail 2: P(play) confirmation
            if "confirmed" in combined_text and data['p_play'] < 0.99:
                logger.warning(f"Hallucination (p_play confirmed) for {p}. Falling back.")
                validated[p] = fallback_dict[p].get('human_readable_explanation', '')
                continue
                
            # Guardrail 3: Numeric consistency check (rough expected FP match)
            # If the model hallucinates wildly different expected FP
            fp_base = str(int(data['expected_fp']))
            if fp_base not in combined_text and str(int(data['expected_fp']) + 1) not in combined_text:
                if "expected fp" in combined_text or "expected fantasy points" in combined_text:
                    logger.warning(f"Hallucination (numeric mismatch) for {p}. Expected {data['expected_fp']}. Falling back.")
                    validated[p] = fallback_dict[p].get('human_readable_explanation', '')
                    continue
                
            # Passed all guardrails
            md = f"**Summary**: {genai_text_dict.get('summary', '')}\n\n"
            md += f"**Why Selected**: {genai_text_dict.get('why_selected', '')}\n\n"
            md += f"**P(play) ({data['p_play']:.0%})**: {genai_text_dict.get('p_play_explanation', '')}\n"
            md += f"**FP Potential**: {genai_text_dict.get('fp_explanation', '')}\n"
            validated[p] = md
            
        return validated

    def explain(self, dream_team, shap_explanations):
        if not self.model:
            logger.info("GenAI Explainer bypassed: GEMINI_API_KEY not set.")
            return None, 0.0
            
        t0 = time.time()
        contract = self._build_input_contract(dream_team, shap_explanations)
        prompt = self._build_prompt(contract)
        
        try:
            response = self.model.generate_content(prompt)
            genai_output = json.loads(response.text)
            validated = self._validate_and_fallback(contract, genai_output, shap_explanations)
            latency = time.time() - t0
            return validated, latency
        except Exception as e:
            logger.error(f"GenAI generation failed: {e}. Falling back to deterministic.")
            return None, 0.0
