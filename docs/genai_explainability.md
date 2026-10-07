# Milestone 11: Grounded GenAI Explanation Layer

## 1. Overview
The GenAI Explanation Layer is a strictly grounded natural-language wrapper around the deterministic ML pipeline outputs. It provides an intuitive summary of the SHAP (SHapley Additive exPlanations) factors and the ILP optimizer decisions.

**Crucially**, GenAI is NEVER used for:
- Predicting fantasy points
- Calculating play probabilities
- Selecting the Best XI or Captains
- Inventing cricket facts or pitch conditions

The ML numerical output is strictly authoritative.

## 2. Architecture
```
ML Pipeline (Frozen M7) -> ILP Optimizer -> SHAP Local Explainer -> Structured JSON -> GenAI (Google Gemini) -> Validated Markdown -> UI
```
The integration occurs purely at the presentation tier (`src/ui/product_page.py` -> `src/inference/genai_explainer.py`).

## 3. Input Schema Contract
GenAI receives a strict JSON contract containing ONLY verified data:
- `p_play`
- `expected_fp`
- `is_captain`, `is_vice_captain`
- `top_positive_fp_factors`, `top_negative_fp_factors`
- `top_positive_play_factors`

## 4. Grounding and Guardrails
The system prompt strictly commands the model to ONLY use the provided JSON features and strictly prohibits hallucinations about match conditions. 

**Guardrail Mechanism**: After GenAI outputs the JSON response, a deterministic post-processor checks the text. For example, if a player's `p_play` is 0.85, but the GenAI claims the player is "Confirmed to play" or "100% playing", the validation fails, and the system falls back to the deterministic M8 SHAP text.

## 5. Fallback Mechanism
If the `GEMINI_API_KEY` is absent, network fails, or the guardrail catches a hallucination, the system deterministically defaults to the `human_readable_explanation` created by the M8 SHAP explainer.

## 6. API Configuration
- Set `GEMINI_API_KEY` in `.env` or system environment variables.
- Required library: `google-generativeai` (or standard HTTP client).
- The toggle "Explain with AI" appears in the Product UI and defaults to OFF for latency preservation.
