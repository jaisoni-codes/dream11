# Final System Architecture (Inter IIT Demo)

## Data & Feature Pipeline (Strictly Pre-Toss)

```
Cricsheet Raw Data
       ↓
Ingestion & Cleaning
       ↓
Normalization & Role Mapping
       ↓
Leakage-safe Features (Expanding/Rolling means, shift(1))
       ↓
Candidate Pool Construction (Strictly historical)
```

## Inference & Optimization Pipeline (Deterministic)

```
Leakage-safe Features
       ↓
FP LightGBM Model + P(play) LightGBM Model
       ↓
Expected FP = P(play) × E[FP | play]
       ↓
ILP Optimizer (Constraints: 1-4 WK, 3-6 BAT, 1-4 AR, 3-6 BOWL, max 7 per team)
       ↓
Best XI + Captain (2x) + Vice-Captain (1.5x)
```

## Explanation & UI Presentation Layer

```
Best XI + C/VC
       ↓
SHAP (TreeExplainer)
       ↓
Structured JSON (Top positive/negative factors)
       ↓
Grounded GenAI (Language Layer - ONLY translates JSON)
       ↓
Validation Guardrails (Checks for hallucinated pitch, weather, numbers)
       ↓
UI (Streamlit)
```

---

### Architectural Boundaries
* **Prediction Engine**: ML models (LightGBM). GenAI **cannot** modify predictions.
* **Decision Engine**: ILP optimizer (PuLP). GenAI **cannot** select players or C/VC.
* **Explanation Engine**: SHAP.
* **Language Layer**: GenAI (Gemini 1.5 Flash). Only translates SHAP structures into markdown text.

### M10 Status
* Opponent and Venue contextual features were thoroughly tested during Milestone 10 via walk-forward evaluation.
* They were **DROPPED** from this final canonical architecture because they added significant grouping/join latency without providing meaningful downstream improvements in Recall@11. 
