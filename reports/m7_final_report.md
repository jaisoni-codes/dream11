# M7 Final Validation Report — Canonical Model

> **CORRECTION NOTICE**: The M6 Recall@11 of 0.439 was computed with a bug where `P(play)` model called `.predict()` returning hard 0/1 labels instead of `.predict_proba()[:, 1]` returning calibrated probabilities. All metrics below reflect the corrected pipeline. The M6 result is SUPERSEDED.

## Canonical Model
- Training window: `2015-01-01` to `2024-06-30`
- FP model: 62,737 rows, 177 features
- Play model: 376,222 rows, 17 features
- FP bundle: `models/model_bundle.pkl`
- Play bundle: `models/play_probability_bundle.pkl`

## Holdout Metrics (2024-07-01+)

- **Training Window**: 2015-01-01 to 2024-06-30
- **FP Training Rows**: 62737
- **FP Features**: 177
- **Play Training Rows**: 376222
- **Play Features**: 17
- **Holdout Matches**: 244
- **Mean Recall@11**: 0.7168
- **Mean Precision@11**: 0.7168
- **Mean XI Overlap**: 7.8852
- **Mean Jaccard**: 0.3442
- **Mean Predicted Team Pts**: 348.6066
- **Mean Dream XI Pts**: 644.9139
- **Mean Team Regret**: 296.3074
- **Captain Acc (in-play)**: 0.7910
- **VC Acc (in-play)**: 0.7910
- **Latency Pool mean (s)**: 218.7197
- **Latency Pool p90 (s)**: 1.8066
- **Latency FP Inf mean (s)**: 0.0251
- **Latency Play Inf mean (s)**: 0.0234
- **Latency ILP mean (s)**: 0.1990
- **Latency Total mean (s)**: 218.9671
- **Latency Total p90 (s)**: 2.0429
- **Reproducibility (100 matches)**: PASS
- **predict_proba correction**: Applied — old M6 Recall@11=0.439 is SUPERSEDED
- **Leakage Status**: CLEAN — no target-match data in inference path

## Leakage Status
- Candidate pool: 365-day historical reconstruction, strictly < target date OK
- FP features: all rolling/EWMA/career stats computed before target date OK
- Play features: include `days_since_last_match` from target date, no scorecards OK
- Target FP never in feature matrix OK
- Reproducibility: identical pool/proba/E[FP]/XI/C/VC across 100 repeated runs OK
