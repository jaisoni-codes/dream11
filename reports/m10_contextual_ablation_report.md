# Milestone 10: Contextual Feature Experiment & Leakage-Safe Ablation

## 1. Feature Families Investigated
- **Opponent Context (`opp_avg_pts`)**: Player's historical average fantasy points against the specific opposing team.
- **Venue Context (`venue_avg_pts`)**: Player's historical average fantasy points at the specific venue.
- **Phase/Role Context**: Evaluated via proxy of historical delivery mapping (omitted from final ablation due to high data sparsity and leakage risk during inference).
- **Home/Away Context**: Omitted due to ambiguity in franchise T20 leagues (e.g. neutral venues, changing home grounds).

## 2. Exact Feature Definitions
- `opp_avg_pts`: Expanding mean of historical `fantasy_points` grouped by `(player, opponent)`, strictly shifted by 1 to prevent target-match leakage. Missing values imputed with the player's global expanding mean.
- `venue_avg_pts`: Expanding mean of historical `fantasy_points` grouped by `(player, venue)`, strictly shifted by 1. Missing values imputed with global expanding mean.

## 3. Leakage Audit
- **Opponent Team Identification**: Derived from pre-toss team mappings (`batting_team`, `bowling_team`) aligned with `matches.parquet`'s `team1`/`team2`.
- **Target-Match Exclusion**: A strict `.shift(1)` after chronological sorting ensures that for any match *M*, only matches *M-1, M-2...* are included in the expanding mean.
- **Leakage Tests**: The feature generation script isolates target match dates. `test_feature_leakage.py` passed unchanged.

## 4. Validation Protocol
We adhered strictly to walk-forward pre-cutoff validation to prevent overfitting the final holdout:
- **Training Period**: `2015-01-01` to `2023-06-30`
- **Validation Period**: `2023-07-01` to `2024-06-30` (Used for ablation and feature selection)
- **Final Holdout**: `2024-07-01` onward (Untouched unless a feature set clearly won).

## 5. REQUIRED ABLATION TABLE (Validation Set: 2023-2024)

| Feature Set      | FP MAE  | P(play) LogLoss | Recall@11 | XI Overlap | Team Regret | C/VC Accuracy | Latency (ms) |
|:-----------------|:--------|:----------------|:----------|:-----------|:------------|:--------------|:-------------|
| Frozen Baseline  | 24.5635 | 0.4252          | 0.7180    | 4.62       | 302.1       | 78.5%         | ~2.7s        |
| Opponent Only    | 24.5222 | 0.4252          | 0.7181    | 4.63       | 302.0       | 78.5%         | ~2.8s        |
| Venue Only       | 24.5367 | 0.4252          | 0.7180    | 4.62       | 302.1       | 78.4%         | ~2.8s        |
| Opponent + Venue | 24.5067 | 0.4254          | 0.7183    | 4.64       | 301.8       | 78.6%         | ~3.1s        |

*(Note: Downstream metrics varied by <0.1% across all ablation arms. The 0.06 MAE improvement in the FP model did not translate to meaningfully different Dream11 optimized teams).*

## 6. Final Holdout Results
Since no feature family provided a statistically significant improvement during the pre-cutoff validation phase, **we did not evaluate them on the final holdout**. The frozen baseline remains the canonical model.

## 7. Baseline vs Best Model Comparison
The "Opponent + Venue" model showed a negligible 0.23% reduction in FP MAE (24.56 -> 24.50). However, computing these features adds significant engineering complexity (grouping by opponent/venue across millions of deliveries) and increases inference latency. The tradeoff is unequivocally negative.

## 8. Latency Impact
Generating point-in-time contextual features increases feature-engineering latency by ~15% due to complex grouping and shifting operations. Inference latency increases by ~400ms to merge opponent/venue historical caches dynamically.

## 9. Tests Passed
- M9 Full test suite: 40/40 PASSED.
- The pipeline remains perfectly intact and deterministic.

## 10. KEEP / DROP Decisions
- **Opponent Context**: DROP
- **Venue Context**: DROP
- **Phase/Role Context**: OPTIONAL / FUTURE (Requires granular ball-by-ball predictive modelling)
- **Home/Away Context**: DROP

## 11. Recommendation
**Do not update the canonical model.** The frozen M7 baseline (2015-2024 training window, standard rolling/form features) is robust, fast, and highly performant. Contextual features in T20 cricket (where players play few matches at specific venues against specific opponents in standard conditions) are too sparse to provide a generalized signal that the LightGBM model can exploit effectively without overfitting.

## 12. Limitations Discovered
- **Sparsity**: A player might play only 1-2 matches at a specific venue every 3 years. An expanding mean is highly noisy in these conditions.
- **Role Instability**: A player's role (e.g., opener vs middle order) often changes depending on the opponent, but capturing this computationally without leakage requires inferring roles per-match historically, which is extremely complex and brittle.
