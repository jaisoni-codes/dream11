# Milestone 8: SHAP Explainability Layer

## Overview
A SHAP-based explainability layer has been integrated into the end-to-end inference path `predict(explain=True)`.
It evaluates feature contributions for BOTH the FP model and the P(play) model simultaneously, without altering predictions or the ILP optimization.

## Global Feature Importance
### Top FP Model Factors
| feature                    |   mean_abs_shap |
|:---------------------------|----------------:|
| balls_bowled_ewma_short    |        3.49513  |
| fantasy_points_ewma_long   |        1.50621  |
| fantasy_points_career_mean |        1.34159  |
| fantasy_points_ewma_medium |        0.6511   |
| balls_bowled_last_1        |        0.631261 |
| balls_bowled_last_3        |        0.597662 |
| econ_last_1                |        0.418178 |
| fantasy_points_career_sum  |        0.408503 |
| bowl_pts_last_1            |        0.335659 |
| runs_career_sum            |        0.299607 |
| balls_bowled_career_sum    |        0.277521 |
| bat_pts_career_sum         |        0.242353 |
| fantasy_points_ewma_short  |        0.221324 |
| fours_career_sum           |        0.221194 |
| runs_ewma_long             |        0.202356 |
| econ_ewma_short            |        0.183606 |
| fours_ewma_short           |        0.172289 |
| balls_bowled_ewma_medium   |        0.155719 |
| sixes_last_20              |        0.152297 |
| fantasy_points_last_20     |        0.151449 |

### Top P(play) Model Factors
| feature                    |   mean_abs_shap |
|:---------------------------|----------------:|
| days_since_last_match      |       2.25768   |
| matches_played_before      |       0.11292   |
| fantasy_points_career_sum  |       0.0824773 |
| fantasy_points_ewma_short  |       0.0790943 |
| batting_opportunity_rate   |       0.0730714 |
| fantasy_points_ewma_long   |       0.0311727 |
| bowling_opportunity_rate   |       0.0281853 |
| runs_career_mean           |       0.0247839 |
| fantasy_points_last_1      |       0.0229102 |
| fantasy_points_career_mean |       0.0226556 |
| wickets_career_mean        |       0.0216309 |
| role_confidence            |       0.0127922 |
| role_cat                   |       0.0117278 |
| batting_strike_rate_career |       0         |
| bowling_strike_rate_career |       0         |
| economy_rate_career        |       0         |
| recent_fantasy_points_var  |       0         |

## Example Local Explanation Output (JSON)
```json
{
  "player": "A Alfonso",
  "team": "Malawi",
  "role": "BAT",
  "p_play": 0.08137529705651524,
  "predicted_fp": 12.210201691802164,
  "expected_fp": 0.9936087897903659,
  "top_positive_fp_factors": [
    {
      "feature": "econ_last_1",
      "value": 0.0,
      "contribution": 0.4415748030896376
    },
    {
      "feature": "fantasy_points_std_last_10",
      "value": 6.082762530298219,
      "contribution": 0.23801978277904604
    },
    {
      "feature": "matches_played_before",
      "value": 3.0,
      "contribution": 0.1995058124430206
    }
  ],
  "top_negative_fp_factors": [
    {
      "feature": "balls_bowled_ewma_short",
      "value": 0.0,
      "contribution": -3.814290411863809
    },
    {
      "feature": "fantasy_points_ewma_long",
      "value": 9.361365528726063,
      "contribution": -2.746766642539941
    },
    {
      "feature": "fantasy_points_career_mean",
      "value": 9.0,
      "contribution": -2.2548688462416995
    }
  ],
  "top_positive_play_factors": [
    {
      "feature": "days_since_last_match",
      "value": 26.0,
      "contribution": 1.2531611147398547
    },
    {
      "feature": "matches_played_before",
      "value": 3.0,
      "contribution": 0.05173295093925207
    },
    {
      "feature": "batting_opportunity_rate",
      "value": 9.666666666666666,
      "contribution": 0.03145679789583412
    }
  ],
  "top_negative_play_factors": [
    {
      "feature": "fantasy_points_ewma_short",
      "value": 10.368421052631579,
      "contribution": -0.1690249394037169
    },
    {
      "feature": "bowling_opportunity_rate",
      "value": 0.0,
      "contribution": -0.041863853245698475
    },
    {
      "feature": "fantasy_points_career_sum",
      "value": 27.0,
      "contribution": -0.03835005274269692
    }
  ],
  "human_readable_explanation": "A Alfonso has a predicted FP of 12.2 because of strong positive indicators like econ_last_1 (contrib: 0.4) and offsets from balls_bowled_ewma_short (contrib: -3.8). The play model assigns a 0.08 probability of playing, mainly due to days_since_last_match and negative pressure from fantasy_points_ewma_short. Final Expected FP: 1.0."
}
```

## Human Readable Output
> A Alfonso has a predicted FP of 12.2 because of strong positive indicators like econ_last_1 (contrib: 0.4) and offsets from balls_bowled_ewma_short (contrib: -3.8). The play model assigns a 0.08 probability of playing, mainly due to days_since_last_match and negative pressure from fantasy_points_ewma_short. Final Expected FP: 1.0.

## Latency
- Additional latency per match (explain=True): 231.4 ms

## Validation & Requirements
- Explainer integrated downstream of M7 models.
- Exact mathematical relationship `E[FP] = P(play) * E[FP|play]` is preserved in explanations.
- Fully deterministic outputs.
- Zero data leakage (SHAP relies strictly on pre-toss features).
- Core M7 metrics unchanged (explanations do not modify predictions).
