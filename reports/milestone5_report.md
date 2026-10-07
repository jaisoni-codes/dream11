### Milestone 5 Status

1. Evaluation date range: 2024-07-01 to Latest
2. Number of evaluation matches: 2282
3. Number evaluated: 192
4. Number skipped + reasons: 2090 (Reasons: {'Not exactly 2 teams found in deliveries', 'Skipped due to simulated latency constraints (sampled 250)', 'ILP Pred Failed: Infeasible: No players available for role WK.', 'ILP Actual Failed: Infeasible: No players available for role AR.', 'ILP Actual Failed: Infeasible: No players available for role WK.', 'ILP Actual Failed: Infeasible: Solver could not find an optimal solution. Status: Infeasible'})
5. Candidate pool method: Historically Reconstructed pre-toss squad (365 days)
6. Frozen model and cutoff: Milestone 3 Bundle (Cutoff: 2024-06-30)
7. Mean overlap: 2.59
8. Mean Recall@11: 0.236
9. Mean Jaccard: 0.140
10. Mean team regret: 464.69
11. Captain accuracy: 0.057
12. VC accuracy: 0.021
13. Individual-player MAE: N/A (Focus is Team Regret here)
14. Prediction latency: 0.0272 s/match
15. Optimization latency: 0.4190 s/match
16. End-to-end latency: 3.8625 s/match
17. Leakage tests: Passed (automated assertions in Pytest)
18. Determinism test: Passed
19. Pytest result: 100% Pass
20. Known limitations: Predicted players who do not ultimately play in the actual XI receive 0 actual points, heavily driving Team Regret. We explicitly lack a P(play) playing-XI probability model, so pre-toss selection naturally suffers from "did not play" errors.

