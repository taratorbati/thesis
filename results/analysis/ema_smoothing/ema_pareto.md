# EMA smoothing — yield vs control-effort Pareto sweep

TD3 v2.21c checkpoints, re-evaluated under a causal EMA filter on the policy action (no retraining). Aggregated over the 9-cell grid x seeds. `alpha=1.0` is the unsmoothed baseline.

MPC Hp8 (perfect) reference: 100% yield at mean|Delta u| = 0.97 mm/day.

| alpha | n | yield kg/ha | %MPC | mean|du| | drought d | waterlog d | water mm | Pareto |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 27 | 3812 | 100.0 | 2.464 | 33.0 | 7.6 | 360 |  |
| 0.70 | 27 | 3814 | 100.1 | 1.611 | 32.8 | 7.4 | 360 |  |
| 0.50 | 27 | 3816 | 100.2 | 1.137 | 32.7 | 7.2 | 360 | YES |
| 0.30 | 27 | 3816 | 100.2 | 0.718 | 33.2 | 7.6 | 360 | YES |
| 0.20 | 27 | 3808 | 99.9 | 0.524 | 34.5 | 8.7 | 361 | YES |
| 0.10 | 27 | 3764 | 98.8 | 0.325 | 37.6 | 12.0 | 361 | YES |
