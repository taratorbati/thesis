# EMA smoothing — yield vs control-effort Pareto sweep

TD3 v2.21c checkpoints, re-evaluated under a causal EMA filter on the policy action (no retraining). Aggregated over the 9-cell grid × seeds. `alpha=1.0` is the unsmoothed baseline.

MPC Hp8 (perfect) reference: 100% yield at mean|Δu| = 0.97 mm/day.


## Pool A

| alpha | n | yield kg/ha | %MPC | mean|du| | drought d | waterlog d | water mm | Pareto |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 27 | 3805 | 99.9 | 2.545 | 33.7 | 7.2 | 362 |  |
| 0.70 | 27 | 3808 | 100.0 | 1.625 | 33.5 | 7.0 | 362 |  |
| 0.50 | 27 | 3810 | 100.0 | 1.149 | 33.3 | 6.9 | 362 |  |
| 0.30 | 27 | 3811 | 100.0 | 0.722 | 33.2 | 7.5 | 363 | YES |
| 0.20 | 27 | 3806 | 99.9 | 0.522 | 33.9 | 8.9 | 364 | YES |
| 0.10 | 27 | 3765 | 98.8 | 0.319 | 38.9 | 12.7 | 363 | YES |

## Pool B

| alpha | n | yield kg/ha | %MPC | mean|du| | drought d | waterlog d | water mm | Pareto |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 27 | 3800 | 99.7 | 2.496 | 35.2 | 6.5 | 358 |  |
| 0.70 | 27 | 3804 | 99.8 | 1.632 | 34.9 | 6.3 | 358 |  |
| 0.50 | 27 | 3807 | 99.9 | 1.146 | 35.0 | 6.1 | 358 |  |
| 0.30 | 27 | 3810 | 100.0 | 0.716 | 35.3 | 6.2 | 359 | YES |
| 0.20 | 27 | 3805 | 99.9 | 0.520 | 36.3 | 7.3 | 361 | YES |
| 0.10 | 27 | 3768 | 98.9 | 0.321 | 39.0 | 11.2 | 362 | YES |

## Mixed (legacy)

| alpha | n | yield kg/ha | %MPC | mean|du| | drought d | waterlog d | water mm | Pareto |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 27 | 3812 | 100.0 | 2.464 | 33.0 | 7.6 | 360 |  |
| 0.70 | 27 | 3814 | 100.1 | 1.611 | 32.8 | 7.4 | 360 |  |
| 0.50 | 27 | 3816 | 100.2 | 1.137 | 32.7 | 7.2 | 360 | YES |
| 0.30 | 27 | 3816 | 100.2 | 0.718 | 33.2 | 7.6 | 360 | YES |
| 0.20 | 27 | 3808 | 99.9 | 0.524 | 34.5 | 8.7 | 361 | YES |
| 0.10 | 27 | 3764 | 98.8 | 0.325 | 37.6 | 12.0 | 361 | YES |
