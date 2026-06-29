# EMA smoothing per alpha3 model — yield vs control-effort

Post-hoc causal EMA filter on each model's action stream (no retraining), aggregated over the 9-cell grid × seed. `alpha=1.0` is the unsmoothed baseline.

MPC Hp8 (perfect) reference: 100% yield at mean|Δu| = 0.97 mm/day.


## Recommended smoothing per model

| model | best α | τ (days) | yield kg/ha | %MPC | mean|Δu| | Δu reduction | drought d | waterlog d | rule |
|---|---|---|---|---|---|---|---|---|---|---|
| a3_0p50 | 0.50 | 1.4 | 3836 | 100.7 | 0.924 | 55% | 21.7 | 5.9 | smoothest alpha with yield>=3815 kg/ha (<= 0.5% below baseline) and waterlog<=6.8 d |
| a3_1p00 | 0.30 | 2.8 | 3752 | 98.4 | 0.605 | 69% | 25.3 | 17.2 | smoothest alpha with yield>=3727 kg/ha (<= 0.5% below baseline) and waterlog<=17.6 d |
| a3_1p15 | 0.30 | 2.8 | 3824 | 100.3 | 0.684 | 69% | 18.8 | 15.5 | smoothest alpha with yield>=3809 kg/ha (<= 0.5% below baseline) and waterlog<=15.9 d |
| a3_1p50 | 0.50 | 1.4 | 3801 | 99.8 | 1.156 | 58% | 23.5 | 13.8 | smoothest alpha with yield>=3784 kg/ha (<= 0.5% below baseline) and waterlog<=14.3 d |

## a3_0p50 — full frontier

| alpha | n | yield kg/ha | %MPC | mean|Δu| | drought d | waterlog d | water mm | Pareto |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 9 | 3834 | 100.6 | 2.044 | 21.9 | 5.8 | 367 |  |
| 0.70 | 9 | 3836 | 100.7 | 1.298 | 21.6 | 5.8 | 367 |  |
| 0.50 | 9 | 3836 | 100.7 | 0.924 | 21.7 | 5.9 | 367 | YES |
| 0.30 | 9 | 3832 | 100.6 | 0.594 | 23.4 | 6.9 | 368 | YES |
| 0.20 | 9 | 3822 | 100.3 | 0.441 | 26.3 | 8.9 | 368 | YES |
| 0.10 | 9 | 3773 | 99.0 | 0.296 | 33.7 | 12.8 | 368 | YES |

## a3_1p00 — full frontier

| alpha | n | yield kg/ha | %MPC | mean|Δu| | drought d | waterlog d | water mm | Pareto |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 9 | 3746 | 98.3 | 1.930 | 25.7 | 16.6 | 373 |  |
| 0.70 | 9 | 3749 | 98.4 | 1.289 | 25.4 | 16.4 | 373 |  |
| 0.50 | 9 | 3751 | 98.4 | 0.934 | 25.1 | 16.5 | 374 |  |
| 0.30 | 9 | 3752 | 98.4 | 0.605 | 25.3 | 17.2 | 375 | YES |
| 0.20 | 9 | 3748 | 98.4 | 0.439 | 26.3 | 18.3 | 376 | YES |
| 0.10 | 9 | 3725 | 97.8 | 0.277 | 31.6 | 20.1 | 374 | YES |

## a3_1p15 — full frontier

| alpha | n | yield kg/ha | %MPC | mean|Δu| | drought d | waterlog d | water mm | Pareto |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 9 | 3828 | 100.5 | 2.242 | 19.4 | 14.9 | 375 |  |
| 0.70 | 9 | 3828 | 100.5 | 1.471 | 18.7 | 15.1 | 376 | YES |
| 0.50 | 9 | 3827 | 100.4 | 1.057 | 18.4 | 15.2 | 377 | YES |
| 0.30 | 9 | 3824 | 100.3 | 0.684 | 18.8 | 15.5 | 378 | YES |
| 0.20 | 9 | 3814 | 100.1 | 0.497 | 20.3 | 16.1 | 378 | YES |
| 0.10 | 9 | 3775 | 99.1 | 0.297 | 24.5 | 18.4 | 376 | YES |

## a3_1p50 — full frontier

| alpha | n | yield kg/ha | %MPC | mean|Δu| | drought d | waterlog d | water mm | Pareto |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 9 | 3803 | 99.8 | 2.722 | 23.6 | 13.3 | 378 | YES |
| 0.70 | 9 | 3802 | 99.8 | 1.653 | 23.5 | 13.6 | 378 | YES |
| 0.50 | 9 | 3801 | 99.8 | 1.156 | 23.5 | 13.8 | 378 | YES |
| 0.30 | 9 | 3796 | 99.6 | 0.733 | 23.3 | 14.4 | 379 | YES |
| 0.20 | 9 | 3786 | 99.3 | 0.545 | 23.0 | 16.0 | 379 | YES |
| 0.10 | 9 | 3756 | 98.6 | 0.362 | 27.1 | 19.9 | 379 | YES |
