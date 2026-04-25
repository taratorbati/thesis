# Optimal Irrigation Control in Cyber-Physical Agro Systems

**MSc thesis, ITMO University, R4237c**
**Student:** Tara Torbati
**Supervisor:** Peregudin A. A.

This repository contains the code, data, and writeup for an MSc thesis investigating whether a Soft Actor-Critic reinforcement learning agent can match the performance of a Model Predictive Controller on constrained irrigation control for a topographically heterogeneous rice field in Gilan Province, Iran.

## Quick start

The complete design and implementation plan is in **[`ARCHITECTURE.md`](ARCHITECTURE.md)**. Read that first.

For a working end-to-end run with no irrigation, on the dry 2022 climate scenario:

```bash
pip install -r requirements.txt
python preprocess.py                                    # one-time: clean climate data
python cross_validate_gwetroot.py                       # validate ABM against satellite
python run_comparison.py                                # demonstrate cascade routing
```

## Repository layout

| Path | Contents |
|---|---|
| `abm.py`, `soil_data.py`, `climate_data.py` | Core domain code: agent-based crop-soil model, crop parameters, climate loader |
| `preprocess.py` | NASA POWER → cleaned CSV with Penman-Monteith ET₀ |
| `plot_*.py`, `run_plots.py` | 25-year climate visualizations |
| `validate_physics_fao.py` | FAO-56 unit test on flat 10-agent topology |
| `cross_validate_gwetroot.py` | ABM cross-validation against NASA GWETROOT (cascade mode) |
| `run_comparison.py`, `run_comparison_2.py` | Runoff-mode (none / simple / cascade) comparison studies |
| `src/` | Shared infrastructure: terrain graph, parquet I/O, runner (in progress) |
| `controllers/` | Controller implementations (no-irrigation, fixed-schedule, MPC, RL) — in progress |
| `gilan_farm.tif` | Digital elevation model, 10 × 13, elevation 74–181 m |
| `POWER_..._LST.csv` | NASA POWER 2000–2026 raw climate data |
| `notes/` | Baseline paper, agronomic notes, thesis draft |
| `reports/` | Finished sub-deliverables (e.g. field dynamics report) |
| `history/` | Superseded code retained for provenance |
| `results/` | Generated outputs (gitignored) |

## Status

- ✅ ABM with cascade water routing, surface ponding state, drought-stress biomass coupling
- ✅ NASA GWETROOT cross-validation (Pearson r = 0.74 dry year)
- ✅ FAO-56 physics unit test
- ✅ 25-year Penman-Monteith ET₀ climatology
- ✅ Step A foundation: `src/terrain.py`, `src/persistence.py`
- ⏳ Step B: controller interface, runner, no-irrigation + fixed-schedule baselines
- ⏳ Step C: precomputation cache (thermal time, forecasts)
- ⏳ Step D: MPC (CasADi + IPOPT)
- ⏳ Step E: RL (Stable-Baselines3 SAC, Kaggle GPU training)
- ⏳ Step F: aggregation, statistical tests, thesis figures

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full plan.
