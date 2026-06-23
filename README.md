# Optimal Irrigation Control in Cyber-Physical Agro Systems

**MSc thesis, ITMO University, R4237c**
**Student:** Tara Torbati
**Supervisor:** Peregudin A. A.

Code, data, and writeup for an MSc thesis investigating whether reinforcement
learning agents (SAC and TD3) can match a Model Predictive Controller on
constrained irrigation of a topographically heterogeneous rice field in Gilan
Province, Iran. The plant is a 130-cell crop-soil agent-based model (ABM) with
cascade water routing and surface-ponding dynamics, cross-validated against NASA
satellite soil moisture (Pearson r = 0.74 on the dry year).

See **[`ARCHITECTURE.md`](ARCHITECTURE.md)** for the full system design.

## Quick start

```bash
pip install -r requirements.txt          # core stack; RL extras: torch, stable-baselines3, gymnasium
python scripts/preprocess/preprocess.py   # NASA POWER -> cleaned CSV with Penman-Monteith ET0
python scripts/validate_physics_fao.py    # FAO-56 physics unit test on a flat topology
```

Run a controller over the 9 evaluation cells (3 held-out years x 3 budgets):

```bash
python -m scripts.experiments.exp_mpc              # MPC (CasADi + IPOPT)
python -m scripts.experiments.exp_fixed_schedule   # fixed-schedule baseline
python -m scripts.experiments.exp_rl --model results/rl/<run>/best_model/best_model.zip
```

Train the RL controllers (GPU recommended; see `notebooks/` for Colab/Kaggle):

```bash
python -m src.rl.train_sac --seed 0       # SAC  (the v2.18-p3b configuration)
python -m src.rl.train_td3 --seed 0       # TD3  (the v2.21c configuration)
```

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Raw inputs: NASA POWER climate CSV and `gilan_farm.tif` DEM (10 x 13, elevation 74-181 m) |
| `src/model/` | Crop-soil ABM, crop/soil parameters, climate loader + year split, terrain graph, forecasts |
| `src/sim/` | `run_season` closed-loop runner, parquet persistence, thermal precomputation |
| `src/controllers/` | No-irrigation, fixed-schedule, reactive-schedule baselines |
| `src/mpc/` | MPC: cost, dynamics, smoothing, solver, controller |
| `src/rl/` | RL: env, networks (SAC + TD3), trainers, eval runner, callbacks |
| `src/analysis/` | Trajectory metrics shared by the analysis scripts |
| `scripts/` | Entry points: `preprocess/`, `experiments/`, `analysis/`, `visualization/`, plus `validate_physics_fao.py` and `run_plots.py` |
| `notebooks/` | Colab/Kaggle training notebooks for the SAC and TD3 controllers |
| `tests/` | Pytest suite (`pytest tests/`) |
| `figures/`, `results/` | Exported thesis figures; generated outputs (models, evaluations) |
| `notes/`, `reports/` | Agronomic notes, baseline paper, thesis drafts and sub-deliverables |
| `history/` | Superseded code retained for provenance |

## The two RL controllers

The project explored many SAC and TD3 variants; the repository keeps the two
chosen configurations:

- **SAC** (`src/rl/train_sac.py`) — parameter-shared LeakyReLU actor + twin VDN
  LayerNorm critic, asymmetric actor LR, weak entropy pin, and a two-phase
  exploration-noise schedule. The standing baseline (~99% of MPC yield).
- **TD3** (`src/rl/train_td3.py`) — deterministic actor reusing the same VDN
  critic, exact n-step returns, a control-rate smoothing reward, and an additive
  terminal-yield bonus (~99.7% of MPC yield; beats MPC on wet-year waterlogging).

Both share the observation, action space, and evaluation protocol in
`src/rl/gym_env.py` and `src/rl/runner.py`.

## Testing

```bash
pytest tests/        # env, observation layout, network architecture, runner equivalence
```
