# src/rl/run_seeds.py  v2.20.0  (Stage 4: seed reproduction = the acceptance test)
# -----------------------------------------------------------------------------
# Run ONE frozen v2.20 config across N seeds and record a resumable campaign
# manifest.  Place in src/rl/ alongside train_v220_td3.py.
#
# WHY (the real acceptance test, not a victory lap)
# -------------------------------------------------
# A single-seed "beats MPC" is an artifact -- the project's "Mistake 5" was
# exactly that (v2.7 "beat MPC in 4 cells" vanished on seed 1).  The seed sweep
# is what turns a candidate config into a defensible result: judge the MEAN
# across seeds against the Stage-0 scorecard, and confirm every seed converges
# (final ~= best, q_pred bounded, guard never trips) -- which is also the only
# real test of whether the Stage-1 stabilisation actually fixed the
# multistability (same config, different draw order -> same outcome?).  N=5 is
# the field standard; budget usually affords 3.
#
# DESIGN
# ------
#   * Freeze the config first (Stages 1-3 pick it); pass only --config + --seeds.
#   * Resumable: a completed seed is skipped on a re-run, so a crashed campaign
#     restarts where it stopped (within-run crash recovery is the trainer's own
#     25k checkpoints; this driver resumes at SEED granularity).
#   * The campaign JSON is rewritten (utf-8) after EVERY seed, so nothing is lost
#     and nothing needs rerunning to know the status.
#   * One seed failing does not abort the campaign; its status is recorded and
#     the next seed proceeds.
#
# RUNS NOTHING ON IMPORT.  Launching this DOES start full training runs
# (~1 compute unit / 250k steps each).  Invoke from the CLI (see __main__).
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.rl.configs_v220 import CONFIGS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_seeds(
    config_name: str = "A",
    seeds: Optional[List[int]] = None,
    output_dir: str = "results/rl",
    total_timesteps: Optional[int] = None,
):
    """Train ``config_name`` for each seed in ``seeds``; write a resumable campaign.

    Returns the campaign dict ({seed -> {status, run_name, ...}}).
    """
    # Imported here (not at module top) so merely importing this file is cheap
    # and side-effect-free; the heavy SB3/torch import happens only on launch.
    from src.rl.train_v220_td3 import train_td3_v220

    if config_name not in CONFIGS:
        raise KeyError(f"unknown config {config_name!r}; choices: {sorted(CONFIGS)}")
    if seeds is None:
        seeds = [0, 1, 2]

    label = CONFIGS[config_name]["label"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    campaign_path = out / f"seed_campaign_{config_name}_{label}.json"

    # Resume: load any prior campaign so completed seeds are skipped.
    campaign: dict = {}
    if campaign_path.exists():
        try:
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        except Exception:
            campaign = {}
    campaign.setdefault("config_name", config_name)
    campaign.setdefault("label", label)
    campaign.setdefault("seeds_requested", list(seeds))
    campaign.setdefault("runs", {})

    def _flush():
        campaign["updated_utc"] = _now()
        campaign_path.write_text(json.dumps(campaign, indent=2), encoding="utf-8")

    _flush()

    for i, seed in enumerate(seeds):
        key = str(seed)
        prior = campaign["runs"].get(key, {})
        if prior.get("status") == "completed":
            print(f"[run_seeds] seed {seed} already completed -- skipping.")
            continue

        run_name = f"td3_v220_{label}_seed{seed}"
        print(f"\n{'#'*72}\n# seed {seed}  ({i + 1}/{len(seeds)})  config {config_name} ({label})\n{'#'*72}")
        campaign["runs"][key] = {"run_name": run_name, "status": "running", "started_utc": _now()}
        _flush()

        try:
            train_td3_v220(
                config_name=config_name,
                seed=seed,
                output_dir=output_dir,
                total_timesteps=total_timesteps,
            )
            campaign["runs"][key].update(status="completed", finished_utc=_now())
        except BaseException as e:  # keep the campaign alive; record and continue
            campaign["runs"][key].update(status="failed", error=repr(e), finished_utc=_now())
            print(f"[run_seeds] seed {seed} FAILED: {e!r} -- continuing to next seed.")
        finally:
            _flush()

    done = sum(1 for v in campaign["runs"].values() if v.get("status") == "completed")
    print(f"\n[run_seeds] campaign complete: {done}/{len(seeds)} seeds finished. "
          f"Manifest: {campaign_path}")
    return campaign


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Stage-4 seed reproduction: train one frozen v2.20 config across N "
            "seeds and write a resumable campaign manifest. Judge the MEAN vs the "
            "Stage-0 scorecard; confirm every seed converges."
        )
    )
    parser.add_argument("--config",          type=str, default="A", choices=sorted(CONFIGS))
    parser.add_argument("--seeds",           type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir",      type=str, default="results/rl")
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()

    run_seeds(
        config_name=args.config,
        seeds=args.seeds,
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
    )
