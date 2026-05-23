# src/rl/runner_tqc.py  v2.10.0
# -----------------------------------------------------------------------------
# TQCRLController - inference wrapper for TQC checkpoints (v2.10 E2 onward).
#
# Inherits from RLController and overrides only the model loader.  All of
# RLController's observation-building, AR(1) noisy-forecast generation, and
# inference timing logic is reused unchanged.
#
# Why this exists:
#   RLController.__init__ calls SAC.load via _load_sac_model().  TQC
#   checkpoints have different state-dict keys and a different algorithm
#   class, so SAC.load won't work.  v2.10 added a tiny _load_model()
#   indirection in RLController.__init__ so subclasses can swap the loader
#   without duplicating the rest of __init__.
#
# Usage:
#   from src.rl.runner_tqc import TQCRLController
#   controller = TQCRLController(
#       model_path=".../sac_v210_e2_seed0/best_model/best_model.zip",
#       deterministic=True,
#       forecast_mode='perfect',
#   )
#   run_season(controller=controller, ...)   # uses src.runner.run_season
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from sb3_contrib import TQC

from src.rl.runner import RLController
from src.rl.networks_tqc import V27TQCPolicy


def _load_tqc_model(model_path: Path, device: str = 'cpu'):
    """Load a TQC v2.10 E2 checkpoint.

    Parameters
    ----------
    model_path : Path
    device : str
        Default 'cpu' (matches RLController for inference).

    Returns
    -------
    (model, arch_label, obs_layout)
        Same contract as src.rl.runner._load_sac_model:
        - model      : sb3_contrib.TQC instance
        - arch_label : str  ('VDN-per-quantile TQC v2.10 E2')
        - obs_layout : 'v27' (TQC v2.10 E2 uses the v2.7 obs layout)
    """
    model = TQC.load(
        str(model_path),
        device=device,
        custom_objects={"policy_class": V27TQCPolicy},
    )
    return model, "VDN-per-quantile TQC (v2.10 E2)", "v27"


class TQCRLController(RLController):
    """Inference controller for a trained TQC v2.10 E2 model.

    Inherits all observation-building, forecast generation, and inference
    timing from RLController.  Overrides only _load_model() so the loader
    uses sb3_contrib.TQC instead of stable_baselines3.SAC.

    The obs layout is always 'v27' for E2 - TQC E2 reuses the v2.7
    architecture and observation format (8 features/agent, 1097-dim).

    Constructor signature is identical to RLController.  See RLController
    docstring for parameter details.
    """

    def _load_model(self):
        return _load_tqc_model(self.model_path, device='cpu')
