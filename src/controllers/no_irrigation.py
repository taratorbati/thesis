# =============================================================================
# src/controllers/no_irrigation.py
# The simplest possible controller: never irrigates.
# Establishes the rainfed lower bound for biomass production.
# =============================================================================

import numpy as np

from src.controllers.base import Controller


class NoIrrigationController(Controller):
    """Returns zero irrigation every day. Lower-bound baseline.

    This controller exists primarily to characterize the rainfed yield
    of each scenario — every other controller must do at least this well,
    and the gap between rainfed and irrigated yield indicates how much
    the irrigation budget is worth.
    """

    def __init__(self):
        super().__init__(name='no_irrigation')
        self._N = None

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        self._N = terrain['N']

    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        return np.zeros(self._N)
