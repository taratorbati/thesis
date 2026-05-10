# =============================================================================
# src/controllers/base.py
# Abstract base class for all irrigation controllers.
# =============================================================================

from abc import ABC, abstractmethod


class Controller(ABC):
    """Abstract base class for all irrigation controllers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        """Initialise the controller for a new season.

        Parameters
        ----------
        terrain : dict
            From src.terrain.load_terrain.
        crop : dict
            From soil_data.get_crop.
        season_days : int
        budget_total : float
            Total seasonal water budget in mm (field-averaged).
        scenario_name : str, optional
            Evaluation scenario: 'dry' (2022), 'moderate' (2018), or
            'wet' (2024). Used by controllers that load scenario-specific
            precomputed data.
        """

    @abstractmethod
    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        """Return irrigation action for the current day.

        Parameters
        ----------
        day : int
            Current day index (0-based).
        state : dict
            Current ABM state: {'x1', 'x2', 'x3', 'x4', 'x5'}.
        climate_today : dict
        budget_remaining : float
        forecast : any, optional

        Returns
        -------
        u : np.ndarray of shape (N,)
            Irrigation depths in mm/day.
        """

    def set_climate(self, climate):
        """Accept the full-season climate dict (optional hook).

        Called by the runner before the first step() for controllers
        that need access to the full climate array (MPC forecast provider,
        SAC runner). Default is a no-op; override as needed.
        """
