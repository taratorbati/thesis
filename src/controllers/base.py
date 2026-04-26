# =============================================================================
# src/controllers/base.py
# Abstract Controller interface implemented by every irrigation policy.
#
# The runner (src.runner.run_season) calls controller.step() once per day
# during the receding-horizon simulation. Controllers are stateful between
# step() calls; reset() must be called at the start of each season.
# =============================================================================

from abc import ABC, abstractmethod


class Controller(ABC):
    """Abstract base class for irrigation controllers.

    A Controller takes daily state, climate, and budget information and
    returns an irrigation action (per-agent water depth in mm).

    Conventions
    -----------
    - Action is a 1-D numpy array of shape (N,), values in [0, UB].
    - Action is in mm/day per agent (not a fraction of UB).
    - The runner is responsible for clipping to the box [0, UB] and
      enforcing the global budget after the controller's step.
    - Controllers should not mutate any of their inputs.
    """

    def __init__(self, name=None):
        """Subclasses should call super().__init__(name=...) for logging."""
        self.name = name or self.__class__.__name__

    @abstractmethod
    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        """Initialize the controller at the start of a season.

        Called once before the day-by-day loop begins.

        Parameters
        ----------
        terrain : dict
            Output of src.terrain.load_terrain. Provides N, sends_to, Nr,
            topological_order, gamma_flat, elevation_flat, etc.
        crop : dict
            Crop parameter dict (from soil_data.get_crop(...)).
        season_days : int
            Number of days in the season being simulated.
        budget_total : float
            Total seasonal water budget (mm summed across the season,
            field-averaged). The runner will track budget_remaining
            externally; this value is given for controllers that want
            to plan against the total.
        scenario_name : str, optional
            'dry', 'moderate', 'wet', or None. Some controllers may
            want to log this; most ignore it.
        """
        ...

    @abstractmethod
    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        """Compute the irrigation action for a single day.

        Parameters
        ----------
        day : int
            Zero-indexed day within the season, 0 ≤ day < season_days.
        state : dict
            Current ABM state with keys 'x1', 'x2', 'x3', 'x4', 'x5',
            each a numpy array of shape (N,).
        climate_today : dict
            Current day's weather. Keys: 'rainfall', 'temp_mean',
            'temp_max', 'radiation', 'ET' (and possibly more — see
            climate_data.extract_scenario).
        budget_remaining : float
            Seasonal water budget remaining as of the start of `day`,
            in mm summed across the season, field-averaged.
        forecast : dict or None
            Multi-day forecast for the next H_p days (controllers that
            ignore forecasts receive None). Structure is
            {'rainfall': (H_p,) array, 'ET': (H_p,) array, ...}.

        Returns
        -------
        u : np.ndarray
            Irrigation action, shape (N,), values in mm/day per agent.
        """
        ...
