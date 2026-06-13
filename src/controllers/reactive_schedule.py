# =============================================================================
# src/controllers/reactive_schedule.py
# Dynamic reactive irrigation schedule with tiered rainfall thresholds,
# ponding checks, and active redistribution of saved water.
# =============================================================================

import numpy as np
from src.controllers.fixed_schedule import (
    FixedScheduleController,
    DEFAULT_NUM_EVENTS,
    DEFAULT_EVENT_INTERVAL,
    DEFAULT_FIRST_EVENT_DAY,
    DEFAULT_UB_MM_PER_DAY
)


class DynamicReactiveScheduleController(FixedScheduleController):
    """
    3-Stage Reactive Schedule that scales to the assigned seasonal budget,
    redistributes unspent water dynamically, and uses tiered rainfall overrides.
    """

    def __init__(self,
                 num_events=DEFAULT_NUM_EVENTS,
                 event_interval=DEFAULT_EVENT_INTERVAL,
                 first_event_day=DEFAULT_FIRST_EVENT_DAY,
                 ub_mm_per_day=DEFAULT_UB_MM_PER_DAY,
                 ponding_threshold_mm=0.0):

        super().__init__(
            num_events=num_events,
            event_interval=event_interval,
            first_event_day=first_event_day,
            ub_mm_per_day=ub_mm_per_day
        )
        self.name = 'dynamic_reactive_schedule'
        self.ponding_threshold_mm = float(ponding_threshold_mm)

        # Internal state for dynamic redistribution
        self.season_days = 0
        self.water_saved = 0.0

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        """Initialise the controller and internal ledgers for a new season."""
        # Call parent reset, which triggers our overridden _build_daily_rate
        super().reset(terrain, crop, season_days, budget_total, scenario_name)
        self.season_days = season_days
        self.water_saved = 0.0  # Reset the saved water pool

    def _build_daily_rate(self, season_days, budget_total):
        """
        Overrides the base class method to create the 3-stage step schedule
        (38 days, 30 days, 25 days) scaled precisely to the budget_total.
        """
        rate = np.zeros(season_days, dtype=float)

        # Baseline total for the 100% budget tier
        # (38 days * 5.5) + (30 days * 5.0) + (25 days * 5.0) = 484.0 mm
        baseline_total = 484.0

        # Calculate scaling fraction based on the assigned budget
        # e.g., if budget_total is 411.4 (85%), scale_factor becomes 0.85
        scale_factor = budget_total / baseline_total

        # Apply scaling to the stage rates
        rate_stage1 = 5.5 * scale_factor
        rate_stage2 = 5.0 * scale_factor
        rate_stage3 = 5.0 * scale_factor

        # Define boundaries, safeguarding against unexpected season lengths
        phase1_end = min(38, season_days)
        phase2_end = min(38 + 30, season_days)

        # Apply rates to the array
        rate[0:phase1_end] = rate_stage1
        rate[phase1_end:phase2_end] = rate_stage2
        rate[phase2_end:season_days] = rate_stage3

        # Safety clip against actuator limits
        rate = np.minimum(rate, self.ub_mm_per_day)

        return rate

    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        """Calculate the daily control action with dynamic environmental checks."""

        # 1. Calculate Redistributed Rate
        # Spread any saved water evenly across all remaining days
        days_remaining = max(self.season_days - day, 1)
        redistributed_rate = self.water_saved / days_remaining

        # 2. Determine Target Rate
        # Baseline scheduled rate + our bonus saved water
        preplanned_rate = float(self._daily_rate[day])
        target_rate = preplanned_rate + redistributed_rate

        # Cap the target rate at the hardware limit
        target_rate = max(min(target_rate, self.ub_mm_per_day), 0.0)

        # 3. Tiered Rain Multiplier Logic
        rain = climate_today.get('rainfall', 0.0)
        if rain >= 10.0:
            rain_multiplier = 0.0   # Cancel entirely
        elif rain >= 5.0:
            rain_multiplier = 0.5   # Cut by half
        else:
            rain_multiplier = 1.0   # Proceed normally

        # 4. Apply Multiplier and Global Budget Safety
        per_agent = target_rate * rain_multiplier
        per_agent = max(min(per_agent, budget_remaining), 0.0)

        # 5. Agent-Level Ponding Override
        # Build the baseline array for all N agents
        u = np.full(self._N, per_agent, dtype=float)

        # Zero out irrigation locally for patches that are heavily flooded
        if 'x5' in state:
            excess_ponding_mask = state['x5'] > self.ponding_threshold_mm
            u[excess_ponding_mask] = 0.0

        # 6. Update the Saved Water Ledger
        # If actual_spent is less than preplanned, water_saved grows.
        # If actual_spent is more than preplanned, water_saved shrinks.
        actual_spent_avg = float(np.mean(u))
        self.water_saved += (preplanned_rate - actual_spent_avg)

        return u
