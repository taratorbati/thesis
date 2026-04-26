# =============================================================================
# src/controllers/fixed_schedule.py
# Front-loaded fixed irrigation schedule with linear-decay weights.
#
# Reflects traditional Gilan rice farming: heavy irrigation early in the
# season (post-transplanting establishment), tapering off toward harvest.
#
# Schedule design:
#   - K = 19 irrigation events, on days {0, 5, 10, ..., 90}
#     (zero-indexed; event 0 on transplanting day, event 18 on day 90).
#   - Each event's allocation w_j = 2(K - j + 1) / (K(K+1)) for j = 1..K.
#     Weights sum to 1; w_1 ≈ 0.10, w_19 ≈ 0.005.
#   - Event j's total water is w_j × W_budget (in mm, field-averaged).
#   - Each event's water is spread evenly across the days from the event
#     until the next event (typically 5 days). The last event spreads
#     across remaining days through the end of the season.
#   - Daily-per-agent rate is clipped to the UB actuator cap. If clipping
#     occurs, the budget is under-utilized rather than violated.
# =============================================================================

import numpy as np

from src.controllers.base import Controller


# Default schedule parameters. Override via constructor if needed.
DEFAULT_NUM_EVENTS = 19
DEFAULT_EVENT_INTERVAL = 5    # days between event start days
DEFAULT_FIRST_EVENT_DAY = 0   # zero-indexed: day 0 = first day of season
DEFAULT_UB_MM_PER_DAY = 12.0  # actuator cap per agent per day


class FixedScheduleController(Controller):
    """Front-loaded linear-decay schedule.

    Parameters
    ----------
    num_events : int
        Number of irrigation events. Default 19.
    event_interval : int
        Days between successive event start days. Default 5.
    first_event_day : int
        Zero-indexed day of the first event. Default 0 (transplanting day).
    ub_mm_per_day : float
        Per-agent per-day actuator cap. Default 12 mm/day.
    """

    def __init__(self,
                 num_events=DEFAULT_NUM_EVENTS,
                 event_interval=DEFAULT_EVENT_INTERVAL,
                 first_event_day=DEFAULT_FIRST_EVENT_DAY,
                 ub_mm_per_day=DEFAULT_UB_MM_PER_DAY):
        super().__init__(name='fixed_schedule')
        self.num_events = int(num_events)
        self.event_interval = int(event_interval)
        self.first_event_day = int(first_event_day)
        self.ub_mm_per_day = float(ub_mm_per_day)

        # Filled in reset()
        self._N = None
        self._daily_rate = None  # shape (season_days,), mm/agent/day

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        self._N = terrain['N']
        self._daily_rate = self._build_daily_rate(season_days, budget_total)

    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        """Apply the precomputed daily rate, also clipped against budget_remaining.

        Two safety nets are applied here even though _build_daily_rate already
        respects the UB cap:
          1. budget_remaining clip: never apply more than what's left in the
             seasonal budget. Prevents budget overshoot if the precomputed
             schedule was built against a different (e.g. nominal) budget.
          2. UB clip: defensive; should be a no-op given _build_daily_rate.
        """
        rate = float(self._daily_rate[day])

        # Per-agent rate already includes the UB cap; we apply uniformly.
        per_agent = min(rate, self.ub_mm_per_day)

        # Budget safety: we have budget_remaining mm field-averaged for the
        # rest of the season; we are about to spend per_agent mm on every
        # one of N agents, which equals per_agent mm field-averaged. Clip
        # at budget_remaining (guard against floating-point drift).
        per_agent = max(min(per_agent, budget_remaining), 0.0)

        return np.full(self._N, per_agent, dtype=float)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_daily_rate(self, season_days, budget_total):
        """Precompute the per-day uniform-across-agents irrigation rate.

        Returns
        -------
        np.ndarray, shape (season_days,)
            Each entry is the per-agent per-day water depth in mm.
            The total over the season equals budget_total, except where
            UB clipping caused under-spending.
        """
        K = self.num_events
        weights = self._linear_decay_weights(K)

        # Event start days, possibly truncated to season_days
        event_days = [
            self.first_event_day + j * self.event_interval
            for j in range(K)
        ]
        event_days = [d for d in event_days if d < season_days]

        # If we lost trailing events because the season is shorter than expected,
        # renormalize the surviving weights so the budget still totals out.
        if len(event_days) < K:
            weights = weights[:len(event_days)]
            weights = weights / weights.sum()

        rate = np.zeros(season_days, dtype=float)

        for j, start in enumerate(event_days):
            # Span this event's water across days [start, next_event)
            if j + 1 < len(event_days):
                end = event_days[j + 1]
            else:
                end = season_days
            span = max(end - start, 1)

            event_total = weights[j] * budget_total
            daily = event_total / span

            # Per-agent UB clip
            daily = min(daily, self.ub_mm_per_day)

            rate[start:end] = daily

        return rate

    @staticmethod
    def _linear_decay_weights(K):
        """Return w_j = 2(K - j + 1) / (K(K+1)) for j = 1..K, summing to 1.

        First event gets the largest weight, last event the smallest.
        """
        if K <= 0:
            raise ValueError(f"num_events must be positive, got {K}")
        j = np.arange(1, K + 1)
        return (2.0 * (K - j + 1)) / (K * (K + 1))
