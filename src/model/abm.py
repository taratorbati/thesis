# =============================================================================
# abm.py
# Agent-based crop-soil model based on Lopez-Jimenez et al. (2024).
#
# Corrections from baseline:
#   1. Mass conservation: phi4_out added to water balance (Eq 1a)
#   2. Drought stress h3 multiplies biomass increment (Eq 1d)
#   3. Surface runoff redistribution (Approach A or B)
#   4. x2_init from nursery, x4_init for transplant biomass
#   5. Calibrated RUE (theta13) — x4 in g/m² total dry matter
#
# Runoff modes:
#   'none'    — original: runoff leaves the system
#   'simple'  — Approach A: redistribute to lower neighbors, instant infiltration
#   'cascade' — Approach B: top-to-bottom processing, SCS re-applied at each level
# =============================================================================

import numpy as np


class CropSoilABM:
    def __init__(self, gamma_flat, sends_to, Nr, theta, N,
                 runoff_mode='none', elevation=None):
        self.gamma = gamma_flat
        self.sends_to = sends_to
        self.Nr = Nr
        self.theta = theta
        self.N = N
        self.runoff_mode = runoff_mode
        self.elevation = elevation

        # For cascade mode: sort agents from highest to lowest elevation
        if elevation is not None:
            self.sorted_agents = np.argsort(-elevation)  # descending
        else:
            self.sorted_agents = np.arange(N)

    def reset(self):
        t = self.theta
        self.x1 = np.full(self.N, t['theta6'] * t['theta5'])
        self.x2 = np.full(self.N, t.get('x2_init', 0.0))
        self.x3 = np.zeros(self.N)
        self.x4 = np.full(self.N, t.get('x4_init', 0.0))
        self.x5 = np.zeros(self.N)
        return self._get_state()

    def step(self, u, climate):
        rain = float(climate['rainfall'])
        ET = float(climate['ET'])
        Tmean = float(climate['temp_mean'])
        Tmax = float(climate['temp_max'])
        rad = float(climate['radiation'])

        # Crop ET calculation
        Kc = self.theta.get('Kc', 1.0)
        ETc = ET * Kc

        # Transpiration
        phi1 = self._transpiration(ETc)

        # Surface Hydrology & Routing
        W_surf = self.x5 + rain + u
        phi2 = np.zeros(self.N)
        I = np.zeros(self.N)
        phi2_in = np.zeros(self.N)

        theta3 = self.theta['theta3']
        theta_sat = self.theta.get('theta_sat', 0.45)
        theta5 = self.theta['theta5']
        I_max_cap = theta_sat * theta5

        if self.runoff_mode == 'cascade':
            # Approach B: process top-to-bottom in the same day
            for n in self.sorted_agents:
                # Sinks (Nr=0) cannot generate runoff
                if self.Nr[n] == 0 or W_surf[n] <= theta3:
                    phi2[n] = 0.0
                else:
                    phi2[n] = ((W_surf[n] - theta3)**2) / \
                        (W_surf[n] + 4 * theta3)

                # Send runoff immediately to lower neighbors' W_surf
                if phi2[n] > 0 and self.Nr[n] > 0:
                    per_neighbor = phi2[n] / self.Nr[n]
                    for m in self.sends_to[n]:
                        W_surf[m] += per_neighbor

                # Infiltration for agent n
                I_max = max(I_max_cap - self.x1[n] + phi1[n], 0.0)
                I[n] = min(W_surf[n] - phi2[n], I_max)

            # Surface Ponding for tomorrow (phi2_in is already inside W_surf)
            x5_new = W_surf - phi2 - I

        else:
            # Approach A ('simple' or 'none'): calculate phi2 independently
            # Sinks (Nr=0) cannot generate runoff
            phi2 = np.where((self.Nr == 0) | (W_surf <= theta3), 0.0,
                            ((W_surf - theta3)**2) / (W_surf + 4 * theta3))

            if self.runoff_mode == 'simple':
                # Delayed routing: save incoming runoff for tomorrow's state
                for n in range(self.N):
                    if phi2[n] > 0 and self.Nr[n] > 0:
                        per_neighbor = phi2[n] / self.Nr[n]
                        for m in self.sends_to[n]:
                            phi2_in[m] += per_neighbor

            # Infiltration
            I_max = np.maximum(I_max_cap - self.x1 + phi1, 0.0)
            I = np.minimum(W_surf - phi2, I_max)

            # Surface Ponding for tomorrow
            x5_new = W_surf - phi2 - I + phi2_in

        # Subsurface Hydrology
        x1_temp = self.x1 + I - phi1
        fc_total = self.theta['theta6'] * self.theta['theta5']

        E_sub = np.maximum(x1_temp - fc_total, 0.0)
        phi3 = self.theta['theta4'] * E_sub

        x1_new = x1_temp - phi3

        # Stress and Growth
        h1 = self._mean_temp(Tmean)
        h2 = self._heat_stress(Tmax)
        h3 = self._drought_stress(phi1, ETc)
        h6 = self._waterlog_stress()
        h7 = self._low_temp_stress(Tmean)
        g = self._growth_function()

        x2_new = self.x2 + h1

        x3_new = (self.x3
                  + self.theta['theta11'] * (1 - h2)
                  + self.theta['theta12'] * (1 - h3))

        x4_new = (self.x4
                  + self.theta['theta13'] * h3 * h6 * h7 * g * rad)

        # Update states
        self.x1 = np.clip(x1_new, 0, None)
        self.x2 = x2_new
        self.x3 = x3_new
        self.x4 = np.clip(x4_new, 0, None)
        self.x5 = np.clip(x5_new, 0, None)

        return self._get_state()

    def _get_state(self):
        return {
            'x1': self.x1.copy(),
            'x2': self.x2.copy(),
            'x3': self.x3.copy(),
            'x4': self.x4.copy(),
            'x5': self.x5.copy()
        }

    # ── Water balance & Stress components ──────────────────────────────────────────

    def _transpiration(self, ETc):
        t = self.theta
        demand = t['theta1'] * (self.x1 - t['theta2'] * t['theta5'])
        return np.minimum(np.maximum(demand, 0), ETc)

    def _mean_temp(self, Tmean):
        t = self.theta
        h1 = max(Tmean - t['theta7'], 0.0)
        return np.full(self.N, h1)

    def _heat_stress(self, Tmax):
        t = self.theta
        if Tmax <= t['theta9']:
            h2 = 1.0
        elif Tmax <= t['theta10']:
            h2 = 1.0 - (Tmax - t['theta9']) / (t['theta10'] - t['theta9'])
        else:
            h2 = 0.0
        return np.full(self.N, h2)

    def _drought_stress(self, phi1, ETc):
        t = self.theta
        h4 = np.where(
            phi1 < ETc,
            1.0 - phi1 / np.maximum(ETc, 1e-6),
            0.0
        )
        return 1.0 - t['theta14'] * h4

    def _waterlog_stress(self):
        t = self.theta
        fc_total = t['theta6'] * t['theta5']
        excess_ratio = (self.x1 - fc_total) / np.maximum(fc_total, 1e-6)
        return np.clip(1.0 - excess_ratio, 0.0, 1.0)

    def _low_temp_stress(self, Tmean):
        t = self.theta
        h7 = 1.0 if Tmean > t['theta7'] else 0.0
        return np.full(self.N, h7)

    def _growth_function(self):
        t = self.theta
        return np.where(
            self.x2 <= t['theta18'] / 2,
            t['theta19'] / (1 + np.exp(-0.01 * (self.x2 - t['theta20']))),
            t['theta19'] /
            (1 + np.exp(0.01 * (self.x2 + self.x3 - t['theta18'])))
        )
