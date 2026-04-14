import numpy as np


class CropSoilABM:
    def __init__(self, gamma_flat, sends_to, Nr, theta, N):
        self.gamma = gamma_flat
        self.sends_to = sends_to
        self.Nr = Nr
        self.theta = theta
        self.N = N

    def reset(self):
        t = self.theta
        self.x1 = np.full(self.N, t['theta6'] * t['theta5'] * 0.6)
        self.x2 = np.zeros(self.N)
        self.x3 = np.zeros(self.N)
        self.x4 = np.zeros(self.N)
        return self._get_state()

    def step(self, u, climate):
        rain = float(climate['rainfall'])
        ET = float(climate['ET'])
        Tmean = float(climate['temp_mean'])
        Tmax = float(climate['temp_max'])
        rad = float(climate['radiation'])

        phi1 = self._transpiration(ET)
        phi2 = self._surface_runoff(rain)
        phi3 = self._deep_drainage(rain, phi2)
        phi4_in, phi4_out = self._water_exchange()
        h1 = self._mean_temp(Tmean)
        h2 = self._heat_stress(Tmax)
        h3 = self._drought_stress(phi1, ET)
        h6 = self._waterlog_stress()
        h7 = self._low_temp_stress(Tmean)
        g = self._growth_function()

        x1_new = (self.x1
                  - phi1 - phi2 - phi3
                  - phi4_out + phi4_in
                  + rain + u)

        x2_new = self.x2 + h1

        x3_new = (self.x3
                  + self.theta['theta11'] * (1 - h2)
                  + self.theta['theta12'] * (1 - h3))

        # h3 multiplies biomass increment: drought reduces radiation use
        # efficiency (FAO AquaCrop formulation). This is the key fix that
        # makes irrigation matter for yield outcomes.
        # Minimum h3 = 1 - theta14*1.0 = 1 - 0.8 = 0.2 (for rice)
        # So even maximum drought reduces biomass by at most 80%, not 100%.
        x4_new = (self.x4
                  + self.theta['theta13'] * h3 * h6 * h7 * g * rad)

        self.x1 = np.clip(x1_new, 0, None)
        self.x2 = x2_new
        self.x3 = x3_new
        self.x4 = np.clip(x4_new, 0, None)

        return self._get_state()

    def _get_state(self):
        return {
            'x1': self.x1.copy(),
            'x2': self.x2.copy(),
            'x3': self.x3.copy(),
            'x4': self.x4.copy()
        }

    def _transpiration(self, ET):
        t = self.theta
        demand = t['theta1'] * (self.x1 - t['theta2'] * t['theta5'])
        return np.minimum(np.maximum(demand, 0), ET)

    def _surface_runoff(self, rain):
        t = self.theta
        if rain > t['theta3']:
            phi2 = (rain - t['theta3'])**2 / (rain + 4 * t['theta3'])
        else:
            phi2 = 0.0
        return np.full(self.N, phi2)

    def _deep_drainage(self, rain, phi2):
        t = self.theta
        fc_total = t['theta6'] * t['theta5']
        excess = self.x1 + rain - phi2 - fc_total
        return np.where(excess > 0, t['theta4'] * excess, 0.0)

    def _water_exchange(self):
        t = self.theta
        fc_total = t['theta6'] * t['theta5']
        phi4_in = np.zeros(self.N)
        phi4_out = np.zeros(self.N)
        for n in range(self.N):
            excess = self.x1[n] - fc_total
            if excess > 0 and self.Nr[n] > 0:
                outflow_per_neighbor = excess / self.Nr[n]
                phi4_out[n] += excess
                for m in self.sends_to[n]:
                    phi4_in[m] += outflow_per_neighbor
        return phi4_in, phi4_out

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

    def _drought_stress(self, phi1, ET):
        t = self.theta
        h4 = np.where(
            phi1 < ET,
            1.0 - phi1 / np.maximum(ET, 1e-6),
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
        # ORIGINAL formulation from Lopez-Jimenez et al. (2024).
        # Uses x2 (thermal time) to drive the sigmoid — x2 accumulates
        # ~10 deg/day for rice (base temp 10C) reaching ~1200 over 120 days.
        # x3 appears only in the SECOND branch (post-maturity senescence).
        # DO NOT replace x2 with x3 in the first branch — x3 only reaches
        # ~0.7 over the season and would freeze the sigmoid at its minimum.
        t = self.theta
        return np.where(
            self.x2 <= t['theta18'] / 2,
            t['theta19'] / (1 + np.exp(-0.01 * (self.x2 - t['theta20']))),
            t['theta19'] /
            (1 + np.exp(0.01 * (self.x2 + self.x3 - t['theta18'])))
        )
