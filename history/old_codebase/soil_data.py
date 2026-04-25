# =============================================================================
# soil_data.py
# Soil and crop parameters for three crops on the Gilan field.
# Soil type: silty loam (FAO Irrigation and Drainage Paper No. 56)
# =============================================================================

# ── Shared soil parameters (same for all crops) ───────────────────────────────
SOIL = {
    'theta1': 0.096,   # water uptake coefficient
    'theta2': 0.15,    # wilting point (volumetric fraction)
    'theta3': 10.0,    # initial abstraction (mm)
    'theta4': 0.05,    # drainage coefficient
    'theta5': 400.0,   # root zone depth (mm)
    'theta6': 0.35,    # field capacity (volumetric fraction)
}

# ── Wheat parameters ──────────────────────────────────────────────────────────
# FAO Kc season-average: 0.85 | Depletion fraction p: 0.55
WHEAT = {
    **SOIL,
    'theta7':  0.0,    # base temperature for phenology (°C)
    'theta9':  34.0,   # heat stress onset temperature (°C)
    'theta10': 40.0,   # extreme heat threshold (°C)
    'theta11': 0.0025,  # heat stress effect on maturity
    'theta12': 0.0025,  # drought stress effect on maturity
    'theta13': 1.24,   # radiation use efficiency (g/MJ)
    'theta14': 0.5,    # drought sensitivity on RUE
    'theta18': 2200.0,  # cumulative temp to maturity (°C·day)
    'theta19': 0.95,   # max radiation interception fraction
    'theta20': 700.0,  # cumulative temp for 50% interception
    # Crop-specific constants for budget calculation
    'Kc':      0.85,   # season-average crop coefficient (FAO)
    'p':       0.55,   # depletion fraction before stress (FAO)
    'name':    'Wheat',
}

# ── Tobacco parameters ────────────────────────────────────────────────────────
# FAO Kc season-average: 0.90 | Depletion fraction p: 0.50
TOBACCO = {
    **SOIL,
    'theta7':  8.0,    # base temperature (tobacco needs warmth)
    'theta9':  32.0,   # heat stress onset
    'theta10': 38.0,   # extreme heat threshold
    'theta11': 0.0025,
    'theta12': 0.0028,  # slightly more drought sensitive than wheat
    'theta13': 1.10,   # lower RUE than wheat
    'theta14': 0.6,    # more sensitive to drought
    'theta18': 1600.0,  # matures faster than wheat
    'theta19': 0.90,
    'theta20': 550.0,
    'Kc':      0.90,
    'p':       0.50,
    'name':    'Tobacco',
}

# ── Rice parameters ───────────────────────────────────────────────────────────
# FAO Kc season-average: 1.15 | Depletion fraction p: 0.20
# Rice is extremely water-sensitive — stress starts near field capacity
RICE = {
    **SOIL,
    'theta7':  10.0,   # base temperature (rice needs warmth to develop)
    'theta9':  35.0,   # heat stress onset
    'theta10': 42.0,   # extreme heat threshold
    'theta11': 0.0030,
    'theta12': 0.0030,
    'theta13': 1.20,   # RUE
    'theta14': 0.8,    # very high drought sensitivity
    'theta18': 1800.0,  # cumulative temp to maturity
    'theta19': 0.95,
    'theta20': 600.0,
    'Kc':      1.15,
    'p':       0.20,   # stress begins after only 20% depletion
    'name':    'Rice',
}

# ── Active crop selection ─────────────────────────────────────────────────────
# Change this line to switch crops: WHEAT, TOBACCO, or RICE
theta = RICE
