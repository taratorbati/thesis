# =============================================================================
# soil_data.py
# Soil and crop parameters for Rice and Tobacco on the Gilan field.
# Soil type: silty loam (FAO Irrigation and Drainage Paper No. 56)
#
# Parameters confirmed against:
#   - FAO-56 (Kc, p, root depth)
#   - Sadidi Shal et al. (2021) — Hashemi rice GDD in Gilan
#   - Paredes et al. (2025) — base temperature review
#   - FAO Crop Information (tobacco water requirements)
#   - 25-year NASA POWER climate analysis (season windows, GDD calibration)
#
# Biomass calibration:
#   - Rice RUE: 1.3 g/MJ intercepted PAR (literature range 0.91–2.76)
#     → 0.65 g/MJ incident solar (PAR ≈ 50% of total solar)
#   - Tobacco RUE: 1.5 g/MJ intercepted PAR (Ferrara et al. 2002)
#     → 0.30 g/MJ incident solar (adjusted for leaf-quality management)
#   - x4 units: g/m² total dry matter
#   - Yield (kg/ha) = x4 × HI × 10
# =============================================================================

# ── Shared soil parameters (silty loam, same for all crops) ───────────────────
# Source: FAO-56 Table 19 (silty loam)

SOIL = {
    'theta1': 0.096,   # water uptake coefficient (dimensionless)
    'theta2': 0.15,    # wilting point (volumetric fraction) | FAO-56 Table 19
    'theta3': 10.0,    # initial abstraction for SCS runoff (mm)
    'theta4': 0.05,    # drainage coefficient (dimensionless)
    'theta6': 0.35,    # field capacity (volumetric fraction) | FAO-56 Table 19
    'theta_sat': 0.55  # Volumetric saturation for Silty Loam
}

# ── Rice parameters (Hashemi cultivar, Gilan) ─────────────────────────────────
# Season: May 20 (DOY 141) to August 20 (DOY 233), 93 field days
# Nursery: 20 days at ~15°C ambient + 5°C greenhouse = ~210 GDD
# Maturity: 1300 GDD total (Sadidi Shal et al. 2021: 1216–1352 GDD)
# FAO Kc: 1.15 (season average) | FAO-56 Table 12
# FAO p: 0.20 (very stress-sensitive) | FAO-56 Table 22

RICE = {
    **SOIL,
    'theta5':  400.0,   # root zone depth (mm) | FAO-56 Table 22 (paddy rice)
    'theta7':  10.0,    # base temperature (°C) | standard GDD convention
    'theta9':  35.0,    # heat stress onset temperature (°C)
    'theta10': 42.0,    # extreme heat threshold (°C)
    'theta11': 0.0030,  # heat stress effect on maturity
    'theta12': 0.0030,  # drought stress effect on maturity
    # radiation use efficiency (g DM/MJ incident solar) | calibrated
    'theta13': 0.65,
    # drought sensitivity on RUE (high — rice is very sensitive)
    'theta14': 0.8,
    # harvest index (grain/total biomass) | Hashemi cultivar
    'HI':      0.42,
    # cumulative temp to maturity (°C·day) | Sadidi Shal et al. 2021
    'theta18': 1250.0,
    'theta19': 0.95,    # max radiation interception fraction
    # cumulative temp for 50% interception (proportional: 1250/1800*600)
    'theta20': 417.0,
    'Kc':      1.15,    # season-average crop coefficient | FAO-56 Table 12
    'p':       0.20,    # depletion fraction before stress | FAO-56 Table 22
    'name':    'Rice',
    'season_start_doy': 141,  # May 20
    'season_end_doy':   233,  # August 20
    'season_days':      93,
    'x2_init': 210.0,   # nursery GDD (20 days, ambient + 5°C greenhouse boost)
    # initial biomass (g/m²) — 25 hills/m² × ~2.5 g/seedling
    'x4_init': 60.0,
}

# ── Tobacco parameters ────────────────────────────────────────────────────────
# Season: May 25 (DOY 146) to September 5 (DOY 249), 104 field days
# No nursery greenhouse — seedlings raised in open seedbeds
# Maturity: 1200 GDD (calibrated from 25-year data, validated against FAO 100–130 day duration)
# FAO Kc: 0.90 (season average)
# FAO p: 0.50 | FAO Crop Information (tobacco)

TOBACCO = {
    **SOIL,
    'theta5':  700.0,   # root zone depth (mm) | FAO: 0.5–1.0 m, mid-range
    'theta7':  10.0,    # base temperature (°C) | consistent with rice
    'theta9':  30.0,    # heat stress onset (°C) | FAO: optimal ceiling 30°C
    'theta10': 38.0,    # extreme heat threshold (°C)
    'theta11': 0.0025,  # heat stress effect on maturity
    # drought stress effect on maturity (slightly more sensitive)
    'theta12': 0.0028,
    # radiation use efficiency (g DM/MJ incident solar) | calibrated
    'theta13': 0.30,
    'theta14': 0.6,     # drought sensitivity on RUE (moderate)
    # harvest index (leaf/total biomass) | tobacco leaf fraction
    'HI':      0.55,
    'theta18': 1200.0,  # cumulative temp to maturity (°C·day) | calibrated
    'theta19': 0.90,    # max radiation interception fraction
    'theta20': 450.0,   # cumulative temp for 50% interception | literature: ~40 DAT
    'Kc':      0.90,    # season-average crop coefficient | FAO
    'p':       0.50,    # depletion fraction before stress | FAO Crop Information
    'name':    'Tobacco',
    'season_start_doy': 146,  # May 25
    'season_end_doy':   249,  # September 5
    'season_days':      104,
    'x2_init': 0.0,     # no nursery data — field GDD starts at transplanting
    'x4_init': 8.0,     # initial biomass (g/m²) — 2 plants/m² × ~4 g/seedling
}

# ── Active crop selection ─────────────────────────────────────────────────────
# Change this line to switch crops: RICE or TOBACCO
theta = RICE
