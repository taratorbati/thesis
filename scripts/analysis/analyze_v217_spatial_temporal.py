# analyze_v217_spatial_temporal.py
# Local read-only analysis of v2.17-P3 eval rollouts. No training, no torch.
# Requires: pandas, numpy, pyarrow.  Run from repo root: python analyze_v217_spatial_temporal.py
import pandas as pd, numpy as np, glob, sys, os, json

sys.path.insert(0, '.')
from src.terrain import load_terrain

FC = 140.0
t = load_terrain('gilan_farm.tif')
N = t['N']
elev = np.asarray(t['elevation_flat'], float)[:N]
nri  = np.array([t['Nr_internal'].get(i, 0) for i in range(N)], float)

DIR = 'results/runs/sac_v217_best_model'

def cell_path(scen, bud, forecast='perfect'):
    pref = 'sac_perfect_det' if forecast == 'perfect' else 'sac_noisy_ns42_det'
    c = glob.glob(f'{DIR}/{pref}_{scen}_*{bud}*.parquet')
    return c[0] if c else None

def analyze(scen, bud):
    p = cell_path(scen, bud)
    if not p: return None
    df = pd.read_parquet(p)

    # ---- TEMPORAL: field-mean action vs rain, on free-budget days ----
    g = df.groupby('day').agg(u=('u','mean'), rain=('rainfall','first'),
                              x1=('x1','median'), brem=('budget_remaining','first')).reset_index().sort_values('day')
    u = g['u'].values; rain = g['rain'].values; D = len(u)
    fwd7 = np.array([rain[d+1:d+8].sum() for d in range(D)])
    nextr = np.concatenate([rain[1:], [rain[-1]]])
    free = g['brem'].values >= 6.0
    def corr(a, b, m):
        a, b = np.asarray(a)[m], np.asarray(b)[m]
        if len(a) < 5 or np.std(a) < 1e-9 or np.std(b) < 1e-9: return float('nan')
        return float(np.corrcoef(a, b)[0,1])

    # ---- SPATIAL: per-agent season behavior vs terrain + outcome uniformity ----
    ua  = df.groupby('agent')['u'].mean().reindex(range(N)).values
    x1a = df.groupby('agent')['x1'].mean().reindex(range(N)).values
    def c2(a, b):
        return float(np.corrcoef(a, b)[0,1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else float('nan')

    return dict(
        # temporal
        corr_u_today  = corr(u, rain, free),
        corr_u_fwd7   = corr(u, fwd7, free),
        corr_u_next   = corr(u, nextr, free),
        u_mean        = float(df['u'].mean()),
        temporal_std  = float(np.std(u)),
        # spatial
        corr_u_elev   = c2(ua, elev),
        corr_u_nri    = c2(ua, nri),
        corr_x1_elev  = c2(x1a, elev),     # near 0 = elevation-balanced soil moisture (GOAL)
        x1_spatial_std= float(np.nanstd(x1a)),  # lower = more uniform soil moisture (GOAL)
        spatial_std_u = float(df.groupby('day')['u'].std().mean()),
    )

scen_order = ['dry','moderate','wet']; bud_order = ['100pct','85pct','70pct']
print(f"{'scenario':15s} | {'cT':>5s} {'cFwd7':>5s} {'cNext':>5s} | {'uMean':>5s} {'tStd':>4s} | "
      f"{'cU_elev':>7s} {'cU_nri':>6s} | {'cX1_elev':>8s} {'x1std':>5s} {'spStd':>5s}")
print('-'*100)
for s in scen_order:
    for b in bud_order:
        r = analyze(s, b)
        if not r: continue
        print(f"{s+'/'+b:15s} | {r['corr_u_today']:5.2f} {r['corr_u_fwd7']:5.2f} {r['corr_u_next']:5.2f} | "
              f"{r['u_mean']:5.2f} {r['temporal_std']:4.2f} | "
              f"{r['corr_u_elev']:7.2f} {r['corr_u_nri']:6.2f} | "
              f"{r['corr_x1_elev']:8.2f} {r['x1_spatial_std']:5.2f} {r['spatial_std_u']:5.3f}")

print("\nKEY:")
print("  cT/cFwd7/cNext = corr(action, rain today / next-7-day / next-day), free-budget days only")
print("                   (negative = correctly cuts water around rain = good temporal control)")
print("  cU_elev  = corr(applied water, elevation). POSITIVE means watering high cells more")
print("             (physically expected for a cascade: high feeds low).")
print("  cX1_elev = corr(RESULTING soil moisture, elevation). NEAR ZERO is the goal")
print("             (means every elevation ends near the same x1 = spatially balanced).")
print("  x1std    = std of per-cell mean soil moisture (mm). LOWER = more uniform = better.")