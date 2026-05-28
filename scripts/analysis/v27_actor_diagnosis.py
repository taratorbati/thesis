import os
import sys
import zipfile
import torch
import numpy as np
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#DEFAULT_CKPT_DIR = (PROJECT_ROOT / 'results' / 'rl' / 'sac_v27_seed0_20260518_011229' / 'checkpoints')
DEFAULT_CKPT_DIR = (PROJECT_ROOT / 'results' / 'rl' / 'sac_v27_seed1_20260518_202518' / 'checkpoints')



def main():
    ckpt_dir = DEFAULT_CKPT_DIR
    if not ckpt_dir.exists():
        ckpt_dir = Path.cwd() # Fallback

    print(f"\nScanning directory: {ckpt_dir}\n")
    print(f"{'Step':>8} | {'mu.weight std':>15} | {'mu.bias[0]':>12}")
    print("-" * 41)

    steps = [i * 50000 for i in range(1, 11)]

    for step in steps:
        filename = ckpt_dir / f"sac_general_seed1_{step}_steps.zip"
        
        if not filename.exists():
            print(f"{step:>8} | {'MISSING':>15} | {'MISSING':>12}")
            continue
            
        try:
            with zipfile.ZipFile(filename, 'r') as z:
                with z.open('policy.pth') as f:
                    # Read the raw dictionary directly, bypassing network graphs
                    state_dict = torch.load(f, map_location='cpu')
                    
                    # Your output head weights are [1, 128] and bias is [1]
                    mu_w = state_dict['actor.mu.weight'].numpy()
                    mu_b = state_dict['actor.mu.bias'].numpy()
                    
                    w_std = float(np.std(mu_w))
                    b_val = float(mu_b[0])
                    
                    print(f"{step:>8} | {w_std:15.5f} | {b_val:12.5f}")
                    
        except Exception as e:
            print(f"{step:>8} | Error: {e}")

    print("-" * 41)
    print("FRESH INIT REFERENCE: mu.weight std ≈ 0.0524")
    print("If it spikes ONLY at 200k, the cascade-shocked-it-awake theory is proven.\n")

if __name__ == '__main__':
    main()