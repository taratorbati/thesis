# =============================================================================
# run_plots.py
# Runs all plot scripts. Requires preprocess.py to have been run first.
# =============================================================================

import subprocess
import sys

plots = [
    'plot_temperature.py',
    'plot_rainfall.py',
    'plot_radiation.py',
    'plot_humidity.py',
    'plot_wind.py',
    'plot_pressure.py',
    'plot_et0.py',
    'plot_soil_moisture.py',
]

for script in plots:
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {script}")
        print(result.stderr)
    else:
        print(result.stdout.strip())

print("\nAll plots complete.")
