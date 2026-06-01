import subprocess, sys
r = subprocess.run([
    sys.executable, '-m', 'scripts.experiments.exp_rl',
    '--mode', 'eval',
    '--model', r'C:\Users\Tara\Documents\Thesis\results\rl\sac_v217_p3_seed0_20260601_010936\best_model\best_model.zip',
    '--scenario', 'all', '--budget', 'all', '--forecast', 'perfect',
], capture_output=True, text=True)
print("RETURN CODE:", r.returncode)
print("=== STDOUT (last 2000) ===\n", r.stdout[-2000:])
print("=== STDERR (last 3000) ===\n", r.stderr[-3000:])