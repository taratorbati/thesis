import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Constants ---
K = 19
season_length = 93
W_total = 484
u_max = 12
event_starts = np.arange(0, 91, 5)

# --- 2. Calculate Allocations ---
j = np.arange(1, K + 1)
w = (2 * (K - j + 1)) / (K * (K + 1))
delta = np.diff(np.append(event_starts, season_length))

# --- 3. Compute Schedule ---
u_fixed_100 = np.zeros(season_length)
for idx in range(K):
    start_day = event_starts[idx]
    end_day = start_day + delta[idx]
    daily_depth = (w[idx] * W_total) / delta[idx]
    u_fixed_100[start_day:end_day] = min(daily_depth, u_max)

# --- 4. Plotting (The Discrete Stem Plot) ---
days = np.arange(season_length)

plt.figure(figsize=(12, 6))

# Use plt.stem for the discrete visual
# basefmt=" " removes the baseline at y=0 to match the clean look of your image
markerline, stemlines, baseline = plt.stem(
    days, u_fixed_100, 
    basefmt=" ", 
    linefmt='#A0C4DF', # Light blue for the stems
    markerfmt='o',      # Circles for the markers
    label='$u_{fixed}(k)$ with full budget'
)


# Adjust marker and stem sizes
plt.setp(stemlines, 'linewidth', 2)
plt.setp(markerline, 'markersize', 4, 'markerfacecolor', '#1f77b4', 'markeredgecolor', '#1f77b4')



# A step plot with where='post' aligns the horizontal segments from the start of the day
#plt.step(days, u_fixed, where='post', linewidth=2.5, color='#0072B2', label='$u_{fixed}(k)$')

# Add the u_max threshold line for visual reference
plt.axhline(u_max, color='#D55E00', linestyle='--', alpha=0.8, label='$u_{max}$ Cap (12 mm/day)')

# Chart Formatting
plt.title('Front-Loaded Linear-Decay Irrigation Schedule ($u_{fixed}$', fontsize=14, pad=15)
plt.xlabel('Day of Season ($k$)', fontsize=12)
plt.ylabel('Daily Application Depth (mm/day)', fontsize=12)

# Adjust axes limits for padding
plt.xlim(-2, season_length + 2)
plt.ylim(0, u_max + 1)

# Set x-ticks
plt.xticks(np.arange(0, season_length + 1, 5))

# Add very light horizontal gridlines like the reference image
plt.grid(axis='y', linestyle='-', color='#E5E5E5', alpha=0.7)

# Clean up spines (borders) for a minimalist aesthetic
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False) 

plt.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.show()

# --- 5. Optional Verification (Matches your text perfectly) ---
print(f"1st event daily application: {u_fixed[0]:.2f} mm/day")
print(f"2nd event daily application: {u_fixed[5]:.2f} mm/day")
print(f"18th event daily application: {u_fixed[85]:.2f} mm/day")
print(f"19th (final) event daily application: {u_fixed[90]:.2f} mm/day")