import matplotlib.patches as patches
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Load DEM ──────────────────────────────────────────────────────────────
file_path = 'gilan_farm.tif'
with rasterio.open(file_path) as dataset:
    elevation_matrix = dataset.read(1)
    nodata = dataset.nodata
    print(f"NoData value: {nodata}")
    print(f"Min elevation: {elevation_matrix.min()}")
    print(f"Max elevation: {elevation_matrix.max()}")

rows, cols = elevation_matrix.shape

# ── 2. 3D Surface Plot ───────────────────────────────────────────────────────
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, elevation_matrix, cmap='terrain',
                           edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,
                 label='Elevation (meters)')
    ax.set_title('3D Topographical Map of the 130-Agent Grid')
    ax.set_xlabel('Columns (X)')
    ax.set_ylabel('Rows (Y)')
    ax.set_zlabel('Elevation (meters)')
    ax.invert_yaxis()
    ax.view_init(elev=25, azim=-45)
    plt.savefig('results/topographical_map.png', dpi=150)
    plt.show()

# ── 3. Build Directed Graph ──────────────────────────────────────────────────


def build_directed_graph(elevation_matrix):
    rows, cols = elevation_matrix.shape

    gamma = (elevation_matrix - elevation_matrix.min()) / \
            (elevation_matrix.max() - elevation_matrix.min())

    sends_to = {}
    Nr = {}

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            lower_neighbors = []
            for dr, dc in directions:
                nr2, nc2 = r + dr, c + dc
                if 0 <= nr2 < rows and 0 <= nc2 < cols:
                    m = nr2 * cols + nc2
                    if gamma[nr2, nc2] < gamma[r, c]:
                        lower_neighbors.append(m)
            sends_to[n] = lower_neighbors
            Nr[n] = len(lower_neighbors)

    return gamma.flatten(), sends_to, Nr


# ── 4. Call the function — this was the missing step ────────────────────────
gamma_flat, sends_to, Nr = build_directed_graph(elevation_matrix)
gamma_2d = gamma_flat.reshape(rows, cols)

# ── 5. Sanity checks ─────────────────────────────────────────────────────────
lowest_agent = int(np.argmin(gamma_flat))
highest_agent = int(np.argmax(gamma_flat))

print(f"\nLowest agent:  {lowest_agent}")
print(f"  Sends water to: {sends_to[lowest_agent]}")
print(f"  Nr = {Nr[lowest_agent]}  (should be 0)")

print(f"\nHighest agent: {highest_agent}")
print(f"  Sends water to: {sends_to[highest_agent]}")
print(f"  Nr = {Nr[highest_agent]}  (should be > 0)")

# ── 6. Arrow Flow Visualization ──────────────────────────────────────────────
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(gamma_2d, cmap='terrain', origin='upper')
    plt.colorbar(im, label='Normalized Elevation (γ)')

    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            for m in sends_to[n]:
                nr2 = m // cols
                nc2 = m % cols
                ax.annotate("",
                            xy=(nc2, nr2),
                            xytext=(c, r),
                            arrowprops=dict(arrowstyle="->", color='blue', lw=0.8))

    ax.set_title('Water Flow Directions Between Agents')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    plt.tight_layout()
    plt.savefig('results/water_flow_directions.png', dpi=150)
    plt.show()

# ── 7. Sink Agent Visualization ──────────────────────────────────────────────

Nr_2d = np.array([Nr[n] for n in range(rows * cols)]).reshape(rows, cols)

if __name__ == '__main__':
    # --- Verification plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Nr_2d, cmap='Blues', origin='upper')
    plt.colorbar(im, label='Number of lower neighbors (Nr)')
    ax.set_title('Water Receiving Capacity — Red box = sink agents (Nr=0)')

    # Draw a clean red rectangle around each sink agent
    for r in range(rows):
        for c in range(cols):
            if Nr_2d[r, c] == 0:
                rect = patches.Rectangle(
                    (c - 0.5, r - 0.5),  # bottom-left corner of cell
                    1, 1,                  # width and height of one cell
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'       # transparent fill
                )
                ax.add_patch(rect)
                # Also print which agent it is
                n = r * cols + c
                print(f"Sink agent: n={n}, row={r}, col={c}, "
                      f"gamma={gamma_2d[r, c]:.3f}")

    plt.tight_layout()
    plt.savefig('results/water_receiving_capacity.png', dpi=150)
    plt.show()

# These variables are exported for use in main.py
__all__ = ['gamma_flat', 'sends_to', 'Nr', 'rows', 'cols']
