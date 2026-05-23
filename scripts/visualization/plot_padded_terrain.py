import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import rasterio

def plot_padded_3d_dem(dem_path='gilan_farm.tif'):
    # 1. Load the original 10x13 farm elevation
    with rasterio.open(dem_path) as src:
        base_elev = src.read(1)[0:10, 0:13]  # Ensure it's exactly 10x13
        
    rows, cols = base_elev.shape

    # 2. Mathematically pad the grid (1 cell on each side)
    # Using reflect_type='odd' perfectly continues the existing slope vector!
    padded_elev = np.pad(base_elev, pad_width=1, mode='reflect', reflect_type='odd')
    
    p_rows, p_cols = padded_elev.shape

    # 3. Create the X, Y meshgrid for the 3D plot
    X, Y = np.meshgrid(np.arange(p_cols), np.arange(p_rows))

    # 4. Set up the matplotlib 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the full 12x15 padded surface
    surf = ax.plot_surface(X, Y, padded_elev, cmap=cm.terrain, 
                           edgecolor='black', linewidth=0.2, alpha=0.8)

    # 5. Highlight the boundary of the ACTUAL farm (the 10x13 interior)
    # This draws a red box around the playable area
    x_farm = [1, cols, cols, 1, 1]
    y_farm = [1, 1, rows, rows, 1]
    z_farm = [padded_elev[y, x] + 0.5 for x, y in zip(x_farm, y_farm)] # Lifted slightly so it's visible
    
    ax.plot(x_farm, y_farm, z_farm, color='red', linewidth=3, label='Actual Farm Boundary (130 agents)')

    # Formatting
    ax.set_title('3D Topographical Map with Off-Farm Drainage Padding', fontsize=14, fontweight='bold')
    ax.set_xlabel('Columns (West to East)')
    ax.set_ylabel('Rows (North to South)')
    ax.set_zlabel('Elevation (meters)')
    
    # Invert Y axis so Row 0 is at the top (standard GIS mapping)
    ax.invert_yaxis()
    
    # Set optimal viewing angle
    ax.view_init(elev=35, azim=-120)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_padded_3d_dem()