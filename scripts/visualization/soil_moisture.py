import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# 1. Base directory where the results are stored
# rglob will search this directory and all subdirectories for parquet files
BASE_DIR = Path('results/runs/td3_final/A')

def safe_mean(val):
    """
    Safely calculates the mean of the soil moisture across all 130 agents.
    """
    try:
        if isinstance(val, (list, np.ndarray)):
            return np.mean(val)
        elif isinstance(val, str):
            clean_str = val.replace('[', '').replace(']', '').replace('\n', ' ').split()
            return np.mean([float(x) for x in clean_str])
        return float(val)
    except Exception:
        return np.nan

def parse_filename(filename):
    """
    Extracts experiment parameters from Tara's filename convention.
    Example: sac_noisy_ns42_det_dry_rice_70pct_seed1.parquet
    """
    name = filename.name.lower()
    
    # Forecast Type
    forecast = 'perfect' if 'perfect' in name else 'noisy'
    
    # Scenario
    scenario = 'unknown'
    if 'dry' in name: scenario = 'dry'
    elif 'moderate' in name: scenario = 'moderate'
    elif 'wet' in name: scenario = 'wet'
        
    # Budget
    budget = 'unknown'
    budget_match = re.search(r'(\d+)pct', name)
    if budget_match: 
        budget = budget_match.group(1)
        
    # Seed
    seed = 'unknown'
    seed_match = re.search(r'seed(\d+)', name)
    if seed_match: 
        seed = seed_match.group(1)
        
    return forecast, scenario, budget, seed

def main():
    print(f"Searching for .parquet files in {BASE_DIR.absolute()}...")
    files = list(BASE_DIR.rglob('*.parquet'))
    
    # Fallback to current directory if not found in results/runs
    if not files:
        print("No files found in 'results/runs', searching current directory...")
        files = list(Path('.').rglob('*.parquet'))
        
    if not files:
        print("Error: No .parquet files found!")
        return
        
    print(f"Found {len(files)} files. Processing data...")
    
    all_data = []
    
    # Process each file
    for f in files:
        forecast, scenario, budget, seed = parse_filename(f)
        
        # Skip files that don't match our expected naming convention
        if scenario == 'unknown' or budget == 'unknown':
            continue
            
        try:
            df = pd.read_parquet(f, engine='pyarrow')
            if 'x1' not in df.columns:
                continue
                
            # Average across agents for this specific file
            df['x1_avg'] = df['x1'].apply(safe_mean)
            
            # Tag the rows with the experiment metadata
            df['forecast'] = forecast
            df['scenario'] = scenario
            df['budget'] = budget
            df['seed'] = seed
            
            all_data.append(df[['day', 'x1_avg', 'forecast', 'scenario', 'budget', 'seed']])
        except Exception as e:
            print(f"Failed to read {f.name}: {e}")
            
    # Combine everything into one massive DataFrame
    master_df = pd.concat(all_data, ignore_index=True)
    
    print("Aggregating data across seeds...")
    # Group by the parameters (ignoring seed) and calculate the mean and std dev for each day
    agg_df = master_df.groupby(['scenario', 'budget', 'forecast', 'day'])['x1_avg'].agg(['mean', 'std']).reset_index()
    
    # 2. Plotting Setup: 3x3 Grid
    scenarios = ['dry', 'moderate', 'wet']
    budgets = ['70', '85', '100']
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    #fig.suptitle('Average Soil Moisture Profiles Across All Scenarios', 
    #             fontsize=18, fontweight='bold', y=0.98)
    
    # Colors for the forecast types
    colors = {'perfect': '#1f77b4', 'noisy': '#ff7f0e'}
    
    for i, sc in enumerate(scenarios):
        for j, bg in enumerate(budgets):
            ax = axes[i, j]
            
            # Filter data for this specific subplot
            sub_data = agg_df[(agg_df['scenario'] == sc) & (agg_df['budget'] == bg)]
            
            for forecast_type in ['perfect', 'noisy']:
                plot_data = sub_data[sub_data['forecast'] == forecast_type].sort_values('day')
                
                if not plot_data.empty:
                    # Plot the Mean Line
                    ax.plot(plot_data['day'], plot_data['mean'], 
                            color=colors[forecast_type], 
                            linewidth=2.0, 
                            label=f'{forecast_type.capitalize()} Forecast')
                    
                    # Add the Confidence Band (Mean ± Std)
                    ax.fill_between(plot_data['day'], 
                                    plot_data['mean'] - plot_data['std'], 
                                    plot_data['mean'] + plot_data['std'], 
                                    color=colors[forecast_type], 
                                    alpha=0.2)
            
            # Subplot Formatting
            if i == 0:
                ax.set_title(f'{bg}% Budget', fontsize=14, fontweight='bold', pad=10)
            if j == 0:
                ax.set_ylabel(f'{sc.capitalize()} Scenario\nMoisture (mm)', fontsize=12, fontweight='bold', labelpad=10)
            if i == 2:
                ax.set_xlabel('Day of Season ($k$)', fontsize=12)
                
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add a legend only to the top right chart to avoid clutter
            if i == 0 and j == 2:
                ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    # Adjust layout to make room for the suptitle
    plt.subplots_adjust(top=0.90) 
    
    output_img = 'soil_moisture_3x3_grid.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Success! Master grid plot saved as '{output_img}'")
    plt.show()

if __name__ == "__main__":
    main()